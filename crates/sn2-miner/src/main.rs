// `Ipv4Addr::is_global`, the same test validators apply to axon IPs.
#![feature(ip)]

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Mimalloc reads option env vars on first option access (lazy). The default
// `purge_delay` of ~1s churns the page tables under our proving workload's
// allocation frequency (witness/proof buffers allocated and freed across
// many concurrent blocking threads), the same single-thread allocator
// bottleneck the validator observed on its dispatch workload (38k
// mmap/munmap syscalls per 3s on mainnet before this fix). Setting the env
// var from a constructor that runs before main() (and crucially before the
// tokio runtime build) captures the desired cadence before any sustained
// allocation. Operators can still override by setting the env var
// explicitly in their process environment.
#[cfg(target_os = "linux")]
#[ctor::ctor]
fn configure_mimalloc_purge_delay() {
    // SAFETY: ctor runs single-threaded before main; no other thread can
    // race on environment state at this point.
    unsafe {
        if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
            std::env::set_var("MIMALLOC_PURGE_DELAY", "60000");
        }
    }
}

mod cli;
mod dsperse;
mod handlers;
mod lightning_server;
mod wai_known_constants;

use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches};
use sn2_chain::ServeAxonOutcome;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tracing::{debug, info, info_span, warn, Instrument, Span};

use crate::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls CryptoProvider");

    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    sn2_types::init_tracing(&cli.log_level);

    if !cli.miners.is_empty()
        && matches.value_source("wallet_hotkey") == Some(ValueSource::CommandLine)
        && !cli.miners.iter().any(|m| m.hotkey == cli.wallet_hotkey)
    {
        warn!(
            wallet_hotkey = %cli.wallet_hotkey,
            "--wallet-hotkey is ignored because --miner is set; add it as a --miner entry to keep serving it"
        );
    }

    info!(version = sn2_types::SOFTWARE_VERSION, "sn2-miner");

    if cli.loopback {
        return run_loopback(cli).await;
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    if !cli.no_auto_update && option_env!("SN2_RELEASE_CHANNEL") == Some("mainnet") {
        let _update_handle =
            sn2_chain::auto_update::spawn_update_loop("sn2-miner", shutdown_tx.clone());
    }

    info!(
        netuid = cli.netuid,
        network = %cli.network,
        "starting sn2-miner"
    );

    let miners = load_miners(&cli)?;
    // Checked before the circuit download so a bad override fails fast.
    let external_ip = Arc::new(ExternalIp::from_cli(cli.external_ip.as_deref())?);
    let endpoint =
        sn2_chain::resolve_endpoint(&cli.network, cli.subtensor_chain_endpoint.as_deref());

    let handlers = init_handlers(&cli, false).await;
    let servers = start_servers(&miners, "0.0.0.0", cli.handler_timeout, handlers, true).await?;

    // Each hotkey signs its own serve_axon and has its own serving rate limit,
    // so every miner keeps its axon registered in its own task, off the path
    // that watches the servers and signals. A hotkey that is not registered on
    // the subnet yet is served anyway; its task registers the axon as soon as
    // the hotkey has a uid.
    let registration = Arc::new(sn2_chain::Registration::new(cli.netuid));
    for miner in &miners {
        info!(
            hotkey = %miner.wallet.hotkey_ss58(),
            quic_port = miner.port,
            "miner running"
        );
        tokio::spawn(
            keep_axon_registered(AxonTask {
                registration: registration.clone(),
                endpoint: endpoint.clone(),
                external_ip: external_ip.clone(),
                wallet: miner.wallet.clone(),
                port: miner.port,
                netuid: cli.netuid,
                network: cli.network.clone(),
            })
            .instrument(miner.span()),
        );
    }

    run_until_shutdown(servers, Some(shutdown_rx)).await
}

async fn run_loopback(cli: Cli) -> Result<()> {
    info!("starting miner in loopback mode (no chain interaction)");

    let miners = load_miners(&cli)?;
    let handlers = init_handlers(&cli, true).await;
    let servers = start_servers(
        &miners,
        &cli.axon_host,
        cli.handler_timeout,
        handlers,
        false,
    )
    .await?;

    for miner in &miners {
        info!(
            hotkey = %miner.wallet.hotkey_ss58(),
            port = miner.port,
            "miner loopback running"
        );
    }

    run_until_shutdown(servers, None).await
}

/// Axon protocol number of a QUIC (btlightning) server.
const QUIC_AXON_PROTOCOL: u8 = 4;
/// Bound on one registration attempt: connecting, reading the chain, and
/// submitting and finalizing `serve_axon`, which sn2-chain bounds at 30 s and
/// 180 s.
const AXON_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(300);
/// After a failed attempt the retry delay doubles from the first to the max.
const AXON_FIRST_BACKOFF: Duration = Duration::from_secs(15);
const AXON_MAX_BACKOFF: Duration = Duration::from_secs(300);
/// How often a registered axon is checked against the chain.
const AXON_RECHECK_INTERVAL: Duration = Duration::from_secs(600);
/// Subtensor's block time.
const CHAIN_BLOCK_TIME: Duration = Duration::from_secs(12);

/// Everything one miner's registration task needs.
struct AxonTask {
    registration: Arc<sn2_chain::Registration>,
    endpoint: String,
    external_ip: Arc<ExternalIp>,
    wallet: Arc<sn2_chain::Wallet>,
    port: u16,
    netuid: u16,
    network: String,
}

/// Keeps one miner's axon registered on chain for as long as the process runs.
///
/// Retries without limit until the chain holds this miner's IP and port, then
/// checks the chain again every [`AXON_RECHECK_INTERVAL`] and registers the
/// axon again if the entry changed, for example because the public IP moved.
/// A hotkey with no uid on the subnet is retried until it has one.
///
/// Every attempt opens its own chain connection. The RPC client neither pings
/// nor reconnects, so a connection left idle while a cold circuit cache
/// downloads, or between checks, can be dead by the time it is used. Retrying
/// after a timeout is safe because `serve_axon` reads the chain first and
/// submits nothing when the axon is already registered or the subnet's serving
/// rate limit would reject the call.
async fn keep_axon_registered(task: AxonTask) {
    let mut backoff = AXON_FIRST_BACKOFF;
    let mut registered: Option<IpAddr> = None;
    let mut awaiting_uid = false;
    loop {
        let delay = match register_axon_once(&task).await {
            Ok((ip, ServeAxonOutcome::AlreadyServed)) => {
                if registered.is_none() {
                    info!(%ip, port = task.port, "axon is registered on chain");
                } else {
                    debug!(%ip, port = task.port, "axon still registered on chain");
                }
                registered = Some(ip);
                awaiting_uid = false;
                backoff = AXON_FIRST_BACKOFF;
                AXON_RECHECK_INTERVAL
            }
            Ok((ip, ServeAxonOutcome::Served { replaced })) => {
                match (registered, replaced) {
                    (Some(previous_ip), Some(chain))
                        if chain.ip == previous_ip && chain.port == task.port =>
                    {
                        warn!(%previous_ip, %ip, port = task.port, "public IP changed; axon registered again");
                    }
                    (Some(_), Some(chain)) => warn!(
                        chain_ip = %chain.ip,
                        chain_port = chain.port,
                        %ip,
                        port = task.port,
                        "the chain's axon for this hotkey had been changed, possibly by another process serving the same hotkey; registered it again"
                    ),
                    _ => info!(%ip, port = task.port, "axon is registered on chain"),
                }
                registered = Some(ip);
                awaiting_uid = false;
                backoff = AXON_FIRST_BACKOFF;
                AXON_RECHECK_INTERVAL
            }
            Ok((_, ServeAxonOutcome::NotRegistered)) => {
                let delay = next_backoff(&mut backoff);
                if awaiting_uid {
                    debug!(
                        retry_in_secs = delay.as_secs(),
                        "hotkey still has no uid on subnet {}", task.netuid
                    );
                } else {
                    warn!(
                        retry_in_secs = delay.as_secs(),
                        "hotkey has no uid on subnet {}; its axon is registered as soon as it has one. Register with: btcli subnets register --netuid {} --network {}",
                        task.netuid,
                        task.netuid,
                        task.network,
                    );
                    awaiting_uid = true;
                }
                registered = None;
                delay
            }
            Ok((ip, ServeAxonOutcome::RateLimited { blocks })) => {
                let delay = rate_limit_delay(blocks);
                info!(
                    %ip,
                    port = task.port,
                    blocks,
                    retry_in_secs = delay.as_secs(),
                    "the subnet's serving rate limit does not let this hotkey update its axon yet; retrying when it does"
                );
                delay
            }
            Err(e) => {
                let delay = next_backoff(&mut backoff);
                warn!(
                    port = task.port,
                    retry_in_secs = delay.as_secs(),
                    // `{:#}` prints the whole context chain; the outermost
                    // context alone ("fetching latest block") hides the cause.
                    error = format!("{e:#}"),
                    "axon registration failed; retrying with a new chain connection"
                );
                delay
            }
        };
        tokio::time::sleep(delay).await;
    }
}

/// One registration attempt on a fresh chain connection.
async fn register_axon_once(task: &AxonTask) -> Result<(IpAddr, ServeAxonOutcome)> {
    let ip = task.external_ip.resolve().await?;
    let outcome = tokio::time::timeout(AXON_ATTEMPT_TIMEOUT, async {
        let client = sn2_chain::connect_chain(&task.endpoint).await?;
        task.registration
            .serve_axon(&client, &task.wallet, ip, task.port, QUIC_AXON_PROTOCOL)
            .await
    })
    .await
    .with_context(|| format!("timed out after {}s", AXON_ATTEMPT_TIMEOUT.as_secs()))??;
    Ok((ip, outcome))
}

/// Returns the delay before the next retry and doubles `backoff` up to
/// [`AXON_MAX_BACKOFF`].
fn next_backoff(backoff: &mut Duration) -> Duration {
    let delay = *backoff;
    *backoff = (*backoff * 2).min(AXON_MAX_BACKOFF);
    delay
}

/// Delay until the serving rate limit lets the hotkey update its axon, plus a
/// block of slack for block-time drift. Capped at [`AXON_RECHECK_INTERVAL`] so a
/// long limit still sees the public IP change.
fn rate_limit_delay(blocks: u64) -> Duration {
    let blocks = u32::try_from(blocks.saturating_add(1)).unwrap_or(u32::MAX);
    CHAIN_BLOCK_TIME
        .saturating_mul(blocks)
        .min(AXON_RECHECK_INTERVAL)
}

/// One miner identity served by this process. Every miner shares the process's
/// circuit store, prover and caches; only the hotkey and QUIC port differ.
struct Miner {
    wallet: Arc<sn2_chain::Wallet>,
    port: u16,
}

impl Miner {
    /// Span carrying the miner's hotkey and port into everything done for it.
    fn span(&self) -> Span {
        info_span!("miner", hotkey = %self.wallet.hotkey_ss58(), port = self.port)
    }
}

fn load_miners(cli: &Cli) -> Result<Vec<Miner>> {
    let mut miners = Vec::new();
    let mut hotkeys = HashSet::new();
    for spec in cli.miner_specs()? {
        let wallet_name = spec.wallet_name.as_deref().unwrap_or(&cli.wallet_name);
        let wallet =
            sn2_chain::Wallet::from_paths(wallet_name, &spec.hotkey, cli.wallet_path.as_deref())
                .with_context(|| format!("loading wallet {wallet_name}/{}", spec.hotkey))?;
        anyhow::ensure!(
            hotkeys.insert(wallet.hotkey_ss58().to_string()),
            "hotkey {} is assigned to more than one miner",
            wallet.hotkey_ss58()
        );
        miners.push(Miner {
            wallet: Arc::new(wallet),
            port: spec.port,
        });
    }
    Ok(miners)
}

async fn init_handlers(cli: &Cli, loopback: bool) -> Arc<handlers::MinerHandlers> {
    let dsperse =
        dsperse::DSperseClient::new(cli.circuit_cache_dir.as_deref(), max_concurrent_proofs(cli));
    let circuit_store = init_circuit_store(
        loopback,
        &cli.additional_circuits,
        cli.circuit_cache_dir.as_deref(),
    )
    .await;
    Arc::new(handlers::MinerHandlers::new(dsperse, circuit_store))
}

/// `--max-concurrent-proofs`, or one proof per CPU available to the process:
/// a proof never uses more than one core.
fn max_concurrent_proofs(cli: &Cli) -> usize {
    if let Some(n) = cli.max_concurrent_proofs {
        return n.get();
    }
    match std::thread::available_parallelism() {
        Ok(n) => n.get(),
        Err(e) => {
            warn!(
                error = %e,
                "cannot count the CPUs available to this process; proofs are not limited (set --max-concurrent-proofs)"
            );
            usize::MAX
        }
    }
}

/// Binds every miner's QUIC server, then serves them all. Binding first means a
/// port conflict fails startup before anything is registered on chain.
async fn start_servers(
    miners: &[Miner],
    host: &str,
    handler_timeout: u64,
    handlers: Arc<handlers::MinerHandlers>,
    restrict_to_allowed_validator: bool,
) -> Result<JoinSet<Result<()>>> {
    let mut bound = Vec::with_capacity(miners.len());
    for miner in miners {
        let wallet = &miner.wallet;
        let span = miner.span();
        let server = lightning_server::start_lightning_server(
            wallet.hotkey_ss58(),
            &wallet.name,
            &wallet.wallet_path,
            &wallet.hotkey_name,
            host,
            miner.port,
            handler_timeout,
            handlers.clone(),
            restrict_to_allowed_validator,
        )
        .instrument(span.clone())
        .await
        .with_context(|| {
            format!(
                "starting QUIC server for {} on port {}",
                wallet.hotkey_ss58(),
                miner.port
            )
        })?;
        bound.push((server, span, wallet.hotkey_ss58().to_string(), miner.port));
    }

    let mut servers = JoinSet::new();
    for (server, span, hotkey, port) in bound {
        servers.spawn(
            async move {
                server
                    .serve_forever()
                    .await
                    .with_context(|| format!("QUIC server for {hotkey} on port {port}"))
            }
            .instrument(span),
        );
    }
    Ok(servers)
}

/// Runs until any miner's server stops, a signal arrives, or the auto-updater
/// asks for a restart. Remaining servers are aborted when `servers` drops.
async fn run_until_shutdown(
    mut servers: JoinSet<Result<()>>,
    update_rx: Option<watch::Receiver<bool>>,
) -> Result<()> {
    let mut sigterm = signal(SignalKind::terminate()).context("registering SIGTERM handler")?;
    let update_requested = async {
        let Some(mut rx) = update_rx else {
            return std::future::pending().await;
        };
        loop {
            rx.changed().await.ok()?;
            if *rx.borrow() {
                return Some(());
            }
        }
    };

    tokio::select! {
        Some(r) = servers.join_next() => {
            r.context("QUIC server task panicked")??;
        }
        _ = tokio::signal::ctrl_c() => {
            info!("shutting down miner");
        }
        _ = sigterm.recv() => {
            info!("received SIGTERM, shutting down miner");
        }
        _ = update_requested => {
            info!("shutting down miner for auto-update restart");
        }
    }

    Ok(())
}

/// How long a detected public IP is reused before it is detected again, so
/// the miners' registration tasks share one lookup.
const EXTERNAL_IP_REUSE: Duration = Duration::from_secs(60);
/// Services that answer with the caller's public IPv4 address as plain text,
/// tried in order.
const IP_ECHO_SERVICES: [&str; 3] = [
    "https://api4.ipify.org",
    "https://checkip.amazonaws.com",
    "https://ipv4.icanhazip.com",
];

/// Where the IP registered for every miner's axon comes from.
enum ExternalIp {
    /// `--external-ip`.
    Fixed(IpAddr),
    /// Detected over HTTPS on each registration attempt, so a changed public
    /// IP is picked up.
    Detected(tokio::sync::Mutex<Option<(Instant, IpAddr)>>),
}

impl ExternalIp {
    fn from_cli(override_ip: Option<&str>) -> Result<Self> {
        let Some(ip) = override_ip else {
            return Ok(Self::Detected(tokio::sync::Mutex::new(None)));
        };
        let ip = require_ipv4(ip.parse().context("parsing --external-ip")?)?;
        if !is_publicly_routable(ip) {
            warn!(
                %ip,
                "--external-ip is not a publicly routable address; validators never send requests to an axon at such an address"
            );
        }
        Ok(Self::Fixed(ip))
    }

    async fn resolve(&self) -> Result<IpAddr> {
        let cache = match self {
            Self::Fixed(ip) => return Ok(*ip),
            Self::Detected(cache) => cache,
        };
        let mut cache = cache.lock().await;
        if let Some((detected_at, ip)) = *cache {
            if detected_at.elapsed() < EXTERNAL_IP_REUSE {
                return Ok(ip);
            }
        }
        let ip = detect_external_ip().await?;
        *cache = Some((Instant::now(), ip));
        Ok(ip)
    }
}

async fn detect_external_ip() -> Result<IpAddr> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("building HTTP client for external-IP detection")?;
    let mut failures = Vec::new();
    for url in IP_ECHO_SERVICES {
        match query_ip_echo(&client, url).await {
            Ok(ip) => return Ok(ip),
            Err(e) => failures.push(format!("{url}: {e:#}")),
        }
    }
    anyhow::bail!(
        "detecting the public IP failed ({}); set --external-ip to skip detection",
        failures.join("; ")
    )
}

async fn query_ip_echo(client: &reqwest::Client, url: &str) -> Result<IpAddr> {
    let body = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let ip: IpAddr = body
        .trim()
        .parse()
        .with_context(|| format!("parsing {body:?}"))?;
    require_ipv4(ip)
}

fn require_ipv4(ip: IpAddr) -> Result<IpAddr> {
    match ip {
        IpAddr::V4(_) => Ok(ip),
        IpAddr::V6(_) => {
            anyhow::bail!(
                "external IP must be IPv4 (axon registration does not support IPv6): {ip}"
            )
        }
    }
}

/// Whether validators send requests to an axon at `ip`. They skip every
/// address that is not a globally routable IPv4 address (`is_valid_ip` in
/// sn2-validator).
fn is_publicly_routable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_global() && !v4.is_multicast(),
        IpAddr::V6(_) => false,
    }
}

async fn init_circuit_store(
    loopback: bool,
    additional_circuits: &[String],
    cache_dir_override: Option<&str>,
) -> sn2_circuit_store::CircuitStore {
    let mut store = sn2_circuit_store::CircuitStore::new(
        None,
        loopback,
        additional_circuits.to_vec(),
        cache_dir_override,
    );
    if let Err(e) = store.load_circuits().await {
        warn!(error = %e, "failed to load circuits from cache");
    }
    for id in additional_circuits {
        if let Err(e) = store.ensure_circuit(id).await {
            warn!(id = %id, error = %e, "failed to preload pinned circuit");
        }
    }
    store
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn proof_limit_defaults_to_the_available_cpus() {
        let cli = Cli::try_parse_from(["sn2-miner"]).unwrap();
        assert_eq!(
            max_concurrent_proofs(&cli),
            std::thread::available_parallelism().unwrap().get()
        );
        let cli = Cli::try_parse_from(["sn2-miner", "--max-concurrent-proofs", "3"]).unwrap();
        assert_eq!(max_concurrent_proofs(&cli), 3);
    }

    #[test]
    fn retry_delay_doubles_up_to_the_cap() {
        let mut backoff = AXON_FIRST_BACKOFF;
        let delays: Vec<u64> = (0..8)
            .map(|_| next_backoff(&mut backoff).as_secs())
            .collect();
        assert_eq!(delays, vec![15, 30, 60, 120, 240, 300, 300, 300]);
    }

    #[test]
    fn rate_limit_delay_covers_the_remaining_blocks_plus_one() {
        assert_eq!(rate_limit_delay(0), Duration::from_secs(12));
        assert_eq!(rate_limit_delay(29), Duration::from_secs(360));
        assert_eq!(rate_limit_delay(49), AXON_RECHECK_INTERVAL);
        assert_eq!(rate_limit_delay(u64::MAX), AXON_RECHECK_INTERVAL);
    }

    #[test]
    fn only_global_ipv4_addresses_are_publicly_routable() {
        for ip in ["8.8.8.8", "54.210.10.20"] {
            assert!(is_publicly_routable(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "10.0.0.1",
            "172.16.5.4",
            "192.168.1.10",
            "100.64.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "169.254.1.1",
            "224.0.0.1",
            "2001:4860:4860::8888",
        ] {
            assert!(!is_publicly_routable(ip.parse().unwrap()), "{ip}");
        }
    }

    #[tokio::test]
    async fn external_ip_override_is_validated_and_used_as_is() {
        let fixed = ExternalIp::from_cli(Some("54.210.10.20")).unwrap();
        assert_eq!(
            fixed.resolve().await.unwrap(),
            "54.210.10.20".parse::<IpAddr>().unwrap()
        );
        assert!(ExternalIp::from_cli(Some("::1")).is_err());
        assert!(ExternalIp::from_cli(Some("not-an-ip")).is_err());
        assert!(matches!(
            ExternalIp::from_cli(None).unwrap(),
            ExternalIp::Detected(_)
        ));
    }

    #[tokio::test]
    async fn a_recently_detected_ip_is_reused() {
        let ip: IpAddr = "54.210.10.20".parse().unwrap();
        let detected = ExternalIp::Detected(tokio::sync::Mutex::new(Some((Instant::now(), ip))));
        assert_eq!(detected.resolve().await.unwrap(), ip);
    }
}
