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
use std::time::Duration;

use anyhow::{Context, Result};
use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches};
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tracing::{info, info_span, warn, Instrument};

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

    let endpoint =
        sn2_chain::resolve_endpoint(&cli.network, cli.subtensor_chain_endpoint.as_deref());

    let chain_client = sn2_chain::connect_chain(&endpoint).await?;

    let registration = sn2_chain::Registration::new(cli.netuid);

    let mut metagraph = sn2_chain::Metagraph::new(cli.netuid);
    metagraph
        .sync(&chain_client)
        .await
        .context("initial metagraph sync")?;

    // A deregistered hotkey must not take the other miners down with it.
    let (miners, unregistered): (Vec<Miner>, Vec<Miner>) = miners.into_iter().partition(|m| {
        metagraph
            .get_uid_by_hotkey(m.wallet.hotkey_ss58())
            .is_some()
    });
    for miner in &unregistered {
        warn!(
            hotkey = %miner.wallet.hotkey_ss58(),
            port = miner.port,
            "hotkey is not registered on subnet {}; not serving it. Register with: btcli subnets register --netuid {} --network {}",
            cli.netuid,
            cli.netuid,
            cli.network,
        );
    }
    anyhow::ensure!(
        !miners.is_empty(),
        "none of the configured hotkeys is registered on subnet {}",
        cli.netuid
    );

    let external_ip = match resolve_external_ip(cli.external_ip.as_deref()).await {
        Ok(ip) => Some(ip),
        Err(e) if cli.external_ip.is_none() => {
            warn!(
                error = ?e,
                "external IP autodetection failed; skipping serve_axon for this boot"
            );
            None
        }
        Err(e) => return Err(e),
    };

    let handlers = init_handlers(&cli, false).await;
    let servers = start_servers(&miners, "0.0.0.0", cli.handler_timeout, handlers, true).await?;

    for miner in &miners {
        info!(
            hotkey = %miner.wallet.hotkey_ss58(),
            quic_port = miner.port,
            "miner running"
        );
    }

    // Registering an axon waits for block finalization. Each hotkey signs its
    // own extrinsic and has its own serving rate limit, so register them
    // concurrently, off the path that watches the servers and signals.
    if let Some(external_ip) = external_ip {
        let registration = Arc::new(registration);
        for miner in &miners {
            let registration = registration.clone();
            let chain_client = chain_client.clone();
            let wallet = miner.wallet.clone();
            let port = miner.port;
            tokio::spawn(async move {
                let served = tokio::time::timeout(
                    SERVE_AXON_TIMEOUT,
                    registration.serve_axon(&chain_client, &wallet, external_ip, port, 4),
                )
                .await;
                let error = match served {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => e.to_string(),
                    Err(_) => format!("timed out after {}s", SERVE_AXON_TIMEOUT.as_secs()),
                };
                warn!(
                    hotkey = %wallet.hotkey_ss58(),
                    port,
                    error,
                    "serve_axon failed (rate-limited or transient); miner will continue"
                );
            });
        }
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

const SERVE_AXON_TIMEOUT: Duration = Duration::from_secs(120);

/// One miner identity served by this process. Every miner shares the process's
/// circuit store, prover and caches; only the hotkey and QUIC port differ.
struct Miner {
    wallet: Arc<sn2_chain::Wallet>,
    port: u16,
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
    let dsperse = dsperse::DSperseClient::new(cli.circuit_cache_dir.as_deref());
    let circuit_store = init_circuit_store(
        loopback,
        &cli.additional_circuits,
        cli.circuit_cache_dir.as_deref(),
    )
    .await;
    Arc::new(handlers::MinerHandlers::new(dsperse, circuit_store))
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
        let span = info_span!("miner", hotkey = %wallet.hotkey_ss58(), port = miner.port);
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

async fn resolve_external_ip(override_ip: Option<&str>) -> Result<IpAddr> {
    if let Some(ip) = override_ip {
        let parsed: IpAddr = ip.parse().context("parsing --external-ip")?;
        return require_ipv4(parsed);
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("building HTTP client for external-IP detection")?;
    let resp = client
        .get("https://api4.ipify.org")
        .send()
        .await
        .context("detecting external IP via api4.ipify.org")?
        .text()
        .await
        .context("reading external IP response body")?;
    let parsed: IpAddr = resp
        .trim()
        .parse()
        .with_context(|| format!("parsing detected IP: {resp}"))?;
    require_ipv4(parsed)
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
