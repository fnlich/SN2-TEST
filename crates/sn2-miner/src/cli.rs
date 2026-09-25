use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "sn2-miner", about = "Subnet-2 Miner")]
pub struct Cli {
    #[arg(long, default_value_t = sn2_types::DEFAULT_NETUID)]
    pub netuid: u16,

    #[arg(long, alias = "subtensor.network", default_value = "finney")]
    pub network: String,

    #[arg(long, alias = "subtensor.chain_endpoint")]
    pub subtensor_chain_endpoint: Option<String>,

    #[arg(long, alias = "wallet.name", default_value = "default")]
    pub wallet_name: String,

    #[arg(long, alias = "wallet.hotkey", default_value = "default")]
    pub wallet_hotkey: String,

    #[arg(long, alias = "wallet.path")]
    pub wallet_path: Option<String>,

    #[arg(long, alias = "logging.level", default_value = "info")]
    pub log_level: String,

    #[arg(long, alias = "axon.host", default_value = "0.0.0.0")]
    pub axon_host: String,

    #[arg(long, alias = "axon.port", default_value_t = 8091)]
    pub axon_port: u16,

    #[arg(long, alias = "axon.external_ip")]
    pub external_ip: Option<String>,

    #[arg(
        long = "miner",
        value_name = "[WALLET/]HOTKEY:PORT",
        value_delimiter = ',',
        value_parser = parse_miner_spec,
        conflicts_with = "axon_port",
        help = "Serve several miners from this one process, each with its own \
                hotkey and QUIC port, e.g. --miner hk1:8091,hk2:8092. Repeat \
                the flag or separate entries with commas. WALLET defaults to \
                --wallet-name. All miners share one circuit cache and prover. \
                Replaces --wallet-hotkey and cannot be combined with \
                --axon-port."
    )]
    pub miners: Vec<MinerSpec>,

    #[arg(long, default_value_t = false)]
    pub no_auto_update: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Run without chain interaction for local integration testing"
    )]
    pub loopback: bool,

    #[arg(long, value_delimiter = ',')]
    pub additional_circuits: Vec<String>,

    #[arg(
        long,
        env = "SN2_CIRCUIT_CACHE_DIR",
        help = "Directory for the persisted circuit cache. Defaults to \
                ~/.bittensor/subnet-2/circuit_cache when unset. May also be \
                supplied via the SN2_CIRCUIT_CACHE_DIR environment variable; \
                the CLI flag wins when both are present. A leading ~ is \
                expanded to the home directory."
    )]
    pub circuit_cache_dir: Option<String>,

    #[arg(long, default_value_t = sn2_types::CIRCUIT_TIMEOUT_SECONDS, value_parser = clap::value_parser!(u64).range(1..))]
    pub handler_timeout: u64,
}

/// One miner identity served by this process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinerSpec {
    /// Wallet (coldkey) directory name; `None` means `--wallet-name`.
    pub wallet_name: Option<String>,
    pub hotkey: String,
    pub port: u16,
}

pub fn parse_miner_spec(spec: &str) -> Result<MinerSpec, String> {
    let spec = spec.trim();
    let usage = || format!("expected [WALLET/]HOTKEY:PORT, got {spec:?}");
    let (identity, port) = spec.rsplit_once(':').ok_or_else(usage)?;
    let port: u16 = port.parse().map_err(|_| usage())?;
    if port == 0 {
        return Err(format!("port must be non-zero in {spec:?}"));
    }
    let (wallet_name, hotkey) = match identity.split_once('/') {
        Some((wallet, hotkey)) => (Some(wallet.to_string()), hotkey.to_string()),
        None => (None, identity.to_string()),
    };
    if hotkey.is_empty() || wallet_name.as_deref() == Some("") {
        return Err(usage());
    }
    Ok(MinerSpec {
        wallet_name,
        hotkey,
        port,
    })
}

impl Cli {
    /// The miners to serve: every `--miner`, or the single miner described by
    /// `--wallet-hotkey` and `--axon-port` when none is given.
    pub fn miner_specs(&self) -> anyhow::Result<Vec<MinerSpec>> {
        let specs = if self.miners.is_empty() {
            anyhow::ensure!(self.axon_port != 0, "QUIC port must be non-zero");
            vec![MinerSpec {
                wallet_name: None,
                hotkey: self.wallet_hotkey.clone(),
                port: self.axon_port,
            }]
        } else {
            self.miners.clone()
        };
        let mut ports = std::collections::HashSet::new();
        for spec in &specs {
            anyhow::ensure!(
                ports.insert(spec.port),
                "port {} is assigned to more than one miner",
                spec.port
            );
        }
        Ok(specs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("sn2-miner").chain(args.iter().copied())).unwrap()
    }

    #[test]
    fn parses_hotkey_and_wallet_forms() {
        assert_eq!(
            parse_miner_spec("hk1:8091").unwrap(),
            MinerSpec {
                wallet_name: None,
                hotkey: "hk1".into(),
                port: 8091
            }
        );
        assert_eq!(
            parse_miner_spec("cold/hk2:8092").unwrap(),
            MinerSpec {
                wallet_name: Some("cold".into()),
                hotkey: "hk2".into(),
                port: 8092
            }
        );
    }

    #[test]
    fn rejects_malformed_specs() {
        for bad in [
            "hk1",
            "hk1:",
            ":8091",
            "hk1:0",
            "hk1:70000",
            "/hk1:8091",
            "cold/:8091",
        ] {
            assert!(parse_miner_spec(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn defaults_to_single_miner_from_wallet_flags() {
        let specs = cli(&["--wallet-hotkey", "main", "--axon-port", "9000"])
            .miner_specs()
            .unwrap();
        assert_eq!(
            specs,
            vec![MinerSpec {
                wallet_name: None,
                hotkey: "main".into(),
                port: 9000
            }]
        );
    }

    #[test]
    fn collects_repeated_and_comma_separated_miners() {
        let specs = cli(&["--miner", "a:8091, b:8092", "--miner", "w/c:8093"])
            .miner_specs()
            .unwrap();
        let ports: Vec<u16> = specs.iter().map(|s| s.port).collect();
        assert_eq!(ports, vec![8091, 8092, 8093]);
        assert_eq!(specs[1].hotkey, "b");
        assert_eq!(specs[2].wallet_name.as_deref(), Some("w"));
    }

    #[test]
    fn rejects_miner_with_axon_port() {
        let parsed = Cli::try_parse_from(["sn2-miner", "--miner", "a:8091", "--axon-port", "8091"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_duplicate_ports() {
        assert!(cli(&["--miner", "a:8091,b:8091"]).miner_specs().is_err());
    }
}
