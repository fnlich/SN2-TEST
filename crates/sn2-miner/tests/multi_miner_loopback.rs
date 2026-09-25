//! Runs the real sn2-miner binary in loopback mode serving two miners from one
//! process and checks that each port answers as its own hotkey.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use btlightning::{LightningClient, QuicAxonInfo, QuicRequest, Sr25519Signer};
use sp_core::{crypto::Ss58Codec, sr25519, Pair};

const VALIDATOR_SEED: [u8; 32] = [2u8; 32];
const HOTKEY_PHRASES: [(&str, &str); 2] = [
    (
        "hk1",
        "bottom drive obey lake curtain smoke basket hold race lonely fit walk",
    ),
    (
        "hk2",
        "legal winner thank year wave sausage worth useful legal winner thank yellow",
    ),
];

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sn2-multi-miner-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_hotkey(wallets: &Path, hotkey: &str, phrase: &str) -> String {
    let dir = wallets.join("testwallet").join("hotkeys");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(hotkey),
        format!(r#"{{"secretPhrase":"{phrase}"}}"#),
    )
    .unwrap();
    sr25519::Pair::from_phrase(phrase, None)
        .unwrap()
        .0
        .public()
        .to_ss58check()
}

/// Distinct free UDP ports. The sockets stay open until every port is read so
/// the OS cannot hand the same port out twice.
fn free_udp_ports<const N: usize>() -> [u16; N] {
    let sockets: Vec<_> = (0..N)
        .map(|_| std::net::UdpSocket::bind("127.0.0.1:0").unwrap())
        .collect();
    std::array::from_fn(|i| sockets[i].local_addr().unwrap().port())
}

fn axon(hotkey: &str, port: u16) -> QuicAxonInfo {
    QuicAxonInfo {
        hotkey: hotkey.to_string(),
        ip: "127.0.0.1".into(),
        port,
        protocol: 4,
    }
}

fn request() -> QuicRequest {
    QuicRequest {
        synapse_type: sn2_types::QueryZkProof::NAME.to_string(),
        data: HashMap::new(),
    }
}

async fn validator_client() -> LightningClient {
    let validator = sr25519::Pair::from_seed(&VALIDATOR_SEED)
        .public()
        .to_ss58check();
    let mut client = LightningClient::new(validator);
    client.set_signer(Box::new(Sr25519Signer::from_seed(VALIDATOR_SEED)));
    client.create_endpoint().await.unwrap();
    client
}

/// Queries until the miner is up. `Ok` means the handshake succeeded, which
/// requires the server on that port to sign as `axon.hotkey`. Each attempt uses
/// a fresh client so the client's reconnect backoff cannot outlast the deadline.
async fn query_when_ready(miner: &mut KillOnDrop, log: &Path, axon: &QuicAxonInfo) {
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if let Some(status) = miner.0.try_wait().unwrap() {
            panic!(
                "sn2-miner exited with {status}:\n{}",
                std::fs::read_to_string(log).unwrap_or_default()
            );
        }
        let client = validator_client().await;
        let result = client.query_axon(axon.clone(), request()).await;
        let _ = client.close_all_connections().await;
        match result {
            Ok(response) => {
                assert_ne!(
                    response.error.as_deref(),
                    Some("authentication failed"),
                    "miner on port {} rejected the validator",
                    axon.port
                );
                return;
            }
            Err(e) if Instant::now() >= deadline => panic!(
                "miner on port {} never answered: {e}\n{}",
                axon.port,
                std::fs::read_to_string(log).unwrap_or_default()
            ),
            Err(_) => tokio::time::sleep(Duration::from_millis(250)).await,
        }
    }
}

#[tokio::test]
async fn one_process_serves_each_miner_on_its_own_port_and_hotkey() {
    let dir = scratch_dir();
    let wallets = dir.join("wallets");
    let hotkeys: Vec<String> = HOTKEY_PHRASES
        .iter()
        .map(|(name, phrase)| write_hotkey(&wallets, name, phrase))
        .collect();
    assert_ne!(hotkeys[0], hotkeys[1]);
    let ports: [u16; 2] = free_udp_ports();

    let log_path = dir.join("miner.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let mut miner = KillOnDrop(
        Command::new(env!("CARGO_BIN_EXE_sn2-miner"))
            .args(["--loopback", "--no-auto-update", "--axon-host", "127.0.0.1"])
            .arg("--wallet-path")
            .arg(&wallets)
            .args(["--wallet-name", "testwallet"])
            .arg("--circuit-cache-dir")
            .arg(dir.join("circuit_cache"))
            .arg("--miner")
            .arg(format!("hk1:{},hk2:{}", ports[0], ports[1]))
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .expect("starting sn2-miner"),
    );

    for (hotkey, port) in hotkeys.iter().zip(ports) {
        query_when_ready(&mut miner, &log_path, &axon(hotkey, port)).await;
    }

    // A port answers only as its own hotkey: expecting the other miner's
    // hotkey there fails the handshake's miner-signature check...
    let mismatched = validator_client().await;
    assert!(mismatched
        .query_axon(axon(&hotkeys[1], ports[0]), request())
        .await
        .is_err());
    let _ = mismatched.close_all_connections().await;
    // ...while the same port still answers as its own hotkey.
    query_when_ready(&mut miner, &log_path, &axon(&hotkeys[0], ports[0])).await;

    drop(miner);
    let _ = std::fs::remove_dir_all(&dir);
}
