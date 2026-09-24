//! A validator behind a NAT (e.g. Cloudflare WARP egress) can have its UDP
//! source port rebound mid-connection. quinn migrates the live QUIC
//! connection to the new port; the miner must keep serving the validator it
//! already authenticated instead of rejecting every request as
//! "Unknown or unauthenticated connection".
//!
//! Topology: LightningClient -> [NAT: inside socket | outside socket] -> LightningServer.
//! `Nat::rebind` swaps the outside socket for a fresh one (new source port) and
//! drops traffic on the old mapping, as a NAT does when it reallocates a flow.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use btlightning::{
    LightningClient, LightningServer, LightningServerConfig, QuicAxonInfo, QuicRequest, Result,
    Sr25519Signer, SynapseHandler,
};
use sp_core::{crypto::Ss58Codec, sr25519, Pair};
use tokio::net::UdpSocket;
use tokio::sync::Mutex;
use tokio::time::timeout;

const MINER_SEED: [u8; 32] = [1u8; 32];
const VALIDATOR_SEED: [u8; 32] = [2u8; 32];
const STEP_TIMEOUT: Duration = Duration::from_secs(20);

struct Echo;

impl SynapseHandler for Echo {
    fn handle(
        &self,
        _synapse_type: &str,
        data: HashMap<String, rmpv::Value>,
    ) -> Result<HashMap<String, rmpv::Value>> {
        Ok(data)
    }
}

struct Nat {
    inside: Arc<UdpSocket>,
    server: SocketAddr,
    outside: Mutex<Arc<UdpSocket>>,
    client: Mutex<Option<SocketAddr>>,
    generation: AtomicU64,
}

impl Nat {
    async fn new(server: SocketAddr) -> Arc<Self> {
        let inside = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let outside = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let nat = Arc::new(Nat {
            inside,
            server,
            outside: Mutex::new(outside.clone()),
            client: Mutex::new(None),
            generation: AtomicU64::new(0),
        });

        let forward = nat.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            loop {
                let Ok((len, from)) = forward.inside.recv_from(&mut buf).await else {
                    return;
                };
                *forward.client.lock().await = Some(from);
                let out = forward.outside.lock().await.clone();
                let _ = out.send_to(&buf[..len], forward.server).await;
            }
        });
        nat.spawn_outside_reader(outside, 0);
        nat
    }

    fn spawn_outside_reader(self: &Arc<Self>, sock: Arc<UdpSocket>, generation: u64) {
        let nat = self.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65535];
            loop {
                let Ok((len, _)) = sock.recv_from(&mut buf).await else {
                    return;
                };
                if nat.generation.load(Ordering::SeqCst) != generation {
                    continue;
                }
                if let Some(client) = *nat.client.lock().await {
                    let _ = nat.inside.send_to(&buf[..len], client).await;
                }
            }
        });
    }

    async fn outside_port(&self) -> u16 {
        self.outside.lock().await.local_addr().unwrap().port()
    }

    async fn rebind(self: &Arc<Self>) -> u16 {
        let fresh = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.spawn_outside_reader(fresh.clone(), generation);
        let port = fresh.local_addr().unwrap().port();
        *self.outside.lock().await = fresh;
        port
    }
}

fn echo_request(tag: &str) -> QuicRequest {
    let mut data = HashMap::new();
    data.insert("tag".to_string(), rmpv::Value::String(tag.into()));
    QuicRequest {
        synapse_type: "echo".to_string(),
        data,
    }
}

async fn assert_query_succeeds(client: &LightningClient, axon: &QuicAxonInfo, tag: &str) {
    let response = timeout(
        STEP_TIMEOUT,
        client.query_axon(axon.clone(), echo_request(tag)),
    )
    .await
    .unwrap_or_else(|_| panic!("query {tag} timed out"))
    .unwrap_or_else(|e| panic!("query {tag} failed: {e}"));
    assert!(
        response.success,
        "query {tag} was rejected: {:?}",
        response.error
    );
    assert_eq!(
        response.data.get("tag"),
        Some(&rmpv::Value::String(tag.into()))
    );
}

#[tokio::test]
async fn authenticated_validator_survives_nat_rebinding() {
    let miner_hotkey = sr25519::Pair::from_seed(&MINER_SEED)
        .public()
        .to_ss58check();
    let validator_hotkey = sr25519::Pair::from_seed(&VALIDATOR_SEED)
        .public()
        .to_ss58check();

    // Same transport settings the miner uses (lightning_server.rs), minus the
    // chain-backed validator permit check.
    let config = LightningServerConfig::builder()
        .idle_timeout_secs(360)
        .require_address_validation(true)
        .build()
        .unwrap();
    let mut server =
        LightningServer::with_config(miner_hotkey.clone(), "127.0.0.1".into(), 0, config).unwrap();
    server.set_miner_keypair(MINER_SEED);
    server
        .register_synapse_handler("echo".to_string(), Arc::new(Echo))
        .await
        .unwrap();
    server.start().await.unwrap();
    let server_addr = server.local_addr().unwrap();
    let server = Arc::new(server);
    let serving = server.clone();
    tokio::spawn(async move { serving.serve_forever().await });

    let nat = Nat::new(server_addr).await;
    let axon = QuicAxonInfo {
        hotkey: miner_hotkey,
        ip: "127.0.0.1".into(),
        port: nat.inside.local_addr().unwrap().port(),
        protocol: 4,
    };

    let mut client = LightningClient::new(validator_hotkey);
    client.set_signer(Box::new(Sr25519Signer::from_seed(VALIDATOR_SEED)));
    client.create_endpoint().await.unwrap();
    timeout(
        STEP_TIMEOUT,
        client.initialize_connections(vec![axon.clone()]),
    )
    .await
    .expect("handshake timed out")
    .unwrap();

    assert_query_succeeds(&client, &axon, "before-rebind").await;

    let first_port = nat.outside_port().await;
    let second_port = nat.rebind().await;
    assert_ne!(first_port, second_port);
    for i in 0..3 {
        assert_query_succeeds(&client, &axon, &format!("after-rebind-{i}")).await;
    }

    let third_port = nat.rebind().await;
    assert_ne!(second_port, third_port);
    for i in 0..3 {
        assert_query_succeeds(&client, &axon, &format!("after-second-rebind-{i}")).await;
    }

    // Every query rode the original connection: the client never had to
    // reconnect and handshake again, which would consume another nonce.
    assert_eq!(server.get_active_nonce_count().await, 1);

    let _ = client.close_all_connections().await;
    let _ = server.stop().await;
}
