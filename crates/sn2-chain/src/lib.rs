pub mod attestation;
pub mod auto_update;
mod metagraph;
mod registration;
mod subxt_helpers;
mod wallet;
mod weights;

use std::sync::Arc;

use anyhow::{Context, Result};
use subxt::backend::LegacyBackend;
use subxt::rpcs::RpcClient;
use subxt::{OnlineClient, PolkadotConfig};

pub use metagraph::{Metagraph, NeuronInfo};
pub use registration::Registration;
pub use wallet::Wallet;
pub use weights::WeightsSetter;

pub const FINNEY_ENDPOINT: &str = "wss://entrypoint-finney.opentensor.ai:443";
pub const TEST_ENDPOINT: &str = "wss://test.finney.opentensor.ai:443";
pub const LOCAL_ENDPOINT: &str = "ws://127.0.0.1:9944";

pub fn resolve_endpoint(network: &str, override_endpoint: Option<&str>) -> String {
    match override_endpoint {
        Some(ep) => ep.to_string(),
        None => match network {
            "finney" | "mainnet" => FINNEY_ENDPOINT.to_string(),
            "test" | "testnet" => TEST_ENDPOINT.to_string(),
            "local" | "localnet" => LOCAL_ENDPOINT.to_string(),
            other => other.to_string(),
        },
    }
}

/// Open a subxt `OnlineClient` against `endpoint`, forcing subxt's plain
/// legacy backend rather than letting it auto-negotiate via
/// `CombinedBackend` (subxt's default for `OnlineClient::from_url`/
/// `from_insecure_url`).
///
/// `CombinedBackend` probes the node's `rpc_methods` on connect and
/// prefers `archive_v1_*`/`chainHead_v1_*` RPC methods when the node
/// claims to support them, falling back to legacy storage calls only if
/// those fail -- but it reports whichever backend it tried *last*
/// (`try_backends` in subxt's combined-backend implementation returns the
/// final attempt's error), so a node that advertises the newer tiers
/// without them actually working surfaces as a hard failure even when
/// plain legacy calls against the same node work fine. That's exactly
/// what was confirmed against wss://entrypoint-finney.opentensor.ai:443:
/// `state_getRuntimeVersion` (a basic legacy RPC call) succeeds directly,
/// while `CombinedBackend`'s negotiated storage fetch fails with
/// "Method not found (-32601)" during metagraph sync. Building an
/// explicit `LegacyBackend` skips the `rpc_methods` negotiation (and the
/// archive/chainHead tiers) entirely, going straight to the calls already
/// confirmed to work.
///
/// `wss://` URLs use the TLS-validating `RpcClient::from_url`; `ws://`
/// URLs use `from_insecure_url`, which subxt requires for non-TLS sockets
/// even when reaching localhost or a private substrate node.
pub async fn connect_chain(endpoint: &str) -> Result<OnlineClient<PolkadotConfig>> {
    let rpc_client = if endpoint.starts_with("ws://") {
        RpcClient::from_insecure_url(endpoint).await
    } else {
        RpcClient::from_url(endpoint).await
    }
    .with_context(|| format!("opening RPC connection to subtensor at {endpoint}"))?;

    let backend = Arc::new(LegacyBackend::<PolkadotConfig>::builder().build(rpc_client));

    OnlineClient::from_backend(backend)
        .await
        .with_context(|| format!("connecting to subtensor at {endpoint}"))
}

pub fn is_rpc_disconnect(err: &anyhow::Error) -> bool {
    for cause in err.chain() {
        if let Some(subxt_err) = cause.downcast_ref::<subxt::Error>() {
            return subxt_err.is_disconnected_will_reconnect();
        }
    }
    false
}
