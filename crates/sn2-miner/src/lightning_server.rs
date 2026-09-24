use std::sync::Arc;

use anyhow::Result;
use btlightning::{typed_async_handler, LightningServer, LightningServerConfig};
use tracing::info;

use sn2_types::*;

use crate::handlers::MinerHandlers;

/// The only validator hotkey this miner serves. Handshakes from any other
/// hotkey are rejected; the peer's IP address and port are not checked.
pub const ALLOWED_VALIDATOR_HOTKEY: &str = "5CFxLBvpyQq3TCP7zLcu8dLcPBxhA6MdLvXiCyvJVojuK17J";

pub async fn run_lightning_server(
    miner_hotkey: &str,
    wallet_name: &str,
    wallet_path: &str,
    hotkey_name: &str,
    host: &str,
    port: u16,
    handler_timeout_secs: u64,
    handlers: Arc<MinerHandlers>,
    restrict_to_allowed_validator: bool,
) -> Result<()> {
    let idle_timeout = handler_timeout_secs.saturating_mul(2).max(150);
    let config = LightningServerConfig::builder()
        .handler_timeout_secs(handler_timeout_secs)
        .idle_timeout_secs(idle_timeout)
        .max_frame_payload_bytes(sn2_types::TRANSPORT_PAYLOAD_LIMIT)
        .require_validator_permit(restrict_to_allowed_validator)
        .require_address_validation(true)
        .build()?;
    let mut server =
        LightningServer::with_config(miner_hotkey.to_string(), host.to_string(), port, config)?;

    server.set_miner_wallet(wallet_name, wallet_path, hotkey_name)?;

    if restrict_to_allowed_validator {
        server.set_validator_permit_resolver(Box::new(AllowedValidator));
        info!(
            validator = ALLOWED_VALIDATOR_HOTKEY,
            "accepting handshakes from a single validator hotkey"
        );
    }

    let h = handlers.clone();
    server
        .register_async_synapse_handler(
            QueryZkProof::NAME.to_string(),
            typed_async_handler(move |query: QueryZkProof| {
                let h = h.clone();
                async move { h.handle_query_zk_proof(query).await }
            }),
        )
        .await?;

    let h = handlers.clone();
    server
        .register_async_synapse_handler(
            DSliceProofGenerationDataModel::NAME.to_string(),
            typed_async_handler(move |query: DSliceProofGenerationDataModel| {
                let h = h.clone();
                async move { h.handle_dslice(query).await }
            }),
        )
        .await?;

    let h = handlers.clone();
    server
        .register_async_synapse_handler(
            ProofOfWeightsDataModel::NAME.to_string(),
            typed_async_handler(move |query: ProofOfWeightsDataModel| {
                let h = h.clone();
                async move { h.handle_proof_of_weights(query).await }
            }),
        )
        .await?;

    server.start().await?;

    info!(host = host, port = port, "QUIC Lightning server listening");

    server.serve_forever().await?;
    Ok(())
}

struct AllowedValidator;

impl btlightning::ValidatorPermitResolver for AllowedValidator {
    fn resolve_permitted_validators(
        &self,
    ) -> btlightning::Result<std::collections::HashSet<String>> {
        Ok([ALLOWED_VALIDATOR_HOTKEY.to_string()].into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use btlightning::ValidatorPermitResolver;

    #[test]
    fn only_the_allowed_validator_is_permitted() {
        let permitted = AllowedValidator.resolve_permitted_validators().unwrap();
        assert_eq!(permitted.len(), 1);
        assert!(permitted.contains("5CFxLBvpyQq3TCP7zLcu8dLcPBxhA6MdLvXiCyvJVojuK17J"));
    }
}
