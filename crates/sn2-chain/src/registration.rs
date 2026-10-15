use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

use anyhow::{Context, Result};
use subxt::dynamic::Value;
use subxt::ext::scale_value::At;
use subxt::{OnlineClient, PolkadotConfig};
use tracing::{debug, info};

use crate::subxt_helpers::{
    at_current_block, fetch_typed, fetch_u128_or, fetch_value, netuid_hotkey_keys, netuid_keys,
    PALLET,
};
use crate::wallet::Wallet;
use crate::weights::{TX_FINALIZATION_TIMEOUT, TX_SUBMIT_TIMEOUT};

/// Subtensor's `InitialServingRateLimit`, for a chain whose metadata carries no
/// default for `ServingRateLimit`.
const DEFAULT_SERVING_RATE_LIMIT: u128 = 50;

/// A hotkey's axon endpoint as the chain records it on a subnet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainAxon {
    pub ip: IpAddr,
    pub port: u16,
    pub protocol: u8,
    /// Block of the `serve_axon` call that set this entry.
    pub block: u64,
}

impl ChainAxon {
    fn from_value(value: &Value) -> Self {
        let field = |name: &str| value.at(name).and_then(|v| v.as_u128()).unwrap_or(0);
        let raw_ip = field("ip");
        let ip = if field("ip_type") == 6 {
            IpAddr::V6(Ipv6Addr::from(raw_ip))
        } else {
            IpAddr::V4(Ipv4Addr::from(raw_ip as u32))
        };
        Self {
            ip,
            port: field("port") as u16,
            protocol: field("protocol") as u8,
            block: field("block") as u64,
        }
    }
}

/// What a [`Registration::serve_axon`] call did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeAxonOutcome {
    /// The chain already holds this endpoint for the hotkey; nothing was submitted.
    AlreadyServed,
    /// A `serve_axon` extrinsic setting the endpoint was finalized. `replaced`
    /// is the endpoint the chain held for the hotkey before it, if any.
    Served { replaced: Option<ChainAxon> },
    /// The hotkey holds no uid on the subnet, so the chain would reject the
    /// call; nothing was submitted.
    NotRegistered,
    /// The subnet's serving rate limit rejects a new `serve_axon` from this
    /// hotkey for `blocks` more blocks; nothing was submitted.
    RateLimited { blocks: u64 },
}

pub struct Registration {
    netuid: u16,
}

impl Registration {
    pub fn new(netuid: u16) -> Self {
        Self { netuid }
    }

    /// Publishes `ip:port` as `wallet`'s axon on this subnet.
    ///
    /// Reads the latest finalized block first and submits nothing when the
    /// chain already holds this endpoint, when the hotkey has no uid on the
    /// subnet, or when the subnet's serving rate limit would make the chain
    /// reject the call (subtensor checks the limit only when the extrinsic
    /// executes, so a rate-limited call would be included and then fail).
    /// Calling it again after any failure, including a timeout after the
    /// extrinsic was submitted, is therefore safe.
    pub async fn serve_axon(
        &self,
        client: &OnlineClient<PolkadotConfig>,
        wallet: &Arc<Wallet>,
        ip: IpAddr,
        port: u16,
        protocol: u8,
    ) -> Result<ServeAxonOutcome> {
        let ip_int = match ip {
            IpAddr::V4(v4) => u32::from(v4) as u64,
            IpAddr::V6(_) => {
                anyhow::bail!("IPv6 is not supported for axon registration");
            }
        };

        let hotkey_bytes = wallet.hotkey_public_bytes()?;
        let at_block = at_current_block(client).await?;

        let uid = fetch_typed::<u16>(
            &at_block,
            "Uids",
            netuid_hotkey_keys(self.netuid, &hotkey_bytes),
        )
        .await
        .context("fetching Uids entry")?;
        if uid.is_none() {
            return Ok(ServeAxonOutcome::NotRegistered);
        }

        let current = fetch_value(
            &at_block,
            "Axons",
            netuid_hotkey_keys(self.netuid, &hotkey_bytes),
        )
        .await
        .context("fetching existing Axons entry")?
        .map(|v| ChainAxon::from_value(&v));

        if let Some(axon) = current {
            if axon.ip == ip && axon.port == port && axon.protocol == protocol {
                debug!(ip = %ip, port, "axon already registered on chain, skipping");
                return Ok(ServeAxonOutcome::AlreadyServed);
            }
            let rate_limit = fetch_u128_or(
                &at_block,
                "ServingRateLimit",
                netuid_keys(self.netuid),
                DEFAULT_SERVING_RATE_LIMIT,
            )
            .await
            .context("fetching ServingRateLimit")?;
            let blocks = serving_rate_limit_wait(
                axon.block,
                u64::try_from(rate_limit).unwrap_or(u64::MAX),
                at_block.block_number(),
            );
            if blocks > 0 {
                return Ok(ServeAxonOutcome::RateLimited { blocks });
            }
        }

        let tx = subxt::dynamic::tx(
            PALLET,
            "serve_axon",
            vec![
                Value::from(self.netuid as u64),
                Value::from(0u64),
                Value::from(ip_int),
                Value::from(port as u64),
                Value::from(4u64),
                Value::from(protocol as u64),
                Value::from(0u64),
                Value::from(0u64),
            ],
        );

        let signer = crate::weights::SubxtSr25519Signer::new(wallet)?;

        let mut tx_client = client.tx().await.context("acquiring transactions client")?;
        let progress = tokio::time::timeout(
            TX_SUBMIT_TIMEOUT,
            tx_client.sign_and_submit_then_watch_default(&tx, &signer),
        )
        .await
        .context("serve_axon submission timed out")?
        .context("submitting serve_axon")?;
        let result = tokio::time::timeout(
            TX_FINALIZATION_TIMEOUT,
            progress.wait_for_finalized_success(),
        )
        .await
        .context("serve_axon finalization timed out")?
        .context("serve_axon finalization")?;

        info!(
            block = %result.extrinsic_hash(),
            ip = %ip,
            port = port,
            "axon registered on chain"
        );

        Ok(ServeAxonOutcome::Served { replaced: current })
    }
}

/// Blocks to wait before a `serve_axon` submitted now passes the subnet's
/// serving rate limit.
///
/// Subtensor accepts the call when `rate_limit == 0 || last_serve == 0 ||
/// block - last_serve >= rate_limit`, where `block` is the block that executes
/// it. That is at least one past the finalized block the caller read.
fn serving_rate_limit_wait(last_serve: u64, rate_limit: u64, finalized_block: u64) -> u64 {
    if rate_limit == 0 || last_serve == 0 {
        return 0;
    }
    last_serve
        .saturating_add(rate_limit)
        .saturating_sub(finalized_block.saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_waits_until_the_executing_block_is_far_enough() {
        // Allowed from block 150; the earliest executing block is finalized + 1.
        assert_eq!(serving_rate_limit_wait(100, 50, 120), 29);
        assert_eq!(serving_rate_limit_wait(100, 50, 148), 1);
        assert_eq!(serving_rate_limit_wait(100, 50, 149), 0);
        assert_eq!(serving_rate_limit_wait(100, 50, 400), 0);
    }

    #[test]
    fn rate_limit_does_not_apply_to_a_first_serve_or_a_zero_limit() {
        assert_eq!(serving_rate_limit_wait(0, 50, 10), 0);
        assert_eq!(serving_rate_limit_wait(100, 0, 101), 0);
        assert_eq!(serving_rate_limit_wait(u64::MAX, u64::MAX, 0), u64::MAX - 1);
    }

    #[test]
    fn decodes_an_axons_entry() {
        let entry = Value::named_composite([
            ("block", Value::u128(4_000_123)),
            ("version", Value::u128(0)),
            (
                "ip",
                Value::u128(u32::from(Ipv4Addr::new(203, 0, 113, 7)) as u128),
            ),
            ("port", Value::u128(8091)),
            ("ip_type", Value::u128(4)),
            ("protocol", Value::u128(4)),
            ("placeholder1", Value::u128(0)),
            ("placeholder2", Value::u128(0)),
        ]);
        assert_eq!(
            ChainAxon::from_value(&entry),
            ChainAxon {
                ip: IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7)),
                port: 8091,
                protocol: 4,
                block: 4_000_123,
            }
        );
    }
}
