# Command Line Arguments

All arguments use `--long-flag` syntax. Flags are shared between miner and validator unless noted otherwise.

## Shared Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--netuid` | `2` | The subnet UID |
| `--network` | `finney` | Network to connect to: `finney`, `test`, `local`, or a custom endpoint |
| `--subtensor-chain-endpoint` | Derived from `--network` | Override the subtensor WebSocket endpoint directly |
| `--wallet-name` | `default` | Bittensor wallet name |
| `--wallet-hotkey` | `default` | Wallet hotkey name |
| `--wallet-path` | `~/.bittensor/wallets` | Path to wallet directory |
| `--log-level` | `info` | Tracing filter directive (e.g. `debug`, `warn`, `sn2_validator=trace`) |
| `--no-auto-update` | `false` | Disable the built-in binary auto-update mechanism |

## Miner Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--axon-host` | `0.0.0.0` | Bind address for the QUIC server in `--loopback` mode |
| `--axon-port` | `8091` | QUIC ([btlightning](https://github.com/inference-labs-inc/lightning)) server port (UDP), registered on-chain as the axon port |
| `--external-ip` | None | Public IP to register on-chain for the axon |
| `--miner` | None | Serve several miners from one process: `[WALLET/]HOTKEY:PORT`, repeatable or comma-separated. Cannot be combined with `--axon-port`; `--wallet-hotkey` is ignored (with a warning) |
| `--circuit-cache-dir` | `~/.bittensor/subnet-2/circuit_cache` | Circuit cache directory (also `SN2_CIRCUIT_CACHE_DIR`) |
| `--additional-circuits` | None | Circuit IDs to preload at startup |
| `--handler-timeout` | `180` | Per-request handler timeout in seconds |
| `--loopback` | `false` | Run without chain interaction, for local testing |

### Running several miners in one process

Each `--miner` entry is a separate miner with its own hotkey and QUIC port. All of them share one circuit cache, one prover and its caches, and one chain connection, so adding a miner does not download circuits or load prover state again, and the miners do not compete with each other's prover thread pools as separate processes would. `WALLET` defaults to `--wallet-name`.

```console
pm2 start target/release/sn2-miner --name subnet-2-miner --kill-timeout 3000 -- \
  --wallet-name miner \
  --miner hk1:8091,hk2:8092,hk3:8093 \
  --netuid 2
```

- Every hotkey needs its own registration. Hotkeys that are not registered when the miner starts are skipped with a warning; restart the miner after registering one. Startup fails only if none is registered.
- Each miner's axon is registered at `--external-ip` (or the detected public IP) with that miner's port.
- Open every port for UDP, for example `sudo ufw allow 8091:8093/udp` plus matching UDP rules in the cloud security group. With Docker, publish each port as UDP with the same host and container port, for example `-p 8091-8093:8091-8093/udp`.
- `make pm2-miner ARGS="--miner ..."` always passes `--wallet-hotkey`, so the "--wallet-hotkey is ignored" warning is expected there.
- The miners share one process, so a restart disconnects all of them at once. Restart rarely and batch hotkey changes.

**When extra miners help.** Validators dispatch work per UID, so more hotkeys bring more work only while the validator's per-miner dispatch, not this machine's CPU, is the limit. Scores grow faster than linearly with each UID's delivered work, so splitting a machine that is already busy across more hotkeys lowers total reward. A new hotkey also earns nothing during its first hours (verification coldstart and the minimum sample count). Add miners only while the machine has steady idle capacity, and compare total delivered work before and after.

## Validator Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-concurrency` | `32` | Maximum concurrent miner queries |
| `--api-miners-pct` | `20` | Percentage of miners allocated to API requests |
| `--disable-benchmark` | `false` | Disable benchmark queries |
| `--relay-url` | None | WebSocket relay URL |
| `--no-relay` | `false` | Disable the relay WebSocket connection (enabled by default) |
| `--metrics-port` | `9090` | Prometheus metrics exporter port |
| `--dsperse-socket` | None | dsperse prover socket address |

## Environment Variables

Tracing can be configured via the `RUST_LOG` environment variable, which takes precedence over `--log-level`. The syntax follows the [tracing-subscriber `EnvFilter` directives](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html).

```console
RUST_LOG=debug sn2-validator --netuid 2
RUST_LOG=sn2_miner=trace,sn2_chain=debug sn2-miner --netuid 2
```
