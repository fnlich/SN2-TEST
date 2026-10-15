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
| `--external-ip` | Detected | Public IPv4 to register on-chain for the axon. When unset it is detected on every registration attempt (api4.ipify.org, then checkip.amazonaws.com, then ipv4.icanhazip.com). Validators ignore axons whose IP is not publicly routable |
| `--miner` | None | Serve several miners from one process: `[WALLET/]HOTKEY:PORT`, repeatable or comma-separated. Cannot be combined with `--axon-port`; `--wallet-hotkey` is ignored (with a warning) |
| `--max-concurrent-proofs` | Number of CPUs | Most proofs generated at once, across all miners in the process. Requests beyond it wait in arrival order |
| `--circuit-cache-dir` | `~/.bittensor/subnet-2/circuit_cache` | Circuit cache directory (also `SN2_CIRCUIT_CACHE_DIR`) |
| `--additional-circuits` | None | Circuit IDs to preload at startup |
| `--handler-timeout` | `180` | Per-request handler timeout in seconds |
| `--loopback` | `false` | Run without chain interaction, for local testing |

### Axon registration

After its QUIC server is listening, each miner registers its axon (IP, port, protocol 4) on chain and keeps it registered:

- Failed attempts are retried without limit, every 15 seconds at first and backing off to every 5 minutes. Each attempt opens a new chain connection and logs the full error.
- If the subnet's serving rate limit (50 blocks, about 10 minutes, on subnet 2) does not yet allow the hotkey to change its axon, the miner waits until it does instead of sending a transaction the chain would reject.
- A hotkey that is not registered on the subnet is still served, and its axon is registered as soon as the hotkey has a uid. Registering a hotkey does not need a miner restart.
- Once registered, the axon is checked against the chain every 10 minutes and registered again if the chain entry changed, for example because the detected public IP moved. If the machine's outbound IP is not the address validators should connect to, set `--external-ip`.

### Running several miners in one process

Each `--miner` entry is a separate miner with its own hotkey and QUIC port. All of them share one circuit cache and one prover with its caches, so adding a miner does not download circuits or load prover state again. `WALLET` defaults to `--wallet-name`.

```console
pm2 start target/release/sn2-miner --name subnet-2-miner --kill-timeout 3000 -- \
  --wallet-name miner \
  --miner hk1:8091,hk2:8092,hk3:8093 \
  --netuid 2
```

- Every hotkey needs its own registration. A hotkey that is not registered yet is served anyway, and its axon is registered once it is (see [Axon registration](#axon-registration)).
- Each miner's axon is registered at `--external-ip` (or the detected public IP) with that miner's port.
- The miners' proofs share `--max-concurrent-proofs`, which defaults to the number of CPUs. A proof runs on a single core, so running more proofs than cores slows every request at once and multiplies the RAM held by in-flight proofs; requests beyond the limit wait in arrival order instead. A request whose validator timed out while it was still proving keeps its slot until the proof finishes, because a running proof cannot be cancelled.
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
