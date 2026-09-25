# Patched btlightning 0.2.15

Source: the crates.io `btlightning` 0.2.15 package, which matches tag `v0.2.15`
(commit `53c090b`) of https://github.com/inference-labs-inc/btlightning.
It is wired in with `[patch.crates-io]` in the workspace `Cargo.toml` and kept
out of the workspace members so the upstream code is not linted or tested as
part of this repo. MIT licensed, see `LICENSE`.

## Why

The server remembered an authenticated validator by the client address its
QUIC connection had during the handshake (`addr_to_hotkey[remote_address]`), and
checked every request against the connection's *current* address. quinn leaves
server-side connection migration enabled, so when a validator's NAT rebinds its
UDP source port (seen with a validator egressing through Cloudflare WARP,
104.28.0.0/16), the live connection moves to the new port. Every later request
on it was rejected with `Unknown or unauthenticated connection from <ip>:<new
port>` and answered "authentication failed". The client treats that as an
ordinary failed response: it neither reconnects nor re-handshakes, so the
rejections continued until the connection closed, and each one counted as a
failed task for that validator.

The same address keying also left the handshake-time address in the index after
a migration. A different peer later handed that address (a recycled port on a
shared NAT) would have been treated as the validator without a handshake.

## What changed

- `src/server/dispatch.rs` `verify_synapse_auth`: authentication is bound to the
  connection object and the client address is not consulted at all. A request
  is served only if it arrives on the verified connection that completed the
  handshake, however many times that connection's address has changed.
- `src/server/handshake.rs`: a successful handshake drops every other address
  the index holds for that validator, not only the replaced connection's current
  one. This covers a re-handshake on a connection that had migrated.
- `src/server/mod.rs` `remove_hotkey_from_maps`: removes all of the validator's
  addresses, so a migrated connection does not leave its handshake-time address
  behind.
- `src/server/mod.rs` accept loop and `src/server/dispatch.rs` per-stream tasks
  run in the caller's tracing span (`in_current_span`), so when one process
  serves several miners, connection and handshake logs carry the miner's span.
- Unit tests for the above in `src/server/mod.rs`. The end-to-end NAT rebinding
  test lives in `crates/sn2-miner/tests/quic_nat_rebind.rs`.

Behavior to be aware of: an authenticated connection that migrates, including
to a different IP, keeps being served. Only the
holder of the connection's QUIC keys can move it, so this is not an
authentication bypass.

## Testing

From this directory:

    cargo test --lib
    cargo test --test integration

Drop this directory and the `[patch.crates-io]` entry once an upstream release
carries an equivalent fix.
