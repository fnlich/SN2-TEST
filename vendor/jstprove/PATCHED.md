# Patched JSTprove `tree` and `poly_commit`

Source: `prover/tree` and `prover/poly_commit` of
https://github.com/inference-labs-inc/JSTprove at commit
`87a1859f3487cf0fb9a463dbfd713b1df4827afc`, the rev dsperse `3a4fbf080bb3`
pins. Only `src/` and `Cargo.toml` of each crate are copied. They are wired in
with `[patch."https://github.com/inference-labs-inc/JSTprove.git"]` in the
workspace `Cargo.toml` and kept out of the workspace members, so the upstream
code is not linted or tested as part of this repo. Every other JSTprove crate
still comes from the pinned git rev.

JSTprove is Copyright (c) 2025 Inference Labs Inc. and distributed under the
Inference Labs Source Access Grant, see `LICENSE` (the same text as
`LICENSES/LicenseRef-Inference-Labs-Source-Grant.txt`). These copies are
redistributed under that license. Each file changed here starts with a
`MODIFIED FOR SN2` notice, and `src/batch.rs` in `tree` is new.

## Why

Every production rfdetr-nano slice is proved with `goldilocks_ext4_whir`
(non-holographic). Profiling the miner's per-request path showed WHIR's
opening dominating proof time. Most of it went to hashing Merkle tree nodes one
at a time with tiny-keccak, and to inverse-FFTing each re-encoded codeword once
more for every out-of-domain sample.

## What changed

- `tree/src/batch.rs` (new), `tree/src/tree.rs`, `tree/src/lib.rs`,
  `tree/Cargo.toml`: `Tree::new_with_leaves` and `new_with_leaf_nodes` hash the
  leaves and each level of nodes with an 8-lane Keccak-f[1600]
  (`keccak::simd::f1600x8`, `keccak` 0.1.6 with the `simd` feature, which needs
  nightly `portable_simd`). The nodes of one level do not depend on each
  other. The digests are the Keccak-256 values `Leaf::leaf_hash` and
  `Node::node_hash` compute, which stay as they were for paths and
  verification.
- `poly_commit/src/whir/pcs_trait_impl.rs`, in `whir_open`:
  - `re_encode` also returns the coefficients it interpolates. The
    out-of-domain samples are evaluated from them instead of interpolating the
    whole re-encoded codeword again for each sample (`evaluate_ood`, removed).
    The coefficients it no longer evaluates, above the polynomial's degree,
    are zero.
  - The re-encoding FFT and inverse FFT multiply by base-field twiddles
    (`mul_by_base_field`) when the extension field's generator of the FFT
    subgroup is the embedded base-field generator. That holds for
    `GoldilocksExt4` and `GoldilocksExt3` at every size, and for
    `GoldilocksExt2` up to 2^32. Otherwise they fall back to the upstream
    transforms.
- Both `Cargo.toml` files spell out what they inherited from the JSTprove
  workspace (same versions and features). They point the other JSTprove
  crates at the pinned git rev and drop the benches.

Nothing changes in what is computed, only how. Proofs are byte-identical to
upstream's, so validators running unmodified JSTprove accept them.

## How it was checked

- Unit tests added here, run in a full JSTprove checkout with these files:
  - batched digests against tiny-keccak for 0 to 25 messages;
  - whole trees of heights 1 to 9 against the unbatched construction;
  - the FFT, inverse FFT, re-encoding and out-of-domain values against
    upstream's code for GoldilocksExt2, 3 and 4.

  All unit tests of both crates pass, JSTprove's own included: 112 in
  `poly_commit` (101 of them WHIR round trips and adversarial cases) and 3 in
  `tree`.
- Production slices 20, 21, 74 and 153 of rfdetr-nano, proved by the miner's
  request path with the same build flags as this repo:
  - the proof, witness and output bytes are identical to an unpatched build's;
  - a validator-equivalent unpatched verifier accepts every proof.
- Measured on an Intel AVX-512 test host (not Turin), one core, miner request
  path (witness + prove), interleaved runs:

  | Slice | neither change | `-vectorize-slp=false` | flag + this patch |
  |---|---|---|---|
  | slice_20 (about 70% of production proving CPU) | 1.31 s | 0.80 s | 0.46 s |
  | slice_21 | 0.195 s | 0.100 s | 0.069 s |
  | slice_74 | 11.5 s | 7.9 s | 4.9 s |

## Updating

The `[patch]` matches JSTprove's git URL, not its rev, and JSTprove's crates
depend on `tree` and `poly_commit` without a version requirement. So when
dsperse moves to another JSTprove rev, Cargo still uses these copies. They
pull in a second copy of the other JSTprove crates at 87a1859f, and **the
build fails** with type errors and notes about multiple different versions of
crate `arith`. Cargo gives no warning beforehand, and there is no fallback to
the unpatched crates.

Before building a change that moves dsperse's JSTprove rev (check the
`source` lines of the JSTprove packages in `Cargo.lock`), do one of these:

- Re-vendor, to keep the speedup:
  1. Copy `src/` and `Cargo.toml` of the two crates from the new rev.
  2. Re-apply the changes marked `MODIFIED FOR SN2`.
  3. Update the `rev` in both manifests.
  4. Before deploying, check that proofs are still byte-identical to an
     unpatched build's.
- Or go back to stock JSTprove: delete the
  `[patch."https://github.com/inference-labs-inc/JSTprove.git"]` block in the
  workspace `Cargo.toml` and `vendor/jstprove`.

If upstream adopts equivalent changes, remove the patch the same way.
