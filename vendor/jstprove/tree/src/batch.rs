//! MODIFIED FOR SN2 (new file, not part of JSTprove 87a1859f). See PATCHED.md
//! at the root of the vendored copy.
//!
//! Keccak-256 of many independent 64-byte messages, eight at a time. Every
//! leaf of a Merkle tree is one 64-byte message, and so is every pair of
//! sibling nodes (2 x 32 bytes). The messages of one tree level do not depend
//! on each other, so they go through Keccak-f[1600] in the lanes of one SIMD
//! state. The digests are the bytes `Leaf::leaf_hash` and `Node::node_hash`
//! compute with tiny-keccak.

use keccak::simd::u64x8;

use crate::{Leaf, Node, LEAF_BYTES, LEAF_HASH_BYTES};

const LANES: usize = 8;
const STATE_WORDS: usize = 25;
/// Keccak-256 absorbs 136 bytes (17 words) per permutation.
const RATE_WORDS: usize = 17;
const MESSAGE_WORDS: usize = LEAF_BYTES / 8;

/// Keccak-256 state after absorbing one 64-byte message, with the original
/// Keccak padding: 0x01 after the message and 0x80 in the last rate byte.
#[inline(always)]
fn absorb(message: &[u8; LEAF_BYTES]) -> [u64; STATE_WORDS] {
    let mut state = [0u64; STATE_WORDS];
    for (word, bytes) in state.iter_mut().zip(message.chunks_exact(8)) {
        *word = u64::from_le_bytes(bytes.try_into().unwrap());
    }
    state[MESSAGE_WORDS] ^= 0x01;
    state[RATE_WORDS - 1] ^= 0x80 << 56;
    state
}

#[inline(always)]
fn squeeze(digest: &mut [u8; LEAF_HASH_BYTES], word: impl Fn(usize) -> u64) {
    for (w, bytes) in digest.chunks_exact_mut(8).enumerate() {
        bytes.copy_from_slice(&word(w).to_le_bytes());
    }
}

/// Sets `digests[i]` to the Keccak-256 of message `i`, which `message(i, buf)`
/// writes into `buf`.
pub(crate) fn keccak256_64_many(
    digests: &mut [[u8; LEAF_HASH_BYTES]],
    message: impl Fn(usize, &mut [u8; LEAF_BYTES]),
) {
    let mut buf = [0u8; LEAF_BYTES];
    let mut chunks = digests.chunks_exact_mut(LANES);
    let mut first = 0;
    for chunk in &mut chunks {
        let mut lanes = [[0u64; STATE_WORDS]; LANES];
        for (k, lane) in lanes.iter_mut().enumerate() {
            message(first + k, &mut buf);
            *lane = absorb(&buf);
        }
        let mut state = [u64x8::splat(0); STATE_WORDS];
        for (w, word) in state.iter_mut().enumerate().take(RATE_WORDS) {
            *word = u64x8::from_array(core::array::from_fn(|k| lanes[k][w]));
        }
        keccak::simd::f1600x8(&mut state);
        for (k, digest) in chunk.iter_mut().enumerate() {
            squeeze(digest, |w| state[w].as_array()[k]);
        }
        first += LANES;
    }
    for (k, digest) in chunks.into_remainder().iter_mut().enumerate() {
        message(first + k, &mut buf);
        let mut state = absorb(&buf);
        keccak::f1600(&mut state);
        squeeze(digest, |w| state[w]);
    }
}

/// `leaf.leaf_hash()` for every leaf.
pub(crate) fn hash_leaves(leaves: &[Leaf]) -> Vec<Node> {
    let mut digests = vec![[0u8; LEAF_HASH_BYTES]; leaves.len()];
    keccak256_64_many(&mut digests, |i, buf| *buf = leaves[i].data);
    digests.into_iter().map(Node::new).collect()
}

/// Sets `parents[i]` to `Node::node_hash(&children[2 * i], &children[2 * i + 1])`.
pub(crate) fn hash_node_pairs(children: &[Node], parents: &mut [Node]) {
    assert_eq!(children.len(), 2 * parents.len());
    let mut digests = vec![[0u8; LEAF_HASH_BYTES]; parents.len()];
    keccak256_64_many(&mut digests, |i, buf| {
        buf[..LEAF_HASH_BYTES].copy_from_slice(&children[2 * i].data);
        buf[LEAF_HASH_BYTES..].copy_from_slice(&children[2 * i + 1].data);
    });
    for (parent, digest) in parents.iter_mut().zip(digests) {
        *parent = Node::new(digest);
    }
}

#[cfg(test)]
mod tests {
    use tiny_keccak::{Hasher, Keccak};

    use super::*;
    use crate::Tree;

    fn tiny_keccak256(message: &[u8]) -> [u8; LEAF_HASH_BYTES] {
        let mut hasher = Keccak::v256();
        hasher.update(message);
        let mut digest = [0u8; LEAF_HASH_BYTES];
        hasher.finalize(&mut digest);
        digest
    }

    fn message(i: usize) -> [u8; LEAF_BYTES] {
        core::array::from_fn(|j| (i * 131 + j * 7 + 3) as u8)
    }

    #[test]
    fn digests_match_tiny_keccak_for_full_and_partial_batches() {
        for n in 0..=3 * LANES + 1 {
            let mut digests = vec![[0u8; LEAF_HASH_BYTES]; n];
            keccak256_64_many(&mut digests, |i, buf| *buf = message(i));
            for (i, digest) in digests.iter().enumerate() {
                assert_eq!(*digest, tiny_keccak256(&message(i)), "n = {n}, message {i}");
            }
        }
    }

    /// The tree as `new_with_leaves` built it before the batched hashing.
    fn reference_nodes(leaves: &[Leaf]) -> Vec<Node> {
        let leaf_nodes: Vec<Node> = leaves.iter().map(Leaf::leaf_hash).collect();
        let mut nodes = vec![Node::default(); leaves.len() - 1];
        let mut level = leaf_nodes.clone();
        let mut end = nodes.len();
        while level.len() > 1 {
            let parents: Vec<Node> = level
                .chunks_exact(2)
                .map(|pair| Node::node_hash(&pair[0], &pair[1]))
                .collect();
            let start = end - parents.len();
            nodes[start..end].copy_from_slice(&parents);
            end = start;
            level = parents;
        }
        nodes.extend(leaf_nodes);
        nodes
    }

    #[test]
    fn tree_nodes_match_unbatched_hashing() {
        for height in 1..=9 {
            let leaves: Vec<Leaf> = (0..1usize << (height - 1))
                .map(|i| Leaf::new(message(i)))
                .collect();
            let tree = Tree::new_with_leaves(leaves.clone());
            assert_eq!(tree.nodes, reference_nodes(&leaves), "height {height}");
        }
    }
}
