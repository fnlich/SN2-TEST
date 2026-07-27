//! Explicit skiplist of slice numbers confirmed, via an offline
//! `circuit-cost-report` run against a real circuit_cache, to require an
//! unusually high constraint count. Proving one of these ties up a
//! capacity slot for far longer than a typical slice; the validator's own
//! adaptive-cap accounting penalizes that indirectly (a slow attempt burns
//! a slot without proportionally raising `delivered_rate_estimate`), so
//! declining them outright -- before spending any CPU on witness/proof
//! generation -- keeps this miner's measured completion rate healthier
//! than attempting everything indiscriminately.
//!
//! This is a static, precomputed lookup rather than a live estimate: the
//! constraint count for a given slice is a property of the circuit's own
//! structure (declared layer shapes), not the specific model weights, so
//! a measurement taken once against one model_id's compiled bundles holds
//! for other model_ids sharing the same slice layout -- confirmed by the
//! circuit-cost-report data itself, where declared_io_elements is
//! identical across model_ids for matching slice_nums.
//!
//! Populated from the heaviest 12 slices (>= HEAVY_SLICE_CONSTRAINT_THRESHOLD
//! constraints) observed for model_33894054aeabfc336348248a9dfdb7b9aca97e0213b6e1da13c50080fc2a8d27,
//! the one model_id in that report run with real (non-estimate-failed,
//! non-zero) constraint counts across all its slices. To refresh: rerun
//! circuit-cost-report against a model with fully-cached, successfully
//! estimated slices and update both the threshold and this list together.

pub const HEAVY_SLICE_CONSTRAINT_THRESHOLD: u64 = 15_012_336;

pub fn is_heavy_slice(slice_num: &str) -> bool {
    HEAVY_SLICE_NUMS.contains(&slice_num)
}

const HEAVY_SLICE_NUMS: &[&str] = &[
    "slice_406", // 39_628_800
    "slice_407", // 39_628_800
    "slice_364", // 39_628_800
    "slice_316", // 39_628_800
    "slice_408", // 39_628_800
    "slice_365", // 39_628_800
    "slice_205", // 26_239_200
    "slice_75",  // 26_239_200
    "slice_140", // 26_239_200
    "slice_204", // 15_012_336
    "slice_139", // 15_012_336
    "slice_74",  // 15_012_336
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_heavy_slices_are_flagged() {
        for slice_num in HEAVY_SLICE_NUMS {
            assert!(is_heavy_slice(slice_num), "{slice_num} should be flagged heavy");
        }
    }

    #[test]
    fn light_slice_is_not_flagged() {
        assert!(!is_heavy_slice("slice_1"));
        assert!(!is_heavy_slice("slice_41")); // 11_581_440, just below the threshold
    }

    #[test]
    fn exactly_twelve_entries() {
        assert_eq!(HEAVY_SLICE_NUMS.len(), 12);
    }
}
