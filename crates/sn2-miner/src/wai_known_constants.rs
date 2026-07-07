use std::collections::HashMap;
use std::sync::LazyLock;

/// Architecture-fixed weights-as-inputs values (GELU's constants, frozen
/// backbone biases, positional-encoding tables) that the validator's
/// dispatch payload does not include and that a donor-sourced circuit's
/// local ONNX copy cannot supply either, because these values only exist
/// after this circuit's own slicing/constant-folding step -- a step the
/// miner never runs for donor-sourced weights-as-inputs circuits. Recovered
/// once offline from the base model and its frozen backbone, keyed by the
/// exact tensor name the compiled circuit declares.
static KNOWN_CONSTANTS: LazyLock<HashMap<String, (Vec<usize>, Vec<f64>)>> = LazyLock::new(|| {
    let raw: serde_json::Value =
        serde_json::from_str(include_str!("../data/wai_known_constants.json"))
            .expect("wai_known_constants.json must parse as JSON");
    let obj = raw
        .as_object()
        .expect("wai_known_constants.json must be a JSON object");
    obj.iter()
        .filter_map(|(name, entry)| {
            let shape: Vec<usize> = entry
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            let values: Vec<f64> = entry
                .get("values")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            Some((name.clone(), (shape, values)))
        })
        .collect()
});

/// Returns `(values, shape)` for a known-fixed weights-as-inputs tensor, if
/// this exact name is in the table.
pub fn lookup(name: &str) -> Option<(Vec<f64>, Vec<usize>)> {
    KNOWN_CONSTANTS
        .get(name)
        .map(|(shape, values)| (values.clone(), shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_loads_and_is_nonempty() {
        assert!(!KNOWN_CONSTANTS.is_empty());
    }

    #[test]
    fn known_gelu_constant_resolves() {
        let (values, shape) = lookup(
            "/backbone/backbone.0/encoder/encoder/encoder/layer.0/mlp/activation/Constant_output_0",
        )
        .expect("sqrt(2) GELU constant must be present");
        assert_eq!(shape, Vec::<usize>::new());
        assert!((values[0] - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn known_bias_tensor_resolves_with_correct_length() {
        let (values, shape) =
            lookup("backbone.0.encoder.encoder.encoder.layer.8.mlp.fc1.bias")
                .expect("layer 8 fc1 bias must be present");
        assert_eq!(shape, vec![1536]);
        assert_eq!(values.len(), 1536);
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(lookup("not.a.real.tensor.name").is_none());
    }
}
