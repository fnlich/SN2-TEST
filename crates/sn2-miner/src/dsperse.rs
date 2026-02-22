use std::path::PathBuf;

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use tracing::info;

pub struct DSperseClient {
    cache_dir: PathBuf,
}

impl DSperseClient {
    pub fn new() -> Self {
        let cache_dir = PathBuf::from(shellexpand::tilde(sn2_types::CIRCUIT_CACHE_DIR).to_string());
        info!(cache_dir = %cache_dir.display(), "initialized DSperseClient");
        Self { cache_dir }
    }

    pub async fn prove_slice(
        &self,
        circuit_id: &str,
        slice_num: &str,
        inputs: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let slices_dir = self
            .cache_dir
            .join(format!("model_{circuit_id}"))
            .join("slices");
        let slice_idx: usize = slice_num.parse().context("parsing slice_num")?;
        let slice_id = format!("slice_{slice_idx}");

        dsperse::archive::extract_single_slice(&slices_dir, &slice_id, None)
            .map_err(|e| anyhow::anyhow!("extracting {slice_id}: {e}"))?;

        let dslice_file = slices_dir.join(format!("{slice_id}.dslice"));
        let slice_meta = dsperse::archive::read_dslice_slice_metadata(&dslice_file)
            .map_err(|e| anyhow::anyhow!("reading slice metadata: {e}"))?;

        anyhow::ensure!(
            slice_meta.compilation.jstprove.compiled,
            "slice {slice_idx} not jstprove-compiled"
        );

        let slice_dir = dsperse::utils::paths::slice_dir_path(&slices_dir, slice_idx);

        let compiled = slice_meta
            .compilation
            .jstprove
            .files
            .compiled
            .as_ref()
            .with_context(|| format!("no compiled circuit for slice {slice_idx}"))?;
        let circuit_path = slice_dir.join("jstprove").join(compiled);

        let onnx_path = slice_dir.join(&slice_meta.path);

        anyhow::ensure!(
            circuit_path.exists(),
            "circuit not found at {}",
            circuit_path.display()
        );
        anyhow::ensure!(
            onnx_path.exists(),
            "onnx model not found at {}",
            onnx_path.display()
        );

        info!(
            circuit_id,
            slice = slice_num,
            circuit_path = %circuit_path.display(),
            "generating witness and proof"
        );

        let input_data = dsperse::utils::io::extract_input_data(inputs)
            .context("no recognized input key in inputs JSON")?
            .clone();

        tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
            let input_tensor = dsperse::utils::io::json_to_arrayd(&input_data)
                .map_err(|e| anyhow::anyhow!("parsing input tensor: {e}"))?;
            let input_flat: Vec<f64> = input_tensor.iter().copied().collect();
            let input_shape: Vec<usize> = input_tensor.shape().to_vec();

            let (output_data, output_shape) =
                dsperse::backend::onnx::run_inference(&onnx_path, &input_flat, &input_shape)
                    .map_err(|e| anyhow::anyhow!("onnx inference: {e}"))?;
            let output_tensor = ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
                .context("reshaping onnx output")?;

            let input_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "input_data": dsperse::utils::io::arrayd_to_json(&input_tensor) }),
            )?;
            let output_json_bytes = serde_json::to_vec(
                &serde_json::json!({ "output_data": dsperse::utils::io::arrayd_to_json(&output_tensor) }),
            )?;

            let backend = dsperse::backend::JstproveBackend::new();

            let witness_bytes = backend
                .witness(&circuit_path, &input_json_bytes, &output_json_bytes)
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let proof_bytes = backend
                .prove(&circuit_path, &witness_bytes)
                .map_err(|e| anyhow::anyhow!("proof generation: {e}"))?;

            info!(
                witness_size = witness_bytes.len(),
                proof_size = proof_bytes.len(),
                "witness and proof generated"
            );

            Ok(serde_json::json!({
                "proof": hex::encode(&proof_bytes),
                "witness": hex::encode(&witness_bytes),
            }))
        })
        .await
        .context("blocking task panicked")?
    }

    pub async fn prove(
        &self,
        model_id: &str,
        inputs: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let model_dir = self.cache_dir.join(format!("model_{model_id}"));
        let circuit_path = model_dir.join("model.compiled");

        anyhow::ensure!(
            circuit_path.exists(),
            "compiled model not found at {}",
            circuit_path.display()
        );

        info!(
            model_id,
            circuit_path = %circuit_path.display(),
            "generating witness and proof"
        );

        let inputs_bytes = serde_json::to_vec(inputs)?;

        tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
            let backend = dsperse::backend::JstproveBackend::new();

            let witness_bytes = backend
                .witness(&circuit_path, &inputs_bytes, &[])
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let proof_bytes = backend
                .prove(&circuit_path, &witness_bytes)
                .map_err(|e| anyhow::anyhow!("proof generation: {e}"))?;

            Ok(serde_json::json!({
                "proof": hex::encode(&proof_bytes),
                "witness": hex::encode(&witness_bytes),
            }))
        })
        .await
        .context("blocking task panicked")?
    }
}
