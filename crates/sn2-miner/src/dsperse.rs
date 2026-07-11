use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{Context, Result};
use sn2_types::json_tensor::flatten_json_to_f64;
use tracing::info;

pub struct DSperseClient {
    cache_dir: PathBuf,
    /// Shared across every request so its internal bundle cache
    /// (`load_bundle_cached`) actually gets reused instead of being
    /// discarded and rebuilt on every single call. A circuit bundle is
    /// tens of megabytes read and parsed from disk; a fresh backend per
    /// request means a fresh cache-miss every time, even for a circuit
    /// that was just proved a moment ago. Mirrors the pattern already
    /// used by `sn2-verify`'s validator-side `BACKEND` static.
    backend: Arc<dsperse::backend::jstprove::JstproveBackend>,
    /// Caches `resolve_component`'s (component_sha, slice_id) -> slice_dir
    /// lookups, avoiding a full scan of every locally cached model
    /// directory on repeat DSlice requests for the same component.
    component_cache: RwLock<HashMap<(String, String), PathBuf>>,
    /// Caches `find_slice_onnx`'s slice_dir -> onnx_path resolution,
    /// avoiding a directory listing on every repeat prove_slice call.
    onnx_cache: RwLock<HashMap<PathBuf, PathBuf>>,
}

fn validate_circuit_id(id: &str) -> Result<()> {
    anyhow::ensure!(
        id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid circuit id: expected 64-char hex string"
    );
    Ok(())
}

pub fn normalize_slice_id(slice_num: &str) -> Result<String> {
    let idx: usize = slice_num
        .strip_prefix("slice_")
        .unwrap_or(slice_num)
        .parse()
        .context("parsing slice_num")?;
    Ok(format!("slice_{idx}"))
}

fn find_slice_onnx(slice_dir: &Path) -> Result<PathBuf> {
    let payload_dir = slice_dir.join("payload");
    if payload_dir.is_dir() {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(&payload_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
            .collect();
        candidates.sort();
        match candidates.len() {
            1 => return Ok(candidates.remove(0)),
            n if n > 1 => anyhow::bail!(
                "multiple .onnx files in {}: {:?}",
                payload_dir.display(),
                candidates,
            ),
            _ => {}
        }
    }
    anyhow::bail!("no .onnx file found in {}", payload_dir.display())
}

fn extract_input_json(inputs: &serde_json::Value) -> &serde_json::Value {
    for key in &["input_data", "input", "data", "inputs"] {
        if let Some(v) = inputs.get(*key) {
            return v;
        }
    }
    inputs
}

pub struct ProveArtifacts {
    pub proof: Vec<u8>,
    pub witness: Vec<u8>,
    pub computed_outputs: Vec<f64>,
}

fn prove_and_build_response(
    backend: &dsperse::backend::jstprove::JstproveBackend,
    circuit_path: &Path,
    witness_bytes: &[u8],
    effective_input_dims: Option<usize>,
) -> Result<ProveArtifacts> {
    let holographic = circuit_path.join("vk.bin").is_file();
    let proof_bytes = if holographic {
        backend
            .prove_holographic(circuit_path, witness_bytes)
            .map_err(|e| anyhow::anyhow!("holographic proof generation: {e}"))?
    } else {
        backend
            .prove(circuit_path, witness_bytes)
            .map_err(|e| anyhow::anyhow!("proof generation: {e}"))?
    };

    let computed_outputs = if let Some(num_model_inputs) = effective_input_dims {
        match backend.extract_outputs(witness_bytes, num_model_inputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    num_model_inputs,
                    witness_len = witness_bytes.len(),
                    "output extraction failed; validator will extract from verified proof"
                );
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    info!(
        witness_size = witness_bytes.len(),
        proof_size = proof_bytes.len(),
        num_outputs = computed_outputs.len(),
        holographic,
        "witness and proof generated"
    );

    Ok(ProveArtifacts {
        proof: proof_bytes,
        witness: witness_bytes.to_vec(),
        computed_outputs,
    })
}

impl DSperseClient {
    pub fn new(cache_dir_override: Option<&str>) -> Self {
        let cache_dir = PathBuf::from(
            shellexpand::tilde(cache_dir_override.unwrap_or(sn2_types::CIRCUIT_CACHE_DIR))
                .to_string(),
        );
        info!(cache_dir = %cache_dir.display(), "initialized DSperseClient");

        let backend = Arc::new(dsperse::backend::jstprove::JstproveBackend::new());
        {
            let backend = Arc::clone(&backend);
            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_secs(sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS));
                loop {
                    interval.tick().await;
                    let evicted = backend.evict_idle(Duration::from_secs(
                        sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS,
                    ));
                    if evicted > 0 {
                        info!(evicted, "evicted idle compiled circuit bundles");
                    }
                }
            });
        }

        Self {
            cache_dir,
            backend,
            component_cache: RwLock::new(HashMap::new()),
            onnx_cache: RwLock::new(HashMap::new()),
        }
    }

    fn resolve_slice_onnx(&self, slice_dir: &Path) -> Result<PathBuf> {
        if let Some(cached) = self
            .onnx_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(slice_dir)
            .cloned()
        {
            if cached.exists() {
                return Ok(cached);
            }
        }
        let resolved = find_slice_onnx(slice_dir)?;
        self.onnx_cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(slice_dir.to_path_buf(), resolved.clone());
        Ok(resolved)
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub async fn resolve_component(
        &self,
        component_sha: &str,
        slice_id: &str,
    ) -> Result<Option<PathBuf>> {
        let cache_key = (component_sha.to_string(), slice_id.to_string());

        if let Some(cached) = self
            .component_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&cache_key)
            .cloned()
        {
            if cached.join("jstprove").join("circuit.bundle").is_dir() {
                return Ok(Some(cached));
            }
            // Stale (e.g. evicted from disk since caching) -- drop it and
            // fall through to a full re-resolution below.
            self.component_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&cache_key);
        }

        let cache_dir = self.cache_dir.clone();
        let scan_component_sha = component_sha.to_string();
        let scan_slice_id = slice_id.to_string();
        let resolved: Option<PathBuf> = tokio::task::spawn_blocking(move || -> Result<Option<PathBuf>> {
            let entries = match std::fs::read_dir(&cache_dir) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "reading cache directory {}: {e}",
                        cache_dir.display()
                    ))
                }
            };
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with("model_") {
                    continue;
                }
                let stamp_path = entry
                    .path()
                    .join("slices")
                    .join(&scan_slice_id)
                    .join("component.sha");
                if let Ok(stamp) = std::fs::read_to_string(&stamp_path) {
                    if stamp.trim() == scan_component_sha {
                        let slice_dir = entry.path().join("slices").join(&scan_slice_id);
                        if !slice_dir.join("jstprove").join("circuit.bundle").is_dir() {
                            continue;
                        }
                        if find_slice_onnx(&slice_dir).is_err() {
                            continue;
                        }
                        return Ok(Some(slice_dir));
                    }
                }
            }
            Ok(None)
        })
        .await
        .context("component resolution task panicked")??;

        if let Some(ref slice_dir) = resolved {
            self.component_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(cache_key, slice_dir.clone());
        }

        Ok(resolved)
    }

    pub async fn prove(
        &self,
        model_id: &str,
        inputs: &serde_json::Value,
    ) -> Result<ProveArtifacts> {
        validate_circuit_id(model_id)?;
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
            "generating witness and proof for non-composable model"
        );

        let inputs_clone = inputs.clone();
        let backend = Arc::clone(&self.backend);

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let inputs_bytes = rmp_serde::to_vec_named(&inputs_clone)?;

            let params = backend
                .load_params(&circuit_path)
                .map_err(|e| anyhow::anyhow!("loading circuit params: {e}"))?;

            let witness_bytes = backend
                .witness(&circuit_path, &inputs_bytes, &[])
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let dims = params.as_ref().map(|p| p.effective_input_dims());
            prove_and_build_response(&backend, &circuit_path, &witness_bytes, dims)
        })
        .await
        .context("blocking task panicked")?
    }

    pub async fn prove_slice(
        &self,
        circuit_id: &str,
        slice_num: &str,
        inputs: &serde_json::Value,
        resolved_component_dir: PathBuf,
    ) -> Result<ProveArtifacts> {
        validate_circuit_id(circuit_id)?;
        // Validate slice format; the normalized path is not needed since
        // resolved_component_dir already contains the canonical slice path.
        let _ = normalize_slice_id(slice_num)?;

        let slice_dir = resolved_component_dir;

        anyhow::ensure!(
            slice_dir.is_dir(),
            "resolved component directory not found at {}",
            slice_dir.display()
        );

        let circuit_path = slice_dir.join("jstprove").join("circuit.bundle");
        let onnx_path = self.resolve_slice_onnx(&slice_dir)?;

        anyhow::ensure!(
            circuit_path.is_dir(),
            "bundle directory not found at {}",
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

        let input_data = extract_input_json(inputs).clone();
        let backend = Arc::clone(&self.backend);

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let input_flat = flatten_json_to_f64(&input_data);
            anyhow::ensure!(
                !input_flat.is_empty(),
                "invalid input tensor: flattened input is empty"
            );

            let params = backend
                .load_params(&circuit_path)
                .map_err(|e| anyhow::anyhow!("loading circuit params: {e}"))?;
            let (activations, inits) = match params.as_ref() {
                Some(p) if p.weights_as_inputs => {
                    match dsperse::pipeline::split_inline_wai_inputs(p, &input_flat) {
                        Some(split) => split,
                        None => {
                            // The validator only sends the true activation
                            // tensor here, assuming every other declared
                            // input (biases, architecture constants) is
                            // independently recoverable locally. For
                            // donor-sourced circuits it isn't: those values
                            // only materialize during this circuit's own
                            // slicing/constant-folding step, which the miner
                            // never runs. Recover known-fixed values (GELU's
                            // constants, frozen backbone biases) from a
                            // static table first; anything not in it falls
                            // back to the existing file-based extraction.
                            let mut cursor = 0usize;
                            let mut activations = Vec::new();
                            let mut inits: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
                            let mut unresolved = Vec::new();

                            for io in &p.inputs {
                                let n: usize = io.shape.iter().product();
                                if dsperse::pipeline::runner::is_activation_placeholder(&io.name) {
                                    let end = (cursor + n).min(input_flat.len());
                                    activations.extend_from_slice(&input_flat[cursor..end]);
                                    cursor = end;
                                } else if let Some((values, shape)) =
                                    crate::wai_known_constants::lookup(&io.name)
                                {
                                    anyhow::ensure!(
                                        values.len() == n,
                                        "known constant '{}' has {} elements but circuit expects {}",
                                        io.name,
                                        values.len(),
                                        n
                                    );
                                    inits.push((values, shape));
                                } else {
                                    unresolved.push(io.name.clone());
                                }
                            }

                            if !unresolved.is_empty() {
                                tracing::warn!(
                                    names = ?unresolved,
                                    "known-constant table missing entries, falling back to file extraction"
                                );
                                let file_inits =
                                    dsperse::pipeline::extract_onnx_initializers(&onnx_path, p)
                                        .map_err(|e| {
                                            anyhow::anyhow!("extracting initializers: {e}")
                                        })?;
                                inits.extend(file_inits);
                            }

                            (activations, inits)
                        }
                    }
                }
                _ => (input_flat.clone(), Vec::new()),
            };

            let witness_bytes = backend
                .witness_f64(&circuit_path, &activations, &inits)
                .map_err(|e| anyhow::anyhow!("witness generation: {e}"))?;

            let dims = params.as_ref().map(|p| p.effective_input_dims());
            prove_and_build_response(&backend, &circuit_path, &witness_bytes, dims)
        })
        .await
        .context("blocking task panicked")?
    }
}

#[cfg(test)]
mod wai_fallback_integration_tests {
    use super::*;

    /// Exercises the real `prove_slice` path against slice_216 of the actual
    /// "tail" fine-tuned model (f8121b74...) -- one of the slices confirmed
    /// failing with "activation length mismatch: expected 2050, got 2048" in
    /// production. The fixture directory holds the real, repository-fetched
    /// circuit.bin/manifest.msgpack/witness_solver.bin for this slice, plus
    /// the real donor.onnx as the payload file `find_slice_onnx` requires to
    /// be present. The activation values are synthetic (their numeric
    /// content doesn't matter for this test): the point is confirming the
    /// known-constants fallback supplies the two missing scalar constants so
    /// witness generation no longer fails on element count, regardless of
    /// whether the resulting proof is otherwise meaningful.
    #[tokio::test]
    async fn slice_216_no_longer_reports_activation_length_mismatch() {
        let slice_dir = PathBuf::from(
            "/private/tmp/claude-501/-Volumes-HDD-Develop-Work-Subnet2/67c52f37-ba8f-43ed-a9d9-286ed7ff6dd6/scratchpad/real_slice_216",
        );
        if !slice_dir.is_dir() {
            eprintln!("skipping: fixture directory not present at {}", slice_dir.display());
            return;
        }

        let client = DSperseClient::new(None);
        let synthetic_activations: Vec<f64> = (0..2048).map(|i| (i % 7) as f64 * 0.01).collect();
        let inputs = serde_json::json!({ "input_data": synthetic_activations });

        let result = client
            .prove_slice(
                "f8121b74a74514a47bc870dd913b7dbb45c8688e5a0a7328209da6a74d9c7094",
                "slice_216",
                &inputs,
                slice_dir,
            )
            .await;

        match &result {
            Ok(_) => { /* fully succeeded: witness and proof generated */ }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("activation length mismatch"),
                    "known-constants fallback did not resolve the missing WAI inputs: {msg}"
                );
                eprintln!(
                    "prove_slice did not fully succeed, but NOT due to activation length mismatch: {msg}"
                );
            }
        }
    }
}
