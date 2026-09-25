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
    ///
    /// Sharded by bundle path: `load_bundle_cached` holds the backend's
    /// cache mutex while it reads and decompresses a cold bundle, so a
    /// single backend makes every concurrent request -- across all miners
    /// this process serves -- wait behind any cold load. Each bundle maps
    /// to exactly one shard, so nothing is cached twice.
    backends: Vec<Arc<dsperse::backend::jstprove::JstproveBackend>>,
    /// Caches `resolve_component`'s (component_sha, slice_id) -> slice_dir
    /// lookups, avoiding a full scan of every locally cached model
    /// directory on repeat DSlice requests for the same component.
    component_cache: RwLock<HashMap<(String, String), PathBuf>>,
    /// Caches `find_slice_onnx`'s slice_dir -> onnx_path resolution,
    /// avoiding a directory listing on every repeat prove_slice call.
    onnx_cache: RwLock<HashMap<PathBuf, PathBuf>>,
    /// Caches the fallback whole-file ONNX initializer extraction
    /// (`extract_onnx_initializers`), keyed by onnx_path. Only hit when
    /// the known-constants table is missing an entry, but when it is,
    /// the result is fully deterministic for a given circuit -- re-reading
    /// and re-parsing the same donor ONNX file on every repeat request
    /// for that slice is pure waste. `Arc`-wrapped since it's read and
    /// populated from inside `spawn_blocking`, not just `&self` methods.
    onnx_initializers_cache: Arc<RwLock<HashMap<PathBuf, Vec<Initializer>>>>,
    /// Caches the whole-model-onnx WAI initializer fallback (see
    /// `prove_slice`'s unresolved-name handling), keyed by the
    /// whole-model `model.onnx` path and shared across every slice of
    /// that model -- the file can be 100+MB and dsperse's own
    /// `extract_onnx_initializers` re-parses it from disk on every call
    /// with no internal memoization, so caching per-slice (as with
    /// `onnx_initializers_cache` below) would re-parse it once per
    /// distinct slice needing the fallback instead of once per model.
    /// Each entry is a `OnceLock`, not a plain map value: two slices of
    /// the SAME model (a real scenario, not theoretical -- slice_366
    /// and slice_409 of one model both need this fallback and can be
    /// dispatched concurrently) that miss the cache at the same moment
    /// share one parse instead of each independently re-parsing a file
    /// measured at 2+ seconds. See `resolve_whole_model_initializers`
    /// for why a failed parse is deliberately evicted rather than left
    /// permanently cached in the `OnceLock`.
    whole_model_initializers_cache: Arc<RwLock<HashMap<PathBuf, Arc<WholeModelOnceCell>>>>,
}

/// `OnceLock` value for `whole_model_initializers_cache` -- `None` means a
/// parse was attempted and failed (see `resolve_whole_model_initializers`
/// for why that state is evicted rather than left permanently cached).
/// One ONNX initializer tensor: flattened f64 values and its shape.
type Initializer = (Vec<f64>, Vec<usize>);

type WholeModelOnceCell = std::sync::OnceLock<Option<Arc<HashMap<String, Initializer>>>>;

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

/// Parses `path` once and returns every graph-level initializer as a
/// name -> (f64 values, shape) map. Used to share a single parse of a
/// (potentially 100+MB) whole-model onnx file across every slice that
/// needs it, instead of dsperse's own `extract_onnx_initializers` -- a
/// convenience wrapper with no internal memoization -- re-parsing the
/// file from disk on every call. Built from dsperse's public
/// `slicer::onnx_proto` primitives, the only ones exposed at this crate
/// rev for a full (unfiltered) name -> tensor map.
fn load_whole_model_initializer_map(path: &Path) -> Result<HashMap<String, Initializer>> {
    use dsperse::slicer::onnx_proto::TensorProto;

    let model = dsperse::slicer::onnx_proto::load_model(path)
        .map_err(|e| anyhow::anyhow!("loading whole-model onnx {}: {e}", path.display()))?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("whole-model onnx {} has no graph", path.display()))?;
    let init_map = dsperse::slicer::onnx_proto::build_initializer_map(graph);
    Ok(init_map
        .into_iter()
        .filter_map(|(name, tensor)| {
            // dsperse's tensor_to_f64 (like tensor_to_f32) reads the shared,
            // dtype-overloaded `int32_data` protobuf field without checking
            // `data_type` first. Per the ONNX TensorProto wire format, that
            // field is also used to carry FLOAT16/BFLOAT16/FLOAT8 values as
            // raw bit patterns -- an unguarded numeric cast would silently
            // reinterpret those as literal integers. That failure preserves
            // element count (passes every downstream shape check) while
            // being numerically wrong, which is a strictly worse outcome
            // than a missing value under this file's policy. Restricting to
            // the four dtypes tensor_to_f64 is confirmed (by reading its
            // source at this pinned rev) to decode correctly turns any other
            // dtype into a clean "not found" here instead.
            let dt = tensor.data_type;
            if dt != TensorProto::FLOAT
                && dt != TensorProto::DOUBLE
                && dt != TensorProto::INT64
                && dt != TensorProto::INT32
            {
                return None;
            }
            // tensor_to_f64, not tensor_to_f32 + a widening cast: the same
            // rev's own doc comment on tensor_to_f64 documents it as the
            // more-precise successor for exactly this reason (an INT64
            // value beyond f32's 24-bit exact-integer range loses precision
            // through an f32 intermediate; tensor_to_f64 round-trips it
            // exactly).
            let values = dsperse::slicer::onnx_proto::tensor_to_f64(tensor);
            let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
            Some((name, (values, shape)))
        })
        .collect())
}

/// Resolves (and caches) the full whole-model initializer map for
/// `whole_model_path`, deduplicating concurrent misses for the SAME path
/// via a per-path `OnceLock` so two slices of one model racing a cold
/// cache share one parse instead of each independently re-parsing a file
/// measured at 2+ seconds for a 100MB+ model -- a real scenario, not
/// theoretical (slice_366 and slice_409 of the same production model both
/// need this fallback and can be dispatched concurrently).
///
/// A failed parse is evicted rather than left permanently cached:
/// `OnceLock` can't be reset once initialized, and permanently caching
/// failure would make a transient cause (a momentary fd-exhaustion or
/// permission error, say) indistinguishable from a genuinely corrupted
/// file for the rest of this process's lifetime -- silently disabling the
/// fallback for every future slice of that model. Only removes its OWN
/// cell (compared by pointer identity) before evicting, so a slow caller
/// from a failed generation can't clobber a newer, independently-started
/// generation's entry.
fn resolve_whole_model_initializers(
    cache: &RwLock<HashMap<PathBuf, Arc<WholeModelOnceCell>>>,
    whole_model_path: &Path,
) -> Option<Arc<HashMap<String, Initializer>>> {
    // Bound to its own `let` rather than used directly as a `match`
    // scrutinee: a match scrutinee's temporaries (including this read
    // guard) live for the whole match, not just the scrutinee -- calling
    // `cache.write()` from the `None` arm while that guard is still held
    // would self-deadlock on every cache miss (std::sync::RwLock isn't
    // reentrant). Confirmed by a minimal repro before this fix landed.
    let existing = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(whole_model_path)
        .cloned();
    let cell = match existing {
        Some(existing) => existing,
        None => Arc::clone(
            cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .entry(whole_model_path.to_path_buf())
                .or_insert_with(|| Arc::new(std::sync::OnceLock::new())),
        ),
    };

    let result = cell
        .get_or_init(
            || match load_whole_model_initializer_map(whole_model_path) {
                Ok(map) => Some(Arc::new(map)),
                Err(e) => {
                    tracing::warn!(
                        path = %whole_model_path.display(),
                        error = %e,
                        "failed to parse whole-model onnx for WAI fallback"
                    );
                    None
                }
            },
        )
        .clone();

    if result.is_none() {
        let mut guard = cache.write().unwrap_or_else(|e| e.into_inner());
        if guard
            .get(whole_model_path)
            .is_some_and(|current| Arc::ptr_eq(current, &cell))
        {
            guard.remove(whole_model_path);
        }
    }

    result
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

/// Number of independent prover backends (and bundle-cache locks).
const BACKEND_SHARDS: usize = 16;

impl DSperseClient {
    /// The backend that owns `circuit_path`'s cached bundle.
    fn backend_for(&self, circuit_path: &Path) -> Arc<dsperse::backend::jstprove::JstproveBackend> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        circuit_path.hash(&mut hasher);
        Arc::clone(&self.backends[hasher.finish() as usize % self.backends.len()])
    }

    pub fn new(cache_dir_override: Option<&str>) -> Self {
        let cache_dir = PathBuf::from(
            shellexpand::tilde(cache_dir_override.unwrap_or(sn2_types::CIRCUIT_CACHE_DIR))
                .to_string(),
        );
        info!(cache_dir = %cache_dir.display(), "initialized DSperseClient");

        let backends: Vec<_> = (0..BACKEND_SHARDS)
            .map(|_| Arc::new(dsperse::backend::jstprove::JstproveBackend::new()))
            .collect();
        {
            let backends = backends.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(
                    sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS,
                ));
                loop {
                    interval.tick().await;
                    let ttl = Duration::from_secs(sn2_types::BUNDLE_CACHE_IDLE_TTL_SECS);
                    let evicted: usize = backends.iter().map(|b| b.evict_idle(ttl)).sum();
                    if evicted > 0 {
                        info!(evicted, "evicted idle compiled circuit bundles");
                    }
                }
            });
        }

        Self {
            cache_dir,
            backends,
            component_cache: RwLock::new(HashMap::new()),
            onnx_cache: RwLock::new(HashMap::new()),
            onnx_initializers_cache: Arc::new(RwLock::new(HashMap::new())),
            whole_model_initializers_cache: Arc::new(RwLock::new(HashMap::new())),
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

        // Read is bound to an owned value BEFORE the `if let`, not used as
        // the scrutinee directly: a match/if-let scrutinee's temporaries
        // (including an unbound RwLock::read() guard) live for the WHOLE
        // match/if-let, so a `.write()` call from inside the arm below would
        // otherwise self-deadlock against this still-alive read guard
        // (std::sync::RwLock is not reentrant) -- same footgun already fixed
        // once in resolve_whole_model_initializers.
        let cached_entry = self
            .component_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&cache_key)
            .cloned();
        if let Some(cached) = cached_entry {
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
        // Returns (slice_dir, onnx_path) together: find_slice_onnx is already
        // called below to VALIDATE a candidate match, so its result is
        // available for free here -- priming onnx_cache with it saves
        // prove_slice's later resolve_slice_onnx call from re-listing the
        // exact same payload/ directory it was just computed from.
        let span = tracing::Span::current();
        let resolved: Option<(PathBuf, PathBuf)> =
            tokio::task::spawn_blocking(move || -> Result<Option<(PathBuf, PathBuf)>> {
                let _span = span.enter();
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
                            let onnx_path = match find_slice_onnx(&slice_dir) {
                                Ok(p) => p,
                                Err(_) => continue,
                            };
                            return Ok(Some((slice_dir, onnx_path)));
                        }
                    }
                }
                Ok(None)
            })
            .await
            .context("component resolution task panicked")??;

        if let Some((ref slice_dir, ref onnx_path)) = resolved {
            self.component_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(cache_key, slice_dir.clone());
            self.onnx_cache
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(slice_dir.clone(), onnx_path.clone());
        }

        Ok(resolved.map(|(slice_dir, _)| slice_dir))
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
        let backend = self.backend_for(&circuit_path);
        let span = tracing::Span::current();

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let _span = span.enter();
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
        // Sibling of every slice dir under the model's cache root
        // (`<cache_dir>/model_<id>/slices/model.onnx`), already
        // downloaded by sn2-circuit-store's download_model_artifacts
        // as part of normal caching -- see the WAI unresolved-name
        // fallback below for why this file (not the per-slice donor
        // onnx) is the correct source for compile-time-folded
        // constants.
        let whole_model_onnx_path = slice_dir.parent().map(|p| p.join("model.onnx"));

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
        let backend = self.backend_for(&circuit_path);
        let onnx_initializers_cache = Arc::clone(&self.onnx_initializers_cache);
        let whole_model_initializers_cache = Arc::clone(&self.whole_model_initializers_cache);
        let span = tracing::Span::current();

        tokio::task::spawn_blocking(move || -> Result<ProveArtifacts> {
            let _span = span.enter();
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

                                // Try the EXISTING donor.onnx extraction first, exactly as
                                // it worked before the whole-model-onnx fallback below was
                                // added -- this is the proven path for genuine named model
                                // parameters (frozen backbone biases, layer-scale constants,
                                // etc.) that already resolves correctly in production for
                                // many slices today. Its result is NOT committed to `inits`
                                // yet: donor.onnx and the sibling model.onnx (see below) are
                                // confirmed, empirically, to sometimes hold DIFFERENT values
                                // for the very same name (different fine-tuning checkpoints
                                // -- e.g. one real shared parameter had trailing zeros in
                                // donor.onnx that weren't zero in model.onnx). Trying
                                // model.onnx FIRST, as an earlier version of this fix did,
                                // risked silently preferring model.onnx's value over
                                // donor.onnx's for any slice where both happen to resolve
                                // the same names -- exactly the silently-wrong-value class
                                // of bug this file exists to avoid, just self-inflicted
                                // instead of inherited. Checking sufficiency before
                                // committing either way keeps every currently-working slice
                                // byte-for-byte unchanged.
                                let cached_file_inits = onnx_initializers_cache
                                    .read()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .get(&onnx_path)
                                    .cloned();
                                let donor_file_inits = match cached_file_inits {
                                    Some(cached) => cached,
                                    None => {
                                        let extracted = dsperse::pipeline::extract_onnx_initializers(
                                            &onnx_path, p,
                                        )
                                        .map_err(|e| {
                                            anyhow::anyhow!("extracting initializers: {e}")
                                        })?;
                                        onnx_initializers_cache
                                            .write()
                                            .unwrap_or_else(|e| e.into_inner())
                                            .insert(onnx_path.clone(), extracted.clone());
                                        extracted
                                    }
                                };

                                // Mirrors jstprove's own positional split formula
                                // (onnx.rs's witness_from_f64_generic) to check sufficiency
                                // proactively instead of committing donor.onnx's result and
                                // finding out downstream.
                                let donor_would_satisfy = {
                                    let hypothetical_len = inits.len() + donor_file_inits.len();
                                    let num_activation_entries =
                                        p.inputs.len().saturating_sub(hypothetical_len);
                                    let expected: usize = p.inputs
                                        [..num_activation_entries.min(p.inputs.len())]
                                        .iter()
                                        .map(|io| io.shape.iter().product::<usize>())
                                        .sum();
                                    activations.len() == expected
                                };

                                if donor_would_satisfy {
                                    inits.extend(donor_file_inits);
                                } else {
                                    // donor.onnx alone isn't enough. Some of `unresolved`
                                    // are values dsperse's own whole-model constant-folding
                                    // pass (fold_constant_nodes / propagate_constants)
                                    // resolved at compile time -- they never existed as
                                    // named parameters in the original model, so
                                    // extract_onnx_initializers can never find them in
                                    // donor.onnx -- confirmed (2026-08-08, slice_366/
                                    // slice_409 of model
                                    // c9956a0e467381884e77328b3d25b0ec60c48d8021e44edb41ff2d1486724b83)
                                    // by downloading the real bundle from
                                    // repository.inferencelabs.com: donor.onnx is a
                                    // weights-only export with zero graph nodes (nothing to
                                    // fold), while model.onnx -- the full pre-slicing model,
                                    // already downloaded as a sibling of every slice dir by
                                    // sn2-circuit-store's download_model_artifacts -- has
                                    // the real Add nodes and, per that bundle's own
                                    // metadata.json original_model_path field, is the exact
                                    // file the compiler folded these constants from.
                                    //
                                    // Try model.onnx for the FULL original unresolved set,
                                    // replacing (not merging with) donor's insufficient
                                    // contribution: without per-name identity from either
                                    // extraction call, merging partial results from two
                                    // different files risks duplicate or misaligned entries.
                                    // Only reached once donor.onnx is already proven
                                    // insufficient, so this can never override a value
                                    // donor.onnx was correctly providing. Only trust a FULL
                                    // match across every still-unresolved name -- a partial
                                    // hit can't safely be reordered against `inits`'s
                                    // existing entries without risking a positional
                                    // mismatch, and per-name shape verification (not just
                                    // presence) rules out a same-named-but-differently-shaped
                                    // false match.
                                    let whole_model_map = whole_model_onnx_path
                                        .as_ref()
                                        .filter(|p| p.exists())
                                        .and_then(|whole_model_path| {
                                            resolve_whole_model_initializers(
                                                &whole_model_initializers_cache,
                                                whole_model_path,
                                            )
                                        });

                                    let all_resolved: Option<Vec<(Vec<f64>, Vec<usize>)>> =
                                        whole_model_map.as_ref().and_then(|whole_model_map| {
                                            unresolved
                                                .iter()
                                                .map(|name| {
                                                    let io =
                                                        p.inputs.iter().find(|io| &io.name == name)?;
                                                    let expected: usize =
                                                        io.shape.iter().product();
                                                    whole_model_map.get(name).and_then(
                                                        |(values, shape)| {
                                                            let actual: usize =
                                                                shape.iter().product();
                                                            (actual == expected
                                                                && values.len() == expected)
                                                                .then(|| {
                                                                    (values.clone(), shape.clone())
                                                                })
                                                        },
                                                    )
                                                })
                                                .collect()
                                        });

                                    match all_resolved {
                                        Some(resolved_inits) => {
                                            tracing::info!(
                                                names = ?unresolved,
                                                path = ?whole_model_onnx_path,
                                                "resolved WAI initializers from whole-model onnx"
                                            );
                                            inits.extend(resolved_inits);
                                            unresolved.clear();
                                        }
                                        None => {
                                            // Whole-model didn't fully resolve it either --
                                            // fall back to donor's (still insufficient)
                                            // contribution so the eventual failure matches
                                            // today's behavior/error text rather than
                                            // reporting zero resolution.
                                            inits.extend(donor_file_inits);
                                        }
                                    }
                                }
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
