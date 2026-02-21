use std::collections::HashMap;
use std::io::Write;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const BENCHMARK_RUN_TIMEOUT_SECS: u64 = 1800;

pub struct DSperseManager {
    socket_path: Option<String>,
}

pub struct BenchmarkRunHandle {
    pub circuit_id: String,
    pub circuit_name: String,
    pub handle: tokio::task::JoinHandle<Result<serde_json::Value>>,
}

impl DSperseManager {
    pub fn new(socket_path: Option<String>) -> Self {
        Self { socket_path }
    }

    pub async fn start_incremental_run(
        &self,
        circuit_id: &str,
        inputs: &serde_json::Value,
        run_source: &str,
        max_tiles: Option<u32>,
    ) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "start_incremental_run",
            "circuit_id": circuit_id,
            "inputs": inputs,
            "run_source": run_source,
            "max_tiles": max_tiles,
        });

        self.send_ipc(&request).await
    }

    pub fn spawn_benchmark_run(
        &self,
        circuit_id: &str,
        circuit_name: &str,
        input_schema: &HashMap<String, serde_json::Value>,
        max_tiles: Option<u32>,
    ) -> Result<BenchmarkRunHandle> {
        let payload = build_benchmark_payload(circuit_id, input_schema, max_tiles)?;
        let socket_path = self
            .socket_path
            .clone()
            .unwrap_or_else(|| "/tmp/dsperse.sock".to_string());
        let handle = tokio::spawn(async move {
            send_and_receive_standalone(&socket_path, &payload, BENCHMARK_RUN_TIMEOUT_SECS).await
        });
        Ok(BenchmarkRunHandle {
            circuit_id: circuit_id.to_string(),
            circuit_name: circuit_name.to_string(),
            handle,
        })
    }

    pub async fn get_run_status(&self, run_uid: &str) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "get_run_status",
            "run_uid": run_uid,
        });

        self.send_ipc(&request).await
    }

    pub async fn get_next_work(&self, run_uid: &str) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "get_next_work",
            "run_uid": run_uid,
        });

        self.send_ipc(&request).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn apply_slice_result(
        &self,
        run_uid: &str,
        slice_num: &str,
        success: bool,
        computed_outputs: Option<&serde_json::Value>,
        proof: Option<&str>,
        proof_system: Option<&str>,
        response_time_sec: f64,
        verification_time_sec: f64,
    ) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "apply_slice_result",
            "run_uid": run_uid,
            "slice_num": slice_num,
            "success": success,
            "computed_outputs": computed_outputs,
            "proof": proof,
            "proof_system": proof_system,
            "response_time_sec": response_time_sec,
            "verification_time_sec": verification_time_sec,
        });

        self.send_ipc(&request).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn apply_tile_result(
        &self,
        run_uid: &str,
        task_id: &str,
        slice_id: &str,
        tile_idx: u32,
        success: bool,
        computed_outputs: Option<&serde_json::Value>,
        proof: Option<&str>,
        witness: Option<&str>,
        proof_system: Option<&str>,
        response_time_sec: f64,
        verification_time_sec: f64,
    ) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "apply_tile_result",
            "run_uid": run_uid,
            "task_id": task_id,
            "slice_id": slice_id,
            "tile_idx": tile_idx,
            "success": success,
            "computed_outputs": computed_outputs,
            "proof": proof,
            "witness": witness,
            "proof_system": proof_system,
            "response_time_sec": response_time_sec,
            "verification_time_sec": verification_time_sec,
        });

        self.send_ipc(&request).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn verify_incremental_slice_with_witness(
        &self,
        circuit_id: &str,
        slice_num: &str,
        original_inputs: &serde_json::Value,
        witness_hex: &str,
        proof_hex: &str,
        proof_system: Option<&str>,
        run_uid: Option<&str>,
    ) -> Result<(bool, Option<serde_json::Value>)> {
        let request = serde_json::json!({
            "method": "verify_incremental_slice_with_witness",
            "circuit_id": circuit_id,
            "slice_num": slice_num,
            "original_inputs": original_inputs,
            "witness_hex": witness_hex,
            "proof_hex": proof_hex,
            "proof_system": proof_system,
            "run_uid": run_uid,
        });

        let response = self.send_ipc(&request).await?;
        if let Some(err) = response.get("error") {
            anyhow::bail!("dsperse verify_incremental error: {err}");
        }
        let success = response
            .get("success")
            .and_then(|v| v.as_bool())
            .context("dsperse verify_incremental: missing 'success' field")?;
        let outputs = response.get("computed_outputs").cloned();
        Ok((success, outputs))
    }

    pub async fn generate_requests(&self) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "generate_requests",
        });

        self.send_ipc(&request).await
    }

    pub async fn get_work_data(&self, task_id: &str) -> Result<serde_json::Value> {
        let request = serde_json::json!({
            "method": "get_work_data",
            "task_id": task_id,
        });

        self.send_ipc(&request).await
    }

    async fn send_ipc(&self, request: &serde_json::Value) -> Result<serde_json::Value> {
        let payload = serde_json::to_vec(request)?;
        self.send_and_receive(&payload, DEFAULT_TIMEOUT_SECS).await
    }

    async fn send_and_receive(
        &self,
        payload: &[u8],
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            self.send_and_receive_inner(payload),
        )
        .await
        .with_context(|| format!("dsperse IPC timed out after {timeout_secs}s"))?
    }

    async fn send_and_receive_inner(&self, payload: &[u8]) -> Result<serde_json::Value> {
        let socket_path = self.socket_path.as_deref().unwrap_or("/tmp/dsperse.sock");

        let mut stream = UnixStream::connect(socket_path)
            .await
            .with_context(|| format!("connecting to dsperse at {socket_path}"))?;

        let len = u32::try_from(payload.len())
            .context("IPC payload exceeds u32::MAX")?
            .to_be_bytes();
        stream.write_all(&len).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;
        anyhow::ensure!(
            resp_len <= 512 * 1024 * 1024,
            "IPC response length {resp_len} exceeds 512MB cap"
        );

        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await?;

        let response: serde_json::Value = serde_json::from_slice(&resp_buf)?;
        Ok(response)
    }
}

async fn send_and_receive_standalone(
    socket_path: &str,
    payload: &[u8],
    timeout_secs: u64,
) -> Result<serde_json::Value> {
    tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), async {
        let mut stream = UnixStream::connect(socket_path)
            .await
            .with_context(|| format!("connecting to dsperse at {socket_path}"))?;

        let len = u32::try_from(payload.len())
            .context("IPC payload exceeds u32::MAX")?
            .to_be_bytes();
        stream.write_all(&len).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;
        anyhow::ensure!(
            resp_len <= 512 * 1024 * 1024,
            "IPC response length {resp_len} exceeds 512MB cap"
        );

        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await?;

        let response: serde_json::Value = serde_json::from_slice(&resp_buf)?;
        Ok(response)
    })
    .await
    .with_context(|| format!("dsperse IPC timed out after {timeout_secs}s"))?
}

fn build_benchmark_payload(
    circuit_id: &str,
    input_schema: &HashMap<String, serde_json::Value>,
    max_tiles: Option<u32>,
) -> Result<Vec<u8>> {
    let total_elements: usize = input_schema
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|dims| {
            dims.iter()
                .filter_map(|d| d.as_u64())
                .map(|d| d as usize)
                .product()
        })
        .unwrap_or(0);

    anyhow::ensure!(
        total_elements > 0,
        "cannot derive tensor size from input_schema"
    );

    let zeros_json_len = total_elements * 4; // "0.0," per element
    let mut buf = Vec::with_capacity(256 + zeros_json_len);

    write!(
        buf,
        r#"{{"method":"start_incremental_run","circuit_id":"{}","inputs":{{"input_data":[["#,
        circuit_id,
    )?;

    for i in 0..total_elements {
        if i > 0 {
            buf.push(b',');
        }
        buf.extend_from_slice(b"0.0");
    }

    write!(buf, r#"]]}},"run_source":"benchmark","max_tiles":"#)?;
    match max_tiles {
        Some(mt) => write!(buf, "{mt}")?,
        None => write!(buf, "null")?,
    }
    buf.push(b'}');

    Ok(buf)
}
