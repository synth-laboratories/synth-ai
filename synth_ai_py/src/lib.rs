use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

use synth_ai_core::config::CoreConfig;
use synth_ai_core::events::{poll_events as core_poll_events, EventKind};
use synth_ai_core::tunnels;
use synth_ai_core::tunnels::cloudflared::ManagedProcess;
use synth_ai_core::tunnels::types::TunnelBackend;
use synth_ai_core::tunnels::errors::TunnelError;
use synth_ai_core::urls::{
    make_local_api_url as core_make_local_api_url,
    normalize_backend_base as core_normalize_backend_base,
    normalize_inference_base as core_normalize_inference_base,
    validate_task_app_url as core_validate_task_app_url,
};
use synth_ai_core::CoreError;

static RUNTIME: Lazy<tokio::runtime::Runtime> =
    Lazy::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));

static PROCESS_REGISTRY: Lazy<Mutex<HashMap<String, ManagedProcess>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[pyclass]
#[derive(Clone)]
struct ProcessHandle {
    id: String,
    pid: u32,
}

#[pymethods]
impl ProcessHandle {
    #[getter]
    fn pid(&self) -> u32 {
        self.pid
    }

    fn terminate(&self) -> PyResult<()> {
        with_process(&self.id, |proc| {
            let _ = proc.child.start_kill();
        })
    }

    fn kill(&self) -> PyResult<()> {
        self.terminate()
    }

    fn poll(&self) -> PyResult<Option<i32>> {
        with_process(&self.id, |proc| {
            let status = proc.child.try_wait().ok().flatten();
            Ok(status.map(|s| s.code().unwrap_or_default()))
        })?
    }

    #[pyo3(signature = (timeout_s=None))]
    fn wait(&self, timeout_s: Option<f64>) -> PyResult<Option<i32>> {
        let mut registry = PROCESS_REGISTRY.lock().unwrap();
        let proc = registry
            .get_mut(&self.id)
            .ok_or_else(|| PyValueError::new_err("process not found"))?;
        let result = RUNTIME.block_on(async {
            if let Some(timeout) = timeout_s {
                let timeout = std::time::Duration::from_secs_f64(timeout);
                match tokio::time::timeout(timeout, proc.child.wait()).await {
                    Ok(Ok(status)) => Ok(Some(status.code().unwrap_or_default())),
                    _ => Ok(None),
                }
            } else {
                proc.child
                    .wait()
                    .await
                    .map(|s| Some(s.code().unwrap_or_default()))
                    .map_err(|e| TunnelError::process(e.to_string()))
            }
        });
        result.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone)]
struct TunnelHandlePy {
    url: String,
    hostname: String,
    local_port: u16,
    backend: String,
    lease_id: Option<String>,
}

#[pymethods]
impl TunnelHandlePy {
    #[getter]
    fn url(&self) -> String {
        self.url.clone()
    }

    #[getter]
    fn hostname(&self) -> String {
        self.hostname.clone()
    }

    #[getter]
    fn local_port(&self) -> u16 {
        self.local_port
    }

    #[getter]
    fn backend(&self) -> String {
        self.backend.clone()
    }

    #[getter]
    fn lease_id(&self) -> Option<String> {
        self.lease_id.clone()
    }

    fn close(&self) -> PyResult<()> {
        if let Some(lease_id) = &self.lease_id {
            let id = lease_id.clone();
            let result = RUNTIME.block_on(async move {
                let mgr = tunnels::manager::get_manager(None, None);
                let mut guard = mgr.lock();
                guard.close(&id).await
            });
            return result.map_err(|e| PyValueError::new_err(e.to_string()));
        }
        Ok(())
    }
}

fn map_core_err(err: CoreError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn normalize_backend_base(url: &str) -> PyResult<String> {
    core_normalize_backend_base(url)
        .map(|u| u.to_string())
        .map_err(map_core_err)
}

#[pyfunction]
fn normalize_inference_base(url: &str) -> PyResult<String> {
    core_normalize_inference_base(url)
        .map(|u| u.to_string())
        .map_err(map_core_err)
}

#[pyfunction]
fn make_local_api_url(host: &str, port: u16) -> PyResult<String> {
    core_make_local_api_url(host, port)
        .map(|u| u.to_string())
        .map_err(map_core_err)
}

#[pyfunction]
fn validate_task_app_url(url: &str) -> PyResult<String> {
    core_validate_task_app_url(url)
        .map(|u| u.to_string())
        .map_err(map_core_err)
}

#[pyfunction]
#[pyo3(signature = (kind, job_id, backend_base_url=None, api_key=None, since_seq=None, limit=None, timeout_ms=None))]
fn poll_events(
    py: Python,
    kind: &str,
    job_id: &str,
    backend_base_url: Option<String>,
    api_key: Option<String>,
    since_seq: Option<i64>,
    limit: Option<usize>,
    timeout_ms: Option<u64>,
) -> PyResult<PyObject> {
    let mut config = CoreConfig::default();
    if let Some(base) = backend_base_url {
        config.backend_base_url = base;
    }
    if let Some(key) = api_key {
        config.api_key = Some(key);
    }
    if let Some(ms) = timeout_ms {
        config.timeout_ms = ms;
    }

    let event_kind = EventKind::from_str(kind)
        .ok_or_else(|| PyValueError::new_err(format!("unknown event kind: {kind}")))?;

    let response = RUNTIME
        .block_on(core_poll_events(event_kind, job_id, &config, since_seq, limit))
        .map_err(map_core_err)?;

    pythonize::pythonize(py, &response).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn store_process(proc: ManagedProcess) -> ProcessHandle {
    let pid = proc.child.id().unwrap_or_default();
    let id = format!("proc-{}", uuid::Uuid::new_v4().simple());
    PROCESS_REGISTRY.lock().unwrap().insert(id.clone(), proc);
    ProcessHandle { id, pid }
}

fn with_process<T>(
    id: &str,
    f: impl FnOnce(&mut ManagedProcess) -> T,
) -> PyResult<T> {
    let mut registry = PROCESS_REGISTRY.lock().unwrap();
    let proc = registry.get_mut(id).ok_or_else(|| PyValueError::new_err("process not found"))?;
    Ok(f(proc))
}


#[pyfunction]
#[pyo3(signature = (port, wait_s=None))]
fn open_quick_tunnel(port: u16, wait_s: Option<f64>) -> PyResult<(String, ProcessHandle)> {
    let wait = wait_s.unwrap_or(10.0);
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::open_quick_tunnel(port, wait).await });
    match result {
        Ok((url, proc)) => Ok((url, store_process(proc))),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (port, wait_s=None, verify_dns=true, api_key=None))]
fn open_quick_tunnel_with_dns_verification(
    port: u16,
    wait_s: Option<f64>,
    verify_dns: bool,
    api_key: Option<String>,
) -> PyResult<(String, ProcessHandle)> {
    let wait = wait_s.unwrap_or(10.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::open_quick_tunnel_with_dns_verification(port, wait, verify_dns, api_key).await
    });
    match result {
        Ok((url, proc)) => Ok((url, store_process(proc))),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
fn open_managed_tunnel(token: String) -> PyResult<ProcessHandle> {
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::open_managed_tunnel(&token).await });
    match result {
        Ok(proc) => Ok(store_process(proc)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (token, timeout_s=None))]
fn open_managed_tunnel_with_connection_wait(token: String, timeout_s: Option<f64>) -> PyResult<ProcessHandle> {
    let timeout = timeout_s.unwrap_or(30.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::open_managed_tunnel_with_connection_wait(&token, timeout).await
    });
    match result {
        Ok(proc) => Ok(store_process(proc)),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
fn stop_tunnel(handle: ProcessHandle) -> PyResult<()> {
    let mut registry = PROCESS_REGISTRY.lock().unwrap();
    if let Some(mut proc) = registry.remove(&handle.id) {
        let _ = proc.child.start_kill();
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (api_key, port, backend_url=None))]
fn rotate_tunnel(api_key: String, port: u16, backend_url: Option<String>) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::rotate_tunnel(&api_key, port, backend_url).await
    });
    match result {
        Ok(res) => Python::with_gil(|py| {
            pythonize::pythonize(py, &res).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
        }),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (api_key, port, subdomain=None))]
fn create_tunnel(api_key: String, port: u16, subdomain: Option<String>) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::create_tunnel(&api_key, port, subdomain).await
    });
    match result {
        Ok(res) => Python::with_gil(|py| {
            pythonize::pythonize(py, &res).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
        }),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (url, timeout_s=None, api_key=None))]
fn verify_tunnel_dns_resolution(
    url: String,
    timeout_s: Option<f64>,
    api_key: Option<String>,
) -> PyResult<()> {
    let timeout = timeout_s.unwrap_or(60.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::verify_tunnel_dns_resolution(&url, "tunnel", timeout, api_key).await
    });
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (host, port, api_key=None, timeout_s=None))]
fn wait_for_health_check(host: String, port: u16, api_key: Option<String>, timeout_s: Option<f64>) -> PyResult<()> {
    let timeout = timeout_s.unwrap_or(30.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::wait_for_health_check(&host, port, api_key, timeout).await
    });
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (prefer_system=None))]
fn get_cloudflared_path(prefer_system: Option<bool>) -> PyResult<Option<String>> {
    let prefer = prefer_system.unwrap_or(true);
    Ok(tunnels::cloudflared::get_cloudflared_path(prefer).map(|p| p.to_string_lossy().to_string()))
}

#[pyfunction]
#[pyo3(signature = (force=None))]
fn ensure_cloudflared_installed(force: Option<bool>) -> PyResult<String> {
    let force = force.unwrap_or(false);
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::ensure_cloudflared_installed(force).await });
    result
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn require_cloudflared() -> PyResult<String> {
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::require_cloudflared().await });
    result
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (port, host=None))]
fn is_port_available(port: u16, host: Option<String>) -> PyResult<bool> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    Ok(tunnels::ports::is_port_available(port, &host))
}

#[pyfunction]
#[pyo3(signature = (start_port, host=None, max_attempts=None))]
fn find_available_port(start_port: u16, host: Option<String>, max_attempts: Option<u16>) -> PyResult<u16> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    let max = max_attempts.unwrap_or(100);
    tunnels::ports::find_available_port(start_port, &host, max)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn kill_port(port: u16) -> PyResult<bool> {
    tunnels::ports::kill_port(port).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (port, behavior, host=None, max_search=None))]
fn acquire_port(port: u16, behavior: String, host: Option<String>, max_search: Option<u16>) -> PyResult<u16> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    let behavior_enum = match behavior.as_str() {
        "fail" => tunnels::ports::PortConflictBehavior::Fail,
        "evict" => tunnels::ports::PortConflictBehavior::Evict,
        "find_new" => tunnels::ports::PortConflictBehavior::FindNew,
        _ => return Err(PyValueError::new_err("unknown conflict behavior")),
    };
    let max = max_search.unwrap_or(100);
    tunnels::ports::acquire_port(port, behavior_enum, &host, max)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (backend, local_port, api_key=None, backend_url=None, verify_local=None, verify_dns=true, progress=false))]
fn tunnel_open(
    backend: String,
    local_port: u16,
    api_key: Option<String>,
    backend_url: Option<String>,
    verify_local: Option<bool>,
    verify_dns: bool,
    progress: bool,
) -> PyResult<TunnelHandlePy> {
    let backend_enum = match backend.as_str() {
        "localhost" => TunnelBackend::Localhost,
        "cloudflare_quick" => TunnelBackend::CloudflareQuick,
        "cloudflare_managed" => TunnelBackend::CloudflareManaged,
        "cloudflare_managed_lease" => TunnelBackend::CloudflareManagedLease,
        _ => return Err(PyValueError::new_err("unknown backend")),
    };
    if matches!(backend_enum, TunnelBackend::CloudflareQuick | TunnelBackend::CloudflareManaged) {
        return Err(PyValueError::new_err(
            "tunnel_open does not support quick/legacy managed backends; use open_quick_tunnel/open_managed_tunnel",
        ));
    }
    let verify_local = verify_local.unwrap_or(false);
    let result = RUNTIME.block_on(async move {
        tunnels::open_tunnel(
            backend_enum,
            local_port,
            api_key,
            backend_url,
            verify_local,
            verify_dns,
            progress,
        )
        .await
    });
    match result {
        Ok(handle) => Ok(TunnelHandlePy {
            url: handle.url.clone(),
            hostname: handle.hostname.clone(),
            local_port: handle.local_port,
            backend: backend,
            lease_id: handle.lease.as_ref().map(|l| l.lease_id.clone()),
        }),
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

// =============================================================================
// Jobs Module - JobLifecycle state machine
// =============================================================================

use synth_ai_core::jobs::{JobLifecycle as RustJobLifecycle, JobStatus as RustJobStatus};

#[pyclass]
struct JobLifecycle {
    inner: RustJobLifecycle,
}

#[pymethods]
impl JobLifecycle {
    #[new]
    fn new(job_id: &str) -> Self {
        Self {
            inner: RustJobLifecycle::new(job_id),
        }
    }

    #[getter]
    fn status(&self) -> String {
        self.inner.status().as_str().to_string()
    }

    #[getter]
    fn job_id(&self) -> String {
        self.inner.job_id().to_string()
    }

    #[getter]
    fn elapsed_seconds(&self) -> Option<f64> {
        self.inner.elapsed_seconds()
    }

    #[pyo3(signature = (data=None, message=None))]
    fn start(&mut self, py: Python, data: Option<PyObject>, message: Option<&str>) -> PyResult<PyObject> {
        let data_value = data.map(|d| pythonize::depythonize(d.bind(py))).transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let event = self.inner.start_with_data(data_value, message)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &event).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (data=None, message=None))]
    fn complete(&mut self, py: Python, data: Option<PyObject>, message: Option<&str>) -> PyResult<PyObject> {
        let data_value = data.map(|d| pythonize::depythonize(d.bind(py))).transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let event = self.inner.complete_with_message(data_value, message)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &event).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (error=None, data=None))]
    fn fail(&mut self, py: Python, error: Option<&str>, data: Option<PyObject>) -> PyResult<PyObject> {
        let data_value = data.map(|d| pythonize::depythonize(d.bind(py))).transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let event = self.inner.fail_with_data(error, data_value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &event).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (message=None))]
    fn cancel(&mut self, py: Python, message: Option<&str>) -> PyResult<PyObject> {
        let event = self.inner.cancel_with_message(message)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &event).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn events(&self, py: Python) -> PyResult<PyObject> {
        pythonize::pythonize(py, self.inner.events()).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn job_status_is_terminal(status: &str) -> bool {
    RustJobStatus::from_str(status)
        .map(|s| s.is_terminal())
        .unwrap_or(false)
}

#[pyfunction]
fn job_status_from_str(status: &str) -> Option<String> {
    RustJobStatus::from_str(status).map(|s| s.as_str().to_string())
}

// =============================================================================
// Auth Module - API key resolution
// =============================================================================

use synth_ai_core::auth;

#[pyfunction]
#[pyo3(signature = (env_key=None))]
fn get_api_key(env_key: Option<&str>) -> Option<String> {
    auth::get_api_key(env_key)
}

#[pyfunction]
#[pyo3(signature = (env_key=None))]
fn get_api_key_from_env(env_key: Option<&str>) -> Option<String> {
    auth::get_api_key_from_env(env_key)
}

#[pyfunction]
#[pyo3(signature = (backend_url=None, allow_mint=true))]
fn get_or_mint_api_key(backend_url: Option<String>, allow_mint: bool) -> PyResult<String> {
    RUNTIME.block_on(async {
        auth::get_or_mint_api_key(backend_url.as_deref(), allow_mint).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (frontend_url=None))]
fn init_device_auth(py: Python, frontend_url: Option<String>) -> PyResult<PyObject> {
    let session = RUNTIME.block_on(async {
        auth::init_device_auth(frontend_url.as_deref()).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &session).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (frontend_url=None, device_code="", poll_interval_secs=None, timeout_secs=None))]
fn poll_device_token(
    py: Python,
    frontend_url: Option<String>,
    device_code: &str,
    poll_interval_secs: Option<u64>,
    timeout_secs: Option<u64>,
) -> PyResult<PyObject> {
    let creds = RUNTIME.block_on(async {
        auth::poll_device_token(
            frontend_url.as_deref(),
            device_code,
            poll_interval_secs,
            timeout_secs,
        ).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &creds).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn mask_str(s: &str) -> String {
    auth::mask_str(s)
}

// =============================================================================
// Polling Module - Backoff calculation
// =============================================================================

use synth_ai_core::polling;

#[pyfunction]
fn calculate_backoff(base_ms: u64, max_ms: u64, consecutive: u32) -> u64 {
    polling::calculate_backoff_ms(base_ms, max_ms, consecutive)
}

// =============================================================================
// Config Module - TOML parsing and config expansion
// =============================================================================

use synth_ai_core::config;

#[pyfunction]
fn parse_toml(py: Python, content: &str) -> PyResult<PyObject> {
    let value = config::parse_toml(content)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &value).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn deep_merge(py: Python, base: PyObject, overrides: PyObject) -> PyResult<PyObject> {
    let mut base_value: serde_json::Value = pythonize::depythonize(base.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let overrides_value: serde_json::Value = pythonize::depythonize(overrides.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    config::deep_merge(&mut base_value, &overrides_value);
    pythonize::pythonize(py, &base_value).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn resolve_seeds(py: Python, seeds: PyObject) -> PyResult<Vec<String>> {
    let seeds_value: serde_json::Value = pythonize::depythonize(seeds.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    config::resolve_seeds(&seeds_value).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn split_train_validation(seeds: Vec<String>, train_ratio: f64) -> (Vec<String>, Vec<String>) {
    config::split_train_validation(&seeds, train_ratio)
}

// =============================================================================
// API Client - SynthClient
// =============================================================================

use synth_ai_core::api::{
    SynthClient as RustSynthClient,
    GepaJobRequest, MiproJobRequest, EvalJobRequest,
    GraphCompletionRequest, VerifierOptions,
};
use synth_ai_core::orchestration::{
    PromptLearningJob as RustPromptLearningJob,
};

#[pyclass]
struct SynthClient {
    inner: RustSynthClient,
}

#[pymethods]
impl SynthClient {
    #[new]
    #[pyo3(signature = (api_key=None, base_url=None))]
    fn new(api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let client = if let Some(key) = api_key {
            RustSynthClient::new(key, base_url)
        } else {
            RustSynthClient::from_env()
        };
        client
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    // -------------------------------------------------------------------------
    // Jobs API
    // -------------------------------------------------------------------------

    #[pyo3(signature = (request))]
    fn submit_gepa(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: GepaJobRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.jobs().submit_gepa(req).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (request))]
    fn submit_mipro(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: MiproJobRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.jobs().submit_mipro(req).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (request))]
    fn submit_job_raw(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: serde_json::Value = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.jobs().submit_raw(req).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn get_job_status(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.jobs().get_status(job_id).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, timeout_secs=3600.0, interval_secs=15.0))]
    fn poll_job(&self, py: Python, job_id: &str, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.jobs().poll_until_complete(job_id, timeout_secs, interval_secs).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, reason=None))]
    fn cancel_job(&self, job_id: &str, reason: Option<&str>) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.inner.jobs().cancel(job_id, reason).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -------------------------------------------------------------------------
    // Eval API
    // -------------------------------------------------------------------------

    #[pyo3(signature = (request))]
    fn submit_eval(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: EvalJobRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.eval().submit(req).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn get_eval_status(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().get_status(job_id).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, timeout_secs=1800.0, interval_secs=10.0))]
    fn poll_eval(&self, py: Python, job_id: &str, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().poll_until_complete(job_id, timeout_secs, interval_secs).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn cancel_eval(&self, job_id: &str) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.inner.eval().cancel(job_id).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -------------------------------------------------------------------------
    // Graphs API
    // -------------------------------------------------------------------------

    #[pyo3(signature = (request))]
    fn graph_complete(&self, py: Python, request: PyObject) -> PyResult<PyObject> {
        let req: GraphCompletionRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        let result = RUNTIME.block_on(async {
            self.inner.graphs().complete(req).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (trace, rubric, options=None))]
    fn verify(&self, py: Python, trace: PyObject, rubric: PyObject, options: Option<PyObject>) -> PyResult<PyObject> {
        let trace_value: serde_json::Value = pythonize::depythonize(trace.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid trace: {}", e)))?;
        let rubric_value: serde_json::Value = pythonize::depythonize(rubric.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid rubric: {}", e)))?;
        let opts: Option<VerifierOptions> = options
            .map(|o| pythonize::depythonize(o.bind(py)))
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("invalid options: {}", e)))?;

        let result = RUNTIME.block_on(async {
            self.inner.graphs().verify(trace_value, rubric_value, opts).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, input, model=None))]
    fn policy_inference(&self, py: Python, job_id: &str, input: PyObject, model: Option<&str>) -> PyResult<PyObject> {
        let input_value: serde_json::Value = pythonize::depythonize(input.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid input: {}", e)))?;
        let result = RUNTIME.block_on(async {
            self.inner.graphs().policy_inference(job_id, input_value, model).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Orchestration - PromptLearningJob
// =============================================================================

#[pyclass]
struct PromptLearningJob {
    inner: std::sync::Mutex<RustPromptLearningJob>,
}

#[pymethods]
impl PromptLearningJob {
    #[staticmethod]
    #[pyo3(signature = (config, api_key=None, base_url=None))]
    fn from_dict(py: Python, config: PyObject, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid config: {}", e)))?;
        let job = RustPromptLearningJob::from_dict(config_value, api_key, base_url)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[staticmethod]
    #[pyo3(signature = (job_id, api_key=None, base_url=None))]
    fn from_job_id(job_id: &str, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let job = RustPromptLearningJob::from_job_id(job_id, api_key, base_url)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[getter]
    fn job_id(&self) -> Option<String> {
        self.inner.lock().unwrap().job_id().map(|s| s.to_string())
    }

    fn submit(&self) -> PyResult<String> {
        let mut job = self.inner.lock().unwrap();
        RUNTIME.block_on(async {
            job.submit().await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_status(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_status().await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (timeout_secs=3600.0, interval_secs=15.0))]
    fn poll_until_complete(&self, py: Python, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.poll_until_complete(timeout_secs, interval_secs).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (timeout_secs=3600.0))]
    fn stream_until_complete(&self, py: Python, timeout_secs: f64) -> PyResult<PyObject> {
        let mut job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.stream_until_complete::<fn(&synth_ai_core::orchestration::ParsedEvent)>(timeout_secs, None).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (reason=None))]
    fn cancel(&self, reason: Option<&str>) -> PyResult<()> {
        let job = self.inner.lock().unwrap();
        RUNTIME.block_on(async {
            job.cancel(reason).await
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_results(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_results().await
        }).map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn tracker_summary(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let summary = job.tracker().to_summary();
        pythonize::pythonize(py, &summary)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Module Registration
// =============================================================================

use pyo3::types::PyModule;

#[pymodule]
fn synth_ai_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // URLs
    m.add_function(wrap_pyfunction!(normalize_backend_base, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_inference_base, m)?)?;
    m.add_function(wrap_pyfunction!(make_local_api_url, m)?)?;
    m.add_function(wrap_pyfunction!(validate_task_app_url, m)?)?;

    // Events
    m.add_function(wrap_pyfunction!(poll_events, m)?)?;

    // Tunnels
    m.add_class::<ProcessHandle>()?;
    m.add_class::<TunnelHandlePy>()?;
    m.add_function(wrap_pyfunction!(open_quick_tunnel, m)?)?;
    m.add_function(wrap_pyfunction!(open_quick_tunnel_with_dns_verification, m)?)?;
    m.add_function(wrap_pyfunction!(open_managed_tunnel, m)?)?;
    m.add_function(wrap_pyfunction!(open_managed_tunnel_with_connection_wait, m)?)?;
    m.add_function(wrap_pyfunction!(stop_tunnel, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_tunnel, m)?)?;
    m.add_function(wrap_pyfunction!(create_tunnel, m)?)?;
    m.add_function(wrap_pyfunction!(verify_tunnel_dns_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(wait_for_health_check, m)?)?;
    m.add_function(wrap_pyfunction!(tunnel_open, m)?)?;
    m.add_function(wrap_pyfunction!(get_cloudflared_path, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_cloudflared_installed, m)?)?;
    m.add_function(wrap_pyfunction!(require_cloudflared, m)?)?;
    m.add_function(wrap_pyfunction!(is_port_available, m)?)?;
    m.add_function(wrap_pyfunction!(find_available_port, m)?)?;
    m.add_function(wrap_pyfunction!(kill_port, m)?)?;
    m.add_function(wrap_pyfunction!(acquire_port, m)?)?;

    // Jobs (NEW)
    m.add_class::<JobLifecycle>()?;
    m.add_function(wrap_pyfunction!(job_status_is_terminal, m)?)?;
    m.add_function(wrap_pyfunction!(job_status_from_str, m)?)?;

    // Auth (NEW)
    m.add_function(wrap_pyfunction!(get_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(get_api_key_from_env, m)?)?;
    m.add_function(wrap_pyfunction!(get_or_mint_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(init_device_auth, m)?)?;
    m.add_function(wrap_pyfunction!(poll_device_token, m)?)?;
    m.add_function(wrap_pyfunction!(mask_str, m)?)?;

    // Polling (NEW)
    m.add_function(wrap_pyfunction!(calculate_backoff, m)?)?;

    // Config (NEW)
    m.add_function(wrap_pyfunction!(parse_toml, m)?)?;
    m.add_function(wrap_pyfunction!(deep_merge, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(split_train_validation, m)?)?;

    // API Client (NEW)
    m.add_class::<SynthClient>()?;

    // Orchestration (NEW)
    m.add_class::<PromptLearningJob>()?;

    Ok(())
}
