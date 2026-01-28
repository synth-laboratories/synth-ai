use once_cell::sync::Lazy;
use pyo3::create_exception;
use pyo3::exceptions::{PyAttributeError, PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyTuple};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use synth_ai_core::config::CoreConfig;
use synth_ai_core::events::{poll_events as core_poll_events, EventKind};
use synth_ai_core::http::{HttpClient as CoreHttpClient, MultipartFile};
use synth_ai_core::localapi::TaskAppClient as CoreTaskAppClient;
use synth_ai_core::orchestration::{CandidateInfo, GEPAProgress, ProgressTracker};
use synth_ai_core::graph_evolve::GraphEvolveJob as CoreGraphEvolveJob;
use synth_ai_core::sse::stream_sse_events as core_stream_sse_events;
use synth_ai_core::streaming::{JobStreamer as CoreJobStreamer, StreamEndpoints, StreamHandler, StreamMessage};
use synth_ai_core::tunnels;
use synth_ai_core::tunnels::cloudflared::ManagedProcess;
use synth_ai_core::tunnels::types::TunnelBackend;
use synth_ai_core::tunnels::errors::TunnelError;
use synth_ai_core::tracing::{
    LibsqlTraceStorage, SessionTracer as CoreSessionTracer, TracingEvent,
    SessionTrace, SessionTimeStep, TimeRecord, MessageContent, MarkovBlanketMessage,
    LLMUsage, LLMRequestParams, LLMContentPart, LLMMessage, ToolCallSpec, ToolCallResult,
    LLMChunk, LLMCallRecord,
};
use synth_ai_core::data::{
    ApplicationErrorType, ApplicationStatus, Artifact, ContextOverride, ContextOverrideStatus,
    Criterion, CriterionExample, CriterionScoreData, EventObjectiveAssignment, Judgement,
    InstanceObjectiveAssignment, ObjectiveSpec, OutcomeObjectiveAssignment, RewardObservation,
    Rubric, RubricAssignment,
    CalibrationExample, EventRewardRecord, GoldExample, OutcomeRewardRecord, RewardAggregates,
};
use synth_ai_core::urls::{
    make_local_api_url as core_make_local_api_url,
    normalize_backend_base as core_normalize_backend_base,
    normalize_inference_base as core_normalize_inference_base,
    validate_task_app_url as core_validate_task_app_url,
};
use synth_ai_core::CoreError as CoreErrorNative;

static RUNTIME: Lazy<tokio::runtime::Runtime> =
    Lazy::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
    });

static PROCESS_REGISTRY: Lazy<Mutex<HashMap<String, ManagedProcess>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// =============================================================================
// Core error types exposed to Python
// =============================================================================

create_exception!(synth_ai_py, CoreError, PyException);
create_exception!(synth_ai_py, InvalidInputError, CoreError);
create_exception!(synth_ai_py, UrlParseError, CoreError);
create_exception!(synth_ai_py, HttpRequestError, CoreError);
create_exception!(synth_ai_py, HttpResponseError, CoreError);
create_exception!(synth_ai_py, AuthenticationError, CoreError);
create_exception!(synth_ai_py, ValidationError, CoreError);
create_exception!(synth_ai_py, UsageLimitError, CoreError);
create_exception!(synth_ai_py, JobError, CoreError);
create_exception!(synth_ai_py, ConfigError, CoreError);
create_exception!(synth_ai_py, TimeoutError, CoreError);
create_exception!(synth_ai_py, ProtocolError, CoreError);
create_exception!(synth_ai_py, InternalError, CoreError);

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
        let mut registry = PROCESS_REGISTRY
            .lock()
            .map_err(|_| PyValueError::new_err("process registry lock poisoned"))?;
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

fn map_core_err(err: CoreErrorNative) -> PyErr {
    match err {
        CoreErrorNative::InvalidInput(msg) => InvalidInputError::new_err(msg),
        CoreErrorNative::UrlParse(e) => UrlParseError::new_err(e.to_string()),
        CoreErrorNative::Http(e) => HttpRequestError::new_err(e.to_string()),
        CoreErrorNative::HttpResponse(info) => HttpResponseError::new_err(info.to_string()),
        CoreErrorNative::Authentication(msg) => AuthenticationError::new_err(msg),
        CoreErrorNative::Validation(msg) => ValidationError::new_err(msg),
        CoreErrorNative::UsageLimit(info) => UsageLimitError::new_err(info.to_string()),
        CoreErrorNative::Job(info) => JobError::new_err(info.to_string()),
        CoreErrorNative::Config(msg) => ConfigError::new_err(msg),
        CoreErrorNative::Timeout(msg) => TimeoutError::new_err(msg),
        CoreErrorNative::Protocol(msg) => ProtocolError::new_err(msg),
        CoreErrorNative::Internal(msg) => InternalError::new_err(msg),
    }
}

fn map_http_err(err: synth_ai_core::http::HttpError) -> PyErr {
    HttpRequestError::new_err(err.to_string())
}

fn map_tunnel_err(_py: Python, err: TunnelError) -> PyErr {
    ConfigError::new_err(err.to_string())
}

fn parse_param_pairs(py: Python, params: Option<PyObject>) -> PyResult<Option<Vec<(String, String)>>> {
    let Some(obj) = params else {
        return Ok(None);
    };
    let value: serde_json::Value = pythonize::depythonize(obj.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match value {
        serde_json::Value::Object(map) => {
            let mut out = Vec::with_capacity(map.len());
            for (key, val) in map {
                let v = if let serde_json::Value::String(s) = val {
                    s
                } else {
                    val.to_string()
                };
                out.push((key, v));
            }
            Ok(Some(out))
        }
        serde_json::Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                if let serde_json::Value::Array(pair) = item {
                    if pair.len() < 2 {
                        continue;
                    }
                    let key = pair[0].as_str().unwrap_or_default().to_string();
                    let val = if let serde_json::Value::String(s) = &pair[1] {
                        s.clone()
                    } else {
                        pair[1].to_string()
                    };
                    if !key.is_empty() {
                        out.push((key, val));
                    }
                }
            }
            Ok(Some(out))
        }
        _ => Ok(None),
    }
}

fn parse_multipart_files(obj: &Bound<'_, PyAny>) -> PyResult<Vec<MultipartFile>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let field: String = key.extract()?;
        let tuple = value.downcast::<PyTuple>()?;
        if tuple.len() < 2 {
            return Err(PyValueError::new_err("file tuple must include filename and bytes"));
        }
        let filename: String = tuple.get_item(0)?.extract()?;
        let content = if let Ok(bytes) = tuple.get_item(1)?.downcast::<PyBytes>() {
            bytes.as_bytes().to_vec()
        } else {
            let fallback: Vec<u8> = tuple.get_item(1)?.extract()?;
            fallback
        };
        let content_type = if tuple.len() > 2 {
            tuple.get_item(2)?.extract::<Option<String>>()?
        } else {
            None
        };
        out.push(MultipartFile {
            field,
            filename,
            content,
            content_type,
        });
    }
    Ok(out)
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

// =============================================================================
// HTTP Client
// =============================================================================

#[pyclass(name = "HttpClient")]
struct HttpClientPy {
    inner: CoreHttpClient,
}

#[pymethods]
impl HttpClientPy {
    #[new]
    #[pyo3(signature = (base_url, api_key, timeout_secs=30))]
    fn new(base_url: &str, api_key: &str, timeout_secs: u64) -> PyResult<Self> {
        CoreHttpClient::new(base_url, api_key, timeout_secs)
            .map(|inner| Self { inner })
            .map_err(map_http_err)
    }

    #[pyo3(signature = (path, params=None))]
    fn get_json(&self, py: Python, path: &str, params: Option<PyObject>) -> PyResult<PyObject> {
        let pairs = parse_param_pairs(py, params)?;
        let refs = pairs
            .as_ref()
            .map(|p| p.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>());
        let inner = self.inner.clone();
        let path = path.to_string();
        let result = py.allow_threads(|| {
            RUNTIME.block_on(async {
                inner.get_json(&path, refs.as_deref()).await
            })
        })
        .map_err(map_http_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path, params=None))]
    fn get_bytes(&self, py: Python, path: &str, params: Option<PyObject>) -> PyResult<PyObject> {
        let pairs = parse_param_pairs(py, params)?;
        let refs = pairs
            .as_ref()
            .map(|p| p.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect::<Vec<_>>());
        let inner = self.inner.clone();
        let path = path.to_string();
        let result = py.allow_threads(|| {
            RUNTIME.block_on(async { inner.get_bytes(&path, refs.as_deref()).await })
        })
        .map_err(map_http_err)?;
        Ok(PyBytes::new_bound(py, &result).into())
    }

    #[pyo3(signature = (path, body))]
    fn post_json(&self, py: Python, path: &str, body: PyObject) -> PyResult<PyObject> {
        let payload: serde_json::Value = pythonize::depythonize(body.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = self.inner.clone();
        let path = path.to_string();
        let result: serde_json::Value = py.allow_threads(|| {
            RUNTIME.block_on(async { inner.post_json(&path, &payload).await })
        })
        .map_err(map_http_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path))]
    fn delete(&self, py: Python, path: &str) -> PyResult<()> {
        let inner = self.inner.clone();
        let path = path.to_string();
        py.allow_threads(|| {
            RUNTIME.block_on(async { inner.delete(&path).await })
        })
        .map_err(map_http_err)
    }

    #[pyo3(signature = (path, data, files))]
    fn post_multipart(
        &self,
        py: Python,
        path: &str,
        data: PyObject,
        files: PyObject,
    ) -> PyResult<PyObject> {
        let params = parse_param_pairs(py, Some(data))?.unwrap_or_default();
        let files = parse_multipart_files(files.bind(py))?;
        let inner = self.inner.clone();
        let path = path.to_string();
        let result: serde_json::Value = py.allow_threads(|| {
            RUNTIME.block_on(async { inner.post_multipart(&path, &params, &files).await })
        })
        .map_err(map_http_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// SSE Streaming
// =============================================================================

#[pyfunction]
#[pyo3(signature = (url, headers=None, method="GET", json_payload=None, timeout_s=None, on_event=None))]
fn stream_sse(
    py: Python,
    url: String,
    headers: Option<HashMap<String, String>>,
    method: &str,
    json_payload: Option<PyObject>,
    timeout_s: Option<f64>,
    on_event: Option<PyObject>,
) -> PyResult<PyObject> {
    eprintln!("[RUST-SSE] stream_sse called: url={} timeout={:?}", url, timeout_s);
    use std::io::Write;
    let _ = std::io::stderr().flush();

    let body: Option<serde_json::Value> = match json_payload {
        Some(obj) => Some(
            pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    let header_pairs = headers
        .unwrap_or_default()
        .into_iter()
        .collect::<Vec<_>>();
    let timeout = timeout_s.map(std::time::Duration::from_secs_f64);

    let mut stream = RUNTIME
        .block_on(core_stream_sse_events(&url, method, header_pairs, body, timeout))
        .map_err(map_core_err)?;

    let mut collected: Vec<serde_json::Value> = Vec::new();
    let mut stop_stream = false;

    RUNTIME
        .block_on(async {
            loop {
                // NOTE: No timeout on stream.next() - SSE streams can have long gaps between events.
                // The overall timeout is handled by the reqwest client timeout if provided.
                let item = match stream.next().await {
                    Some(item) => {
                        // Log that we received an event (only periodically to avoid spam)
                        if collected.len() % 20 == 0 {
                            eprintln!("[RUST-SSE] Received event #{}", collected.len());
                        }
                        item
                    },
                    None => {
                        eprintln!("[RUST-SSE] Stream ended normally after {} events", collected.len());
                        break;
                    },
                };

                let evt = item.map_err(|e| PyValueError::new_err(e.to_string()))?;
                if let serde_json::Value::String(done) = &evt {
                    if done == "[DONE]" {
                        break;
                    }
                }
                if let Some(ref cb) = on_event {
                    Python::with_gil(|py| -> PyResult<()> {
                        let arg = pythonize::pythonize(py, &evt)
                            .map(|b| b.unbind())
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        let res = cb.call1(py, (arg,))?;
                        if res.is_none(py) {
                            return Ok(());
                        }
                        if let Ok(stop) = res.extract::<bool>(py) {
                            if !stop {
                                stop_stream = true;
                            }
                        }
                        Ok(())
                    })?;
                }
                collected.push(evt);
                if stop_stream {
                    break;
                }
            }
            Ok::<(), PyErr>(())
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    if on_event.is_some() {
        return Ok(py.None());
    }
    pythonize::pythonize(py, &collected)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Local API Client
// =============================================================================

#[pyclass(name = "TaskAppClient")]
struct TaskAppClientPy {
    inner: CoreTaskAppClient,
}

#[pymethods]
impl TaskAppClientPy {
    #[new]
    #[pyo3(signature = (base_url, api_key=None, timeout_secs=None))]
    fn new(base_url: &str, api_key: Option<&str>, timeout_secs: Option<u64>) -> PyResult<Self> {
        let client = if let Some(timeout) = timeout_secs {
            CoreTaskAppClient::with_timeout(base_url, api_key, timeout)
        } else {
            CoreTaskAppClient::new(base_url, api_key)
        };
        Ok(Self { inner: client })
    }

    fn health(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async { self.inner.health().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn is_healthy(&self) -> PyResult<bool> {
        let result = RUNTIME.block_on(async { self.inner.is_healthy().await });
        Ok(result)
    }

    fn info(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async { self.inner.info().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (seeds=None))]
    fn task_info(&self, py: Python, seeds: Option<Vec<i64>>) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(async { self.inner.task_info(seeds.as_deref()).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rollout(&self, py: Python, request: PyObject) -> PyResult<PyObject> {
        let req: synth_ai_core::localapi::RolloutRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(async { self.inner.rollout(&req).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn taskset_info(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(async { self.inner.taskset_info().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn done(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(async { self.inner.done().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path))]
    fn get(&self, py: Python, path: &str) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(async { self.inner.get(path).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path, body))]
    fn post(&self, py: Python, path: &str, body: PyObject) -> PyResult<PyObject> {
        let payload: serde_json::Value = pythonize::depythonize(body.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(async { self.inner.post(path, &payload).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (timeout_secs=60.0, poll_interval_secs=2.0))]
    fn wait_for_healthy(&self, timeout_secs: f64, poll_interval_secs: f64) -> PyResult<()> {
        RUNTIME
            .block_on(async {
                self.inner
                    .wait_for_healthy(timeout_secs, poll_interval_secs)
                    .await
            })
            .map_err(map_core_err)
    }

    #[pyo3(signature = (env_name, payload))]
    fn env_initialize(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(async { self.inner.env().initialize(env_name, &payload).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (env_name, payload))]
    fn env_step(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(async { self.inner.env().step(env_name, &payload).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (env_name, payload=None))]
    fn env_terminate(
        &self,
        py: Python,
        env_name: &str,
        payload: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let payload: serde_json::Value = match payload {
            Some(obj) => pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => serde_json::json!({}),
        };
        let result = RUNTIME
            .block_on(async { self.inner.env().terminate(env_name, &payload).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (env_name, payload=None))]
    fn env_reset(
        &self,
        py: Python,
        env_name: &str,
        payload: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let payload: serde_json::Value = match payload {
            Some(obj) => pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => serde_json::json!({}),
        };
        let result = RUNTIME
            .block_on(async { self.inner.env().reset(env_name, &payload).await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
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
fn open_quick_tunnel(py: Python, port: u16, wait_s: Option<f64>) -> PyResult<(String, ProcessHandle)> {
    let wait = wait_s.unwrap_or(10.0);
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::open_quick_tunnel(port, wait).await });
    match result {
        Ok((url, proc)) => Ok((url, store_process(proc))),
        Err(e) => Err(map_tunnel_err(py, e)),
    }
}

#[pyfunction]
#[pyo3(signature = (port, wait_s=None, verify_dns=true, api_key=None))]
fn open_quick_tunnel_with_dns_verification(
    py: Python,
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
        Err(e) => Err(map_tunnel_err(py, e)),
    }
}

#[pyfunction]
fn open_managed_tunnel(py: Python, token: String) -> PyResult<ProcessHandle> {
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::open_managed_tunnel(&token).await });
    match result {
        Ok(proc) => Ok(store_process(proc)),
        Err(e) => Err(map_tunnel_err(py, e)),
    }
}

#[pyfunction]
#[pyo3(signature = (token, timeout_s=None))]
fn open_managed_tunnel_with_connection_wait(
    py: Python,
    token: String,
    timeout_s: Option<f64>,
) -> PyResult<ProcessHandle> {
    let timeout = timeout_s.unwrap_or(30.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::open_managed_tunnel_with_connection_wait(&token, timeout).await
    });
    match result {
        Ok(proc) => Ok(store_process(proc)),
        Err(e) => Err(map_tunnel_err(py, e)),
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
fn rotate_tunnel(py: Python, api_key: String, port: u16, backend_url: Option<String>) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::rotate_tunnel(&api_key, port, backend_url).await
    });
    match result {
        Ok(res) => pythonize::pythonize(py, &res)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string())),
        Err(e) => Err(map_tunnel_err(py, e)),
    }
}

#[pyfunction]
#[pyo3(signature = (api_key, port, subdomain=None))]
fn create_tunnel(py: Python, api_key: String, port: u16, subdomain: Option<String>) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::create_tunnel(&api_key, port, subdomain).await
    });
    match result {
        Ok(res) => pythonize::pythonize(py, &res)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string())),
        Err(e) => Err(map_tunnel_err(py, e)),
    }
}

#[pyfunction]
#[pyo3(signature = (url, timeout_s=None, api_key=None))]
fn verify_tunnel_dns_resolution(
    py: Python,
    url: String,
    timeout_s: Option<f64>,
    api_key: Option<String>,
) -> PyResult<()> {
    let timeout = timeout_s.unwrap_or(60.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::verify_tunnel_dns_resolution(&url, "tunnel", timeout, api_key).await
    });
    result.map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (host, port, api_key=None, timeout_s=None))]
fn wait_for_health_check(
    py: Python,
    host: String,
    port: u16,
    api_key: Option<String>,
    timeout_s: Option<f64>,
) -> PyResult<()> {
    let timeout = timeout_s.unwrap_or(30.0);
    let result = RUNTIME.block_on(async move {
        tunnels::cloudflared::wait_for_health_check(&host, port, api_key, timeout).await
    });
    result.map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (prefer_system=None))]
fn get_cloudflared_path(prefer_system: Option<bool>) -> PyResult<Option<String>> {
    let prefer = prefer_system.unwrap_or(true);
    Ok(tunnels::cloudflared::get_cloudflared_path(prefer).map(|p| p.to_string_lossy().to_string()))
}

#[pyfunction]
#[pyo3(signature = (force=None))]
fn ensure_cloudflared_installed(py: Python, force: Option<bool>) -> PyResult<String> {
    let force = force.unwrap_or(false);
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::ensure_cloudflared_installed(force).await });
    result
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
fn require_cloudflared(py: Python) -> PyResult<String> {
    let result = RUNTIME.block_on(async move { tunnels::cloudflared::require_cloudflared().await });
    result
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (port, host=None))]
fn is_port_available(port: u16, host: Option<String>) -> PyResult<bool> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    Ok(tunnels::ports::is_port_available(port, &host))
}

#[pyfunction]
#[pyo3(signature = (start_port, host=None, max_attempts=None))]
fn find_available_port(
    py: Python,
    start_port: u16,
    host: Option<String>,
    max_attempts: Option<u16>,
) -> PyResult<u16> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    let max = max_attempts.unwrap_or(100);
    tunnels::ports::find_available_port(start_port, &host, max).map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
fn kill_port(py: Python, port: u16) -> PyResult<bool> {
    tunnels::ports::kill_port(port).map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (port, behavior, host=None, max_search=None))]
fn acquire_port(
    py: Python,
    port: u16,
    behavior: String,
    host: Option<String>,
    max_search: Option<u16>,
) -> PyResult<u16> {
    let host = host.unwrap_or_else(|| "0.0.0.0".to_string());
    let behavior_enum = match behavior.as_str() {
        "fail" => tunnels::ports::PortConflictBehavior::Fail,
        "evict" => tunnels::ports::PortConflictBehavior::Evict,
        "find_new" => tunnels::ports::PortConflictBehavior::FindNew,
        _ => return Err(PyValueError::new_err("unknown conflict behavior")),
    };
    let max = max_search.unwrap_or(100);
    tunnels::ports::acquire_port(port, behavior_enum, &host, max).map_err(|e| map_tunnel_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (backend, local_port, api_key=None, backend_url=None, verify_local=None, verify_dns=true, progress=false))]
fn tunnel_open(
    py: Python,
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
        Err(e) => Err(map_tunnel_err(py, e)),
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
            .map_err(map_core_err)
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
        }).map_err(map_core_err)
    }

    #[pyo3(signature = (request))]
    fn submit_mipro(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: MiproJobRequest = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.jobs().submit_mipro(req).await
        }).map_err(map_core_err)
    }

    #[pyo3(signature = (request))]
    fn submit_job_raw(&self, py: Python, request: PyObject) -> PyResult<String> {
        let req: serde_json::Value = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        RUNTIME.block_on(async {
            self.inner.jobs().submit_raw(req).await
        }).map_err(map_core_err)
    }

    #[pyo3(signature = (job_id))]
    fn get_job_status(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.jobs().get_status(job_id).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, timeout_secs=3600.0, interval_secs=15.0))]
    fn poll_job(&self, py: Python, job_id: &str, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.jobs().poll_until_complete(job_id, timeout_secs, interval_secs).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id, reason=None))]
    fn cancel_job(&self, job_id: &str, reason: Option<&str>) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.inner.jobs().cancel(job_id, reason).await
        }).map_err(map_core_err)
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
        }).map_err(map_core_err)
    }

    #[pyo3(signature = (job_id))]
    fn get_eval_status(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().get_status(job_id).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn get_eval_results(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().get_results(job_id).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn download_eval_traces(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let bytes = RUNTIME.block_on(async {
            self.inner.eval().download_traces(job_id).await
        }).map_err(map_core_err)?;
        Ok(PyBytes::new_bound(py, &bytes).into())
    }

    #[pyo3(signature = (job_id, timeout_secs=1800.0, interval_secs=10.0))]
    fn poll_eval(&self, py: Python, job_id: &str, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().poll_until_complete(job_id, timeout_secs, interval_secs).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn cancel_eval(&self, job_id: &str) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.inner.eval().cancel(job_id).await
        }).map_err(map_core_err)
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
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (kind=None, limit=None))]
    fn list_graphs(&self, py: Python, kind: Option<&str>, limit: Option<i32>) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.graphs().list_graphs(kind, limit).await
        }).map_err(map_core_err)?;
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
        }).map_err(map_core_err)?;
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
        }).map_err(map_core_err)?;
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
            .map_err(map_core_err)?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[staticmethod]
    #[pyo3(signature = (job_id, api_key=None, base_url=None))]
    fn from_job_id(job_id: &str, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        eprintln!("[PY-BIND] PromptLearningJob.from_job_id: job_id={}", job_id);
        let job = RustPromptLearningJob::from_job_id(job_id, api_key, base_url)
            .map_err(map_core_err)?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[getter]
    fn job_id(&self) -> Option<String> {
        self.inner.lock().unwrap().job_id().map(|s| s.to_string())
    }

    fn submit(&self) -> PyResult<String> {
        eprintln!("[PY-BIND] PromptLearningJob.submit called");
        let mut job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.submit().await
        });
        match &result {
            Ok(job_id) => eprintln!("[PY-BIND] PromptLearningJob.submit: SUCCESS job_id={}", job_id),
            Err(e) => eprintln!("[PY-BIND] PromptLearningJob.submit: ERROR {:?}", e),
        }
        result.map_err(map_core_err)
    }

    fn get_status(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let job_id_str = job.job_id().map(|s| s.to_string()).unwrap_or_default();
        eprintln!("[PY-BIND] PromptLearningJob.get_status: job_id={}", job_id_str);
        let result = RUNTIME.block_on(async {
            job.get_status().await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (timeout_secs=3600.0, interval_secs=15.0))]
    fn poll_until_complete(&self, py: Python, timeout_secs: f64, interval_secs: f64) -> PyResult<PyObject> {
        eprintln!("[PY-BIND] PromptLearningJob.poll_until_complete: timeout={}s interval={}s", timeout_secs, interval_secs);
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.poll_until_complete(timeout_secs, interval_secs).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (timeout_secs=3600.0))]
    fn stream_until_complete(&self, py: Python, timeout_secs: f64) -> PyResult<PyObject> {
        eprintln!("[PY-BIND] stream_until_complete called: timeout_secs={}", timeout_secs);
        
        // Get job_id first while holding lock briefly
        let job_id = {
            let job = self.inner.lock().unwrap();
            job.job_id().map(|s| s.to_string()).unwrap_or_else(|| "unknown".to_string())
        };
        eprintln!("[PY-BIND] stream_until_complete: job_id={}", job_id);
        
        eprintln!("[PY-BIND] stream_until_complete: entering RUNTIME.block_on (GIL held)");
        
        // Note: We keep GIL here because we need to hold the Mutex lock during the async operation
        // Releasing GIL while holding a Rust Mutex could cause issues
        let mut job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.stream_until_complete::<fn(&synth_ai_core::orchestration::ParsedEvent)>(timeout_secs, None).await
        });
        
        match &result {
            Ok(r) => eprintln!(
                "[PY-BIND] stream_until_complete: SUCCESS status={:?} best_score={:?}",
                r.status, r.best_score
            ),
            Err(e) => eprintln!("[PY-BIND] stream_until_complete: ERROR {:?}", e),
        }
        
        let result = result.map_err(map_core_err)?;
        
        eprintln!("[PY-BIND] stream_until_complete: pythonizing result");
        use std::io::Write;
        let _ = std::io::stderr().flush();

        let py_result = pythonize::pythonize(py, &result);
        eprintln!("[PY-BIND] stream_until_complete: pythonize done");
        let _ = std::io::stderr().flush();

        py_result
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (reason=None))]
    fn cancel(&self, reason: Option<&str>) -> PyResult<()> {
        let job = self.inner.lock().unwrap();
        RUNTIME.block_on(async {
            job.cancel(reason).await
        }).map_err(map_core_err)
    }

    fn get_results(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_results().await
        }).map_err(map_core_err)?;
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
// Orchestration - GraphEvolveJob
// =============================================================================

#[pyclass]
struct GraphEvolveJob {
    inner: std::sync::Mutex<CoreGraphEvolveJob>,
}

#[pymethods]
impl GraphEvolveJob {
    #[staticmethod]
    #[pyo3(signature = (payload, api_key=None, base_url=None))]
    fn from_payload(py: Python, payload: PyObject, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = CoreGraphEvolveJob::from_payload(payload_value, api_key, base_url)
            .map_err(map_core_err)?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[staticmethod]
    #[pyo3(signature = (job_id, api_key=None, base_url=None))]
    fn from_job_id(job_id: &str, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let job = CoreGraphEvolveJob::from_job_id(job_id, api_key, base_url)
            .map_err(map_core_err)?;
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
        }).map_err(map_core_err)
    }

    fn get_status(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_status().await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn start(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.start().await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (since_seq=0, limit=1000))]
    fn get_events(&self, py: Python, since_seq: i64, limit: i64) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_events(since_seq, limit).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_metrics(&self, py: Python, query: &str) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_metrics(query).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn download_prompt(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.download_prompt().await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn download_graph_txt(&self) -> PyResult<String> {
        let job = self.inner.lock().unwrap();
        RUNTIME.block_on(async {
            job.download_graph_txt().await
        }).map_err(map_core_err)
    }

    fn run_inference(&self, py: Python, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.run_inference(payload_value).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_graph_record(&self, py: Python, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.get_graph_record(payload_value).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn query_workflow_state(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.query_workflow_state().await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (reason=None))]
    fn cancel(&self, py: Python, reason: Option<String>) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async {
            job.cancel(reason).await
        }).map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Data Types (core/data)
// =============================================================================

macro_rules! wrap_data_type {
    ($wrapper:ident, $inner:ty, $pyname:literal) => {
        #[pyclass(name = $pyname)]
        #[derive(Clone)]
        struct $wrapper {
            inner: $inner,
        }

        #[pymethods]
        impl $wrapper {
            #[staticmethod]
            fn from_dict(py: Python, obj: PyObject) -> PyResult<Self> {
                let parsed: $inner = pythonize::depythonize(obj.bind(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: parsed })
            }

            fn to_dict(&self, py: Python) -> PyResult<PyObject> {
                pythonize::pythonize(py, &self.inner)
                    .map(|b| b.unbind())
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
                let value = serde_json::to_value(&self.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                match value.get(name) {
                    Some(field) => pythonize::pythonize(py, field)
                        .map(|b| b.unbind())
                        .map_err(|e| PyValueError::new_err(e.to_string())),
                    None => Err(PyAttributeError::new_err(format!(
                        "{} has no attribute '{}'",
                        $pyname, name
                    ))),
                }
            }

            fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
                if name == "inner" {
                    return Err(PyAttributeError::new_err("cannot set internal attribute"));
                }
                let mut obj_value = serde_json::to_value(&self.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let mut map = match obj_value {
                    serde_json::Value::Object(map) => map,
                    _ => {
                        return Err(PyAttributeError::new_err(
                            "object does not support attribute assignment",
                        ))
                    }
                };
                let new_value: serde_json::Value = pythonize::depythonize(value.bind(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                map.insert(name.to_string(), new_value);
                obj_value = serde_json::Value::Object(map);
                self.inner = serde_json::from_value(obj_value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(())
            }
        }
    };
}

// ArtifactPy and ContextOverridePy defined manually below with extra methods
wrap_data_type!(ContextOverrideStatusPy, ContextOverrideStatus, "ContextOverrideStatus");
wrap_data_type!(ApplicationStatusPy, ApplicationStatus, "ApplicationStatus");
wrap_data_type!(ApplicationErrorTypePy, ApplicationErrorType, "ApplicationErrorType");
wrap_data_type!(RubricPy, Rubric, "Rubric");
wrap_data_type!(CriterionPy, Criterion, "Criterion");
wrap_data_type!(CriterionExamplePy, CriterionExample, "CriterionExample");
wrap_data_type!(JudgementPy, Judgement, "Judgement");
wrap_data_type!(RubricAssignmentPy, RubricAssignment, "RubricAssignment");
wrap_data_type!(CriterionScoreDataPy, CriterionScoreData, "CriterionScoreData");
wrap_data_type!(ObjectiveSpecPy, ObjectiveSpec, "ObjectiveSpec");
wrap_data_type!(RewardObservationPy, RewardObservation, "RewardObservation");
wrap_data_type!(OutcomeObjectiveAssignmentPy, OutcomeObjectiveAssignment, "OutcomeObjectiveAssignment");
wrap_data_type!(EventObjectiveAssignmentPy, EventObjectiveAssignment, "EventObjectiveAssignment");
wrap_data_type!(InstanceObjectiveAssignmentPy, InstanceObjectiveAssignment, "InstanceObjectiveAssignment");
wrap_data_type!(OutcomeRewardRecordPy, OutcomeRewardRecord, "OutcomeRewardRecord");
wrap_data_type!(EventRewardRecordPy, EventRewardRecord, "EventRewardRecord");
wrap_data_type!(RewardAggregatesPy, RewardAggregates, "RewardAggregates");
wrap_data_type!(CalibrationExamplePy, CalibrationExample, "CalibrationExample");
wrap_data_type!(GoldExamplePy, GoldExample, "GoldExample");
wrap_data_type!(TracingEventPy, TracingEvent, "TracingEvent");
wrap_data_type!(TimeRecordPy, TimeRecord, "TimeRecord");
wrap_data_type!(MessageContentPy, MessageContent, "MessageContent");
wrap_data_type!(MarkovBlanketMessagePy, MarkovBlanketMessage, "MarkovBlanketMessage");
wrap_data_type!(SessionTimeStepPy, SessionTimeStep, "SessionTimeStep");
wrap_data_type!(SessionTracePy, SessionTrace, "SessionTrace");
wrap_data_type!(LLMUsagePy, LLMUsage, "LLMUsage");
wrap_data_type!(LLMRequestParamsPy, LLMRequestParams, "LLMRequestParams");
wrap_data_type!(LLMContentPartPy, LLMContentPart, "LLMContentPart");
wrap_data_type!(LLMMessagePy, LLMMessage, "LLMMessage");
wrap_data_type!(ToolCallSpecPy, ToolCallSpec, "ToolCallSpec");
wrap_data_type!(ToolCallResultPy, ToolCallResult, "ToolCallResult");
wrap_data_type!(LLMChunkPy, LLMChunk, "LLMChunk");
wrap_data_type!(LLMCallRecordPy, LLMCallRecord, "LLMCallRecord");
wrap_data_type!(GEPAProgressPy, GEPAProgress, "GEPAProgress");
wrap_data_type!(CandidateInfoPy, CandidateInfo, "CandidateInfo");

// Manual definitions for types with extra methods
#[pyclass(name = "Artifact")]
#[derive(Clone)]
struct ArtifactPy {
    inner: Artifact,
}

#[pymethods]
impl ArtifactPy {
    #[staticmethod]
    fn from_dict(py: Python, obj: PyObject) -> PyResult<Self> {
        let parsed: Artifact = pythonize::depythonize(obj.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        pythonize::pythonize(py, &self.inner)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        match value.get(name) {
            Some(field) => pythonize::pythonize(py, field)
                .map(|b| b.unbind())
                .map_err(|e| PyValueError::new_err(e.to_string())),
            None => Err(PyAttributeError::new_err(format!(
                "Artifact has no attribute '{}'",
                name
            ))),
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        if name == "inner" {
            return Err(PyAttributeError::new_err("cannot set internal attribute"));
        }
        let mut obj_value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut map = match obj_value {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(PyAttributeError::new_err(
                    "object does not support attribute assignment",
                ))
            }
        };
        let new_value: serde_json::Value = pythonize::depythonize(value.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        map.insert(name.to_string(), new_value);
        obj_value = serde_json::Value::Object(map);
        self.inner = serde_json::from_value(obj_value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[pyo3(signature = (max_size_bytes=10 * 1024 * 1024))]
    fn validate_size(&self, max_size_bytes: i64) -> PyResult<()> {
        self.inner
            .validate_size(max_size_bytes)
            .map_err(PyValueError::new_err)
    }

    fn compute_size(&mut self) {
        self.inner.compute_size();
    }
}

#[pyclass(name = "ContextOverride")]
#[derive(Clone)]
struct ContextOverridePy {
    inner: ContextOverride,
}

#[pymethods]
impl ContextOverridePy {
    #[staticmethod]
    fn from_dict(py: Python, obj: PyObject) -> PyResult<Self> {
        let parsed: ContextOverride = pythonize::depythonize(obj.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        pythonize::pythonize(py, &self.inner)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        match value.get(name) {
            Some(field) => pythonize::pythonize(py, field)
                .map(|b| b.unbind())
                .map_err(|e| PyValueError::new_err(e.to_string())),
            None => Err(PyAttributeError::new_err(format!(
                "ContextOverride has no attribute '{}'",
                name
            ))),
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        if name == "inner" {
            return Err(PyAttributeError::new_err("cannot set internal attribute"));
        }
        let mut obj_value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut map = match obj_value {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(PyAttributeError::new_err(
                    "object does not support attribute assignment",
                ))
            }
        };
        let new_value: serde_json::Value = pythonize::depythonize(value.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        map.insert(name.to_string(), new_value);
        obj_value = serde_json::Value::Object(map);
        self.inner = serde_json::from_value(obj_value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    fn file_count(&self) -> usize {
        self.inner.file_count()
    }

    fn env_var_count(&self) -> usize {
        self.inner.env_var_count()
    }
}

// =============================================================================
// Progress Tracker
// =============================================================================

#[pyclass(name = "ProgressTracker")]
struct ProgressTrackerPy {
    inner: Mutex<ProgressTracker>,
}

#[pymethods]
impl ProgressTrackerPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(ProgressTracker::new()),
        }
    }

    fn update(&self, py: Python, event: PyObject) -> PyResult<()> {
        let value: serde_json::Value = pythonize::depythonize(event.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let parsed = synth_ai_core::orchestration::EventParser::parse(&value);
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        guard.update(&parsed);
        Ok(())
    }

    fn summary(&self, py: Python) -> PyResult<PyObject> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        let summary = guard.to_summary();
        pythonize::pythonize(py, &summary)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn progress(&self, py: Python) -> PyResult<PyObject> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        pythonize::pythonize(py, &guard.progress)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn candidates(&self, py: Python) -> PyResult<PyObject> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        pythonize::pythonize(py, &guard.candidates)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Streaming
// =============================================================================

struct PyCallbackHandler {
    callback: Py<PyAny>,
}

impl StreamHandler for PyCallbackHandler {
    fn handle(&self, message: &StreamMessage) {
        Python::with_gil(|py| {
            if let Ok(payload) = pythonize::pythonize(py, message) {
                let _ = self.callback.call1(py, (payload,));
            }
        });
    }
}

#[pyclass(name = "StreamEndpoints")]
#[derive(Clone)]
struct StreamEndpointsPy {
    inner: StreamEndpoints,
}

#[pymethods]
impl StreamEndpointsPy {
    #[staticmethod]
    fn learning(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::learning(job_id),
        }
    }

    #[staticmethod]
    fn prompt_learning(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::prompt_learning(job_id),
        }
    }

    #[staticmethod]
    fn eval(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::eval(job_id),
        }
    }

    #[staticmethod]
    fn rl(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::rl(job_id),
        }
    }

    #[staticmethod]
    fn graph_optimization(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::graph_optimization(job_id),
        }
    }

    #[staticmethod]
    fn graph_evolve(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::graph_evolve(job_id),
        }
    }

    #[staticmethod]
    fn graphgen(job_id: &str) -> Self {
        Self {
            inner: StreamEndpoints::graphgen(job_id),
        }
    }

    fn events_stream_url(&self) -> Option<String> {
        self.inner.events_stream_url()
    }
}

#[pyclass(name = "JobStreamer")]
struct JobStreamerPy {
    inner: Mutex<CoreJobStreamer>,
}

#[pymethods]
impl JobStreamerPy {
    #[new]
    #[pyo3(signature = (base_url, api_key, job_id, endpoints=None, handlers=None))]
    fn new(
        base_url: &str,
        api_key: &str,
        job_id: &str,
        endpoints: Option<StreamEndpointsPy>,
        handlers: Option<Vec<PyObject>>,
    ) -> Self {
        let mut streamer = CoreJobStreamer::new(base_url, api_key, job_id);
        if let Some(endpoints) = endpoints {
            streamer = streamer.with_endpoints(endpoints.inner);
        }
        if let Some(callbacks) = handlers {
            for callback in callbacks {
                streamer.add_handler(PyCallbackHandler { callback: callback.into() });
            }
        }
        Self {
            inner: Mutex::new(streamer),
        }
    }

    fn add_handler(&self, handler: PyObject) -> PyResult<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("job streamer lock poisoned"))?;
        guard.add_handler(PyCallbackHandler { callback: handler.into() });
        Ok(())
    }

    fn stream_until_terminal(&self, py: Python) -> PyResult<PyObject> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("job streamer lock poisoned"))?;
        let result = RUNTIME
            .block_on(async { guard.stream_until_terminal().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn poll_status(&self, py: Python) -> PyResult<PyObject> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("job streamer lock poisoned"))?;
        let result = RUNTIME
            .block_on(async { guard.poll_status().await })
            .map_err(map_core_err)?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Tracing
// =============================================================================

#[pyclass(name = "SessionTracer")]
struct SessionTracerPy {
    inner: Arc<CoreSessionTracer>,
}

#[pymethods]
impl SessionTracerPy {
    #[staticmethod]
    fn memory() -> PyResult<Self> {
        let storage = RUNTIME
            .block_on(async { LibsqlTraceStorage::new_memory().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(CoreSessionTracer::new(Arc::new(storage))),
        })
    }

    #[staticmethod]
    fn file(path: &str) -> PyResult<Self> {
        let storage = RUNTIME
            .block_on(async { LibsqlTraceStorage::new_file(path).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(CoreSessionTracer::new(Arc::new(storage))),
        })
    }

    #[pyo3(signature = (session_id=None, metadata=None))]
    fn start_session(
        &self,
        py: Python,
        session_id: Option<&str>,
        metadata: Option<PyObject>,
    ) -> PyResult<String> {
        let metadata_value: serde_json::Value = match metadata {
            Some(obj) => pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            None => serde_json::json!({}),
        };
        let metadata_map = metadata_value
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<HashMap<String, serde_json::Value>>();
        let result = RUNTIME
            .block_on(async { self.inner.start_session(session_id, metadata_map).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result)
    }

    fn record_event(&self, py: Python, event: PyObject) -> PyResult<Option<i64>> {
        let value: serde_json::Value = pythonize::depythonize(event.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tracing_event: TracingEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(async { self.inner.record_event(tracing_event).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result)
    }

    #[pyo3(signature = (save=true))]
    fn end_session(&self, py: Python, save: bool) -> PyResult<PyObject> {
        let trace = RUNTIME
            .block_on(async { self.inner.end_session(save).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &trace)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_session(&self, py: Python, session_id: &str) -> PyResult<PyObject> {
        let trace = RUNTIME
            .block_on(async { self.inner.get_session(session_id).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &trace)
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
    // Core error types
    let py = m.py();
    m.add("CoreError", py.get_type_bound::<CoreError>())?;
    m.add("InvalidInputError", py.get_type_bound::<InvalidInputError>())?;
    m.add("UrlParseError", py.get_type_bound::<UrlParseError>())?;
    m.add("HttpRequestError", py.get_type_bound::<HttpRequestError>())?;
    m.add("HttpResponseError", py.get_type_bound::<HttpResponseError>())?;
    m.add("AuthenticationError", py.get_type_bound::<AuthenticationError>())?;
    m.add("ValidationError", py.get_type_bound::<ValidationError>())?;
    m.add("UsageLimitError", py.get_type_bound::<UsageLimitError>())?;
    m.add("JobError", py.get_type_bound::<JobError>())?;
    m.add("ConfigError", py.get_type_bound::<ConfigError>())?;
    m.add("TimeoutError", py.get_type_bound::<TimeoutError>())?;
    m.add("ProtocolError", py.get_type_bound::<ProtocolError>())?;
    m.add("InternalError", py.get_type_bound::<InternalError>())?;

    // URLs
    m.add_function(wrap_pyfunction!(normalize_backend_base, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_inference_base, m)?)?;
    m.add_function(wrap_pyfunction!(make_local_api_url, m)?)?;
    m.add_function(wrap_pyfunction!(validate_task_app_url, m)?)?;

    // Events
    m.add_function(wrap_pyfunction!(poll_events, m)?)?;

    // HTTP + SSE
    m.add_class::<HttpClientPy>()?;
    m.add_function(wrap_pyfunction!(stream_sse, m)?)?;

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
    m.add_class::<GraphEvolveJob>()?;

    // Data types (NEW)
    m.add_class::<ArtifactPy>()?;
    m.add_class::<ContextOverridePy>()?;
    m.add_class::<ContextOverrideStatusPy>()?;
    m.add_class::<ApplicationStatusPy>()?;
    m.add_class::<ApplicationErrorTypePy>()?;
    m.add_class::<RubricPy>()?;
    m.add_class::<CriterionPy>()?;
    m.add_class::<CriterionExamplePy>()?;
    m.add_class::<JudgementPy>()?;
    m.add_class::<RubricAssignmentPy>()?;
    m.add_class::<CriterionScoreDataPy>()?;
    m.add_class::<ObjectiveSpecPy>()?;
    m.add_class::<RewardObservationPy>()?;
    m.add_class::<OutcomeObjectiveAssignmentPy>()?;
    m.add_class::<EventObjectiveAssignmentPy>()?;
    m.add_class::<InstanceObjectiveAssignmentPy>()?;
    m.add_class::<OutcomeRewardRecordPy>()?;
    m.add_class::<EventRewardRecordPy>()?;
    m.add_class::<RewardAggregatesPy>()?;
    m.add_class::<CalibrationExamplePy>()?;
    m.add_class::<GoldExamplePy>()?;
    m.add_class::<TracingEventPy>()?;
    m.add_class::<TimeRecordPy>()?;
    m.add_class::<MessageContentPy>()?;
    m.add_class::<MarkovBlanketMessagePy>()?;
    m.add_class::<SessionTimeStepPy>()?;
    m.add_class::<SessionTracePy>()?;
    m.add_class::<LLMUsagePy>()?;
    m.add_class::<LLMRequestParamsPy>()?;
    m.add_class::<LLMContentPartPy>()?;
    m.add_class::<LLMMessagePy>()?;
    m.add_class::<ToolCallSpecPy>()?;
    m.add_class::<ToolCallResultPy>()?;
    m.add_class::<LLMChunkPy>()?;
    m.add_class::<LLMCallRecordPy>()?;
    m.add_class::<GEPAProgressPy>()?;
    m.add_class::<CandidateInfoPy>()?;
    m.add_class::<ProgressTrackerPy>()?;

    // Streaming (NEW)
    m.add_class::<StreamEndpointsPy>()?;
    m.add_class::<JobStreamerPy>()?;

    // Tracing (NEW)
    m.add_class::<SessionTracerPy>()?;

    // Local API Client (NEW)
    m.add_class::<TaskAppClientPy>()?;

    Ok(())
}
