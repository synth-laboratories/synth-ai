use once_cell::sync::Lazy;
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAnyMethods, PyBool, PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods, PyDate,
    PyDateTime, PyDict, PyDictMethods, PyFloat, PyList, PyListMethods, PyLong, PySet, PySetMethods,
    PyString, PyStringMethods, PyTuple, PyTupleMethods, PyType,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Method;
use futures_util::StreamExt;
use serde_json::Value;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::{engine::general_purpose, Engine as _};

use synth_ai_core::config::CoreConfig;
use synth_ai_core::data::{Artifact, ContextOverride, SuccessStatus};
use synth_ai_core::events::{poll_events as core_poll_events, EventKind};
use synth_ai_core::http::{HttpClient as RustHttpClient, HttpError as RustHttpError, MultipartFile as RustMultipartFile};
use synth_ai_core::tunnels;
use synth_ai_core::tunnels::cloudflared::ManagedProcess;
use synth_ai_core::tunnels::lease_client::LeaseClient as RustLeaseClient;
use synth_ai_core::tunnels::types::TunnelBackend;
use synth_ai_core::tunnels::types::{
    LeaseInfo as RustLeaseInfo, ConnectorStatus as RustConnectorStatus,
    GatewayStatus as RustGatewayStatus, Diagnostics as RustDiagnostics,
    TunnelHandle as RustTunnelHandle,
};
use synth_ai_core::tunnels::errors::TunnelError;
use synth_ai_core::urls::{
    make_local_api_url as core_make_local_api_url,
    normalize_backend_base as core_normalize_backend_base,
    normalize_inference_base as core_normalize_inference_base,
    validate_task_app_url as core_validate_task_app_url,
    backend_url_base as core_backend_url_base,
    backend_url_api as core_backend_url_api,
    backend_url_synth_research_base as core_backend_url_synth_research_base,
    backend_url_synth_research_openai as core_backend_url_synth_research_openai,
    backend_url_synth_research_anthropic as core_backend_url_synth_research_anthropic,
    frontend_url_base as core_frontend_url_base,
    join_url as core_join_url,
    local_backend_url as core_local_backend_url,
    backend_health_url as core_backend_health_url,
    backend_me_url as core_backend_me_url,
    backend_demo_keys_url as core_backend_demo_keys_url,
};
use synth_ai_core::CoreError;
use synth_ai_core::localapi::{EnvClient as RustEnvClient, TaskAppClient as RustTaskAppClient};
use synth_ai_core::localapi::auth as core_localapi_auth;
use synth_ai_core::localapi::helpers as core_localapi_helpers;
use synth_ai_core::localapi::health as core_localapi_health;
use synth_ai_core::localapi::datasets as core_localapi_datasets;
use synth_ai_core::localapi::override_helpers as core_localapi_override;
use synth_ai_core::localapi::proxy as core_localapi_proxy;
use synth_ai_core::localapi::rollout_helpers as core_localapi_rollout;
use synth_ai_core::localapi::tracing_utils as core_localapi_tracing_utils;
use synth_ai_core::localapi::llm_guards as core_localapi_llm_guards;
use synth_ai_core::localapi::trace_helpers as core_localapi_trace;
use synth_ai_core::localapi::validation as core_localapi_validation;
use synth_ai_core::localapi::validators as core_localapi_validators;
use synth_ai_core::localapi::datasets::TaskDatasetSpec as RustTaskDatasetSpec;
use synth_ai_core::localapi::vendors as core_localapi_vendors;
use synth_ai_core::models as core_models;
use synth_ai_core::streaming::{
    JobStreamer as RustJobStreamer, StreamConfig as RustStreamConfig,
    StreamEndpoints as RustStreamEndpoints, StreamType as RustStreamType,
};
use synth_ai_core::sse::stream_sse_request as core_stream_sse_request;
use synth_ai_core::sse::SseStream as CoreSseStream;
use synth_ai_core::trace_upload::{
    TraceUploadClient as RustTraceUploadClient,
    UploadUrlResponse as RustUploadUrlResponse,
};
use synth_ai_core::utils as core_utils;
use synth_ai_core::tracing::{
    EventReward as RustEventReward, LibsqlTraceStorage, MarkovBlanketMessage as RustMarkovBlanketMessage,
    OutcomeReward as RustOutcomeReward, SessionTrace as RustSessionTrace, LLMCallRecord as RustLLMCallRecord,
    SessionTracer as RustSessionTracer, StorageConfig as RustStorageConfig,
    TraceStorage as RustTraceStorage, TracingEvent as RustTracingEvent, QueryParams as RustQueryParams,
    TimeRecord as RustTimeRecord, MessageContent as RustMessageContent, BaseEventFields as RustBaseEventFields,
    LMCAISEvent as RustLMCAISEvent, EnvironmentEvent as RustEnvironmentEvent, RuntimeEvent as RustRuntimeEvent,
    SessionTimeStep as RustSessionTimeStep, LLMUsage as RustLLMUsage,
    LLMRequestParams as RustLLMRequestParams, LLMContentPart as RustLLMContentPart,
    LLMMessage as RustLLMMessage, ToolCallSpec as RustToolCallSpec, ToolCallResult as RustToolCallResult,
    LLMChunk as RustLLMChunk,
};
use synth_ai_core::data::{
    Rubric as RustRubric, Criterion as RustCriterion, CriterionExample as RustCriterionExample,
    Judgement as RustJudgement, CriterionScoreData as RustCriterionScoreData,
    RubricAssignment as RustRubricAssignment,
    ObjectiveSpec as RustObjectiveSpec, RewardObservation as RustRewardObservation,
    OutcomeObjectiveAssignment as RustOutcomeObjectiveAssignment,
    EventObjectiveAssignment as RustEventObjectiveAssignment,
    InstanceObjectiveAssignment as RustInstanceObjectiveAssignment,
    OutcomeRewardRecord as RustOutcomeRewardRecord,
    EventRewardRecord as RustEventRewardRecord,
    RewardAggregates as RustRewardAggregates,
    CalibrationExample as RustCalibrationExample,
    GoldExample as RustGoldExample,
    ContextOverride as RustContextOverride, ContextOverrideStatus as RustContextOverrideStatus,
    Artifact as RustArtifact, ArtifactBundle as RustArtifactBundle,
    data_enum_values as core_data_enum_values,
};
use synth_ai_core::orchestration::schemas::{
    MutationTypeStats as RustMutationTypeStats,
    MutationSummary as RustMutationSummary,
    SeedAnalysis as RustSeedAnalysis,
    PhaseSummary as RustPhaseSummary,
    ProgramCandidate as RustProgramCandidate,
    StageInfo as RustStageInfo,
    SeedInfo as RustSeedInfo,
    TokenUsage as RustSchemaTokenUsage,
    MAX_INSTRUCTION_LENGTH as RUST_MAX_INSTRUCTION_LENGTH,
    MAX_ROLLOUT_SAMPLES as RUST_MAX_ROLLOUT_SAMPLES,
    MAX_SEED_INFO_COUNT as RUST_MAX_SEED_INFO_COUNT,
};
use synth_ai_core::orchestration::base_events::{
    BaseJobEvent as RustBaseJobEvent,
    JobEvent as RustJobEvent,
    CandidateEvent as RustCandidateEvent,
};

static RUNTIME: Lazy<tokio::runtime::Runtime> =
    Lazy::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));
static LOCALAPI_HTTP_CLIENT: Lazy<Mutex<Option<PyObject>>> = Lazy::new(|| Mutex::new(None));

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

#[pyclass(name = "LeaseClient")]
struct LeaseClientPy {
    inner: RustLeaseClient,
}

#[pymethods]
impl LeaseClientPy {
    #[new]
    #[pyo3(signature = (api_key, backend_url=None, timeout_s=None))]
    fn new(api_key: String, backend_url: Option<String>, timeout_s: Option<u64>) -> PyResult<Self> {
        let backend = backend_url
            .or_else(|| std::env::var("SYNTH_BACKEND_URL").ok())
            .unwrap_or_else(|| "https://api.usesynth.ai".to_string());
        let timeout = timeout_s.unwrap_or(30);
        let client = RustLeaseClient::new(api_key, backend, timeout)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: client })
    }

    #[pyo3(signature = (client_instance_id, local_port, local_host="127.0.0.1", app_name=None, requested_ttl_seconds=3600, reuse_connector=true, idempotency_key=None))]
    fn create_lease(
        &self,
        client_instance_id: &str,
        local_port: u16,
        local_host: &str,
        app_name: Option<&str>,
        requested_ttl_seconds: i64,
        reuse_connector: bool,
        idempotency_key: Option<&str>,
    ) -> PyResult<LeaseInfoPy> {
        let client = self.inner.clone();
        let lease = RUNTIME
            .block_on(async move {
                client
                    .create_lease(
                        client_instance_id,
                        local_host,
                        local_port,
                        app_name,
                        requested_ttl_seconds,
                        reuse_connector,
                        idempotency_key,
                    )
                    .await
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(LeaseInfoPy { inner: lease })
    }

    #[pyo3(signature = (lease_id, connected_to_edge=false, gateway_ready=false, local_ready=false, last_error=None))]
    fn heartbeat(
        &self,
        lease_id: &str,
        connected_to_edge: bool,
        gateway_ready: bool,
        local_ready: bool,
        last_error: Option<&str>,
    ) -> PyResult<(String, i64)> {
        let client = self.inner.clone();
        let result = RUNTIME
            .block_on(async move {
                client
                    .heartbeat(
                        lease_id,
                        connected_to_edge,
                        gateway_ready,
                        local_ready,
                        last_error,
                    )
                    .await
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result)
    }

    fn release(&self, lease_id: &str) -> PyResult<()> {
        let client = self.inner.clone();
        RUNTIME
            .block_on(async move { client.release(lease_id).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (client_instance_id=None, include_expired=false))]
    fn list_leases(
        &self,
        client_instance_id: Option<&str>,
        include_expired: bool,
    ) -> PyResult<Vec<LeaseInfoPy>> {
        let client = self.inner.clone();
        let leases = RUNTIME
            .block_on(async move { client.list_leases(client_instance_id, include_expired).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(leases.into_iter().map(|lease| LeaseInfoPy { inner: lease }).collect())
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
    process_id: Option<usize>,
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

    #[getter]
    fn process_id(&self) -> Option<usize> {
        self.process_id
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
        if let Some(process_id) = self.process_id {
            let result = RUNTIME.block_on(async move {
                tunnels::cloudflared::stop_tracked(process_id).await
            });
            return result.map_err(|e| PyValueError::new_err(e.to_string()));
        }
        Ok(())
    }
}

fn map_core_err(py: Python, err: CoreError) -> PyErr {
    let errors_mod = py.import_bound("synth_ai.core.errors");
    if let Ok(errors) = errors_mod {
        match &err {
            CoreError::Authentication(msg) => {
                if let Ok(cls) = errors.getattr("AuthenticationError") {
                    if let Ok(instance) = cls.call1((msg.clone(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Validation(msg) | CoreError::InvalidInput(msg) | CoreError::Protocol(msg) => {
                if let Ok(cls) = errors.getattr("ValidationError") {
                    if let Ok(instance) = cls.call1((msg.clone(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Config(msg) => {
                if let Ok(cls) = errors.getattr("ConfigError") {
                    if let Ok(instance) = cls.call1((msg.clone(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Timeout(msg) => {
                if let Ok(cls) = errors.getattr("TimeoutError") {
                    if let Ok(instance) = cls.call1((msg.clone(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::UsageLimit(info) => {
                if let Ok(cls) = errors.getattr("UsageLimitError") {
                    if let Ok(instance) = cls.call1((
                        info.limit_type.clone(),
                        info.api.clone(),
                        info.current,
                        info.limit,
                        info.tier.clone(),
                        info.retry_after_seconds,
                        info.upgrade_url.clone(),
                    )) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Job(info) => {
                let msg = format!("Job {} failed: {}", info.job_id, info.message);
                if let Ok(cls) = errors.getattr("JobError") {
                    if let Ok(instance) = cls.call1((msg,)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::HttpResponse(info) => {
                let detail: Option<PyObject> = None;
                if let Ok(cls) = errors.getattr("HTTPError") {
                    if let Ok(instance) = cls.call1((
                        info.status,
                        info.url.clone(),
                        info.message.clone(),
                        info.body_snippet.clone(),
                        detail,
                    )) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Http(err) => {
                let status = err.status().map(|s| s.as_u16()).unwrap_or(0);
                let detail: Option<PyObject> = None;
                if let Ok(cls) = errors.getattr("HTTPError") {
                    if let Ok(instance) = cls.call1((
                        status,
                        "",
                        err.to_string(),
                        Option::<String>::None,
                        detail,
                    )) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::UrlParse(err) => {
                if let Ok(cls) = errors.getattr("ValidationError") {
                    if let Ok(instance) = cls.call1((err.to_string(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
            CoreError::Internal(msg) => {
                if let Ok(cls) = errors.getattr("SynthError") {
                    if let Ok(instance) = cls.call1((msg.clone(),)) {
                        return PyErr::from_value_bound(instance);
                    }
                }
            }
        }
    }

    PyValueError::new_err(err.to_string())
}

fn map_tunnel_err(py: Python, err: TunnelError) -> PyErr {
    let errors_mod = py.import_bound("synth_ai.core.tunnels.errors");
    if let Ok(errors) = errors_mod {
        let (cls_name, message) = match &err {
            TunnelError::Config(msg) => ("TunnelConfigurationError", msg.clone()),
            TunnelError::Api(msg) => ("TunnelAPIError", msg.clone()),
            TunnelError::Lease(msg) => ("LeaseError", msg.clone()),
            TunnelError::Connector(msg) => ("ConnectorError", msg.clone()),
            TunnelError::Gateway(msg) => ("GatewayError", msg.clone()),
            TunnelError::LocalApp(msg) => ("LocalAppError", msg.clone()),
            TunnelError::Dns(msg) => ("TunnelError", msg.clone()),
            TunnelError::Process(msg) => ("TunnelError", msg.clone()),
        };
        if let Ok(cls) = errors.getattr(cls_name) {
            if let Ok(instance) = cls.call1((message,)) {
                return PyErr::from_value_bound(instance);
            }
        }
    }
    PyValueError::new_err(err.to_string())
}

fn normalize_value<T>(py: Python, obj: PyObject) -> PyResult<PyObject>
where
    T: DeserializeOwned + Serialize,
{
    let value: Value = pythonize::depythonize(obj.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed: T = serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn normalize_python_json<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<PyObject> {
    if obj.is_none() {
        return Ok(py.None());
    }

    if obj.hasattr("to_dict")? {
        if let Ok(value) = obj.call_method0("to_dict") {
            return normalize_python_json(py, &value);
        }
    }

    if let Ok(dataclasses) = py.import_bound("dataclasses") {
        if let Ok(is_dc) = dataclasses.call_method1("is_dataclass", (obj,)) {
            if is_dc.is_truthy()? && !obj.is_instance_of::<PyType>() {
                if let Ok(asdict) = dataclasses.call_method1("asdict", (obj,)) {
                    return normalize_python_json(py, &asdict);
                }
            }
        }
    }

    if let Ok(dict) = obj.downcast::<PyDict>() {
        let out = PyDict::new_bound(py);
        for (key, value) in dict.iter() {
            let key_str = key.str()?.to_string();
            let normalized = normalize_python_json(py, &value)?;
            out.set_item(key_str, normalized)?;
        }
        return Ok(out.to_object(py));
    }

    if let Ok(list) = obj.downcast::<PyList>() {
        let mut out: Vec<PyObject> = Vec::new();
        for item in list.iter() {
            out.push(normalize_python_json(py, &item)?);
        }
        return Ok(PyList::new_bound(py, out).to_object(py));
    }
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut out: Vec<PyObject> = Vec::new();
        for item in tuple.iter() {
            out.push(normalize_python_json(py, &item)?);
        }
        return Ok(PyTuple::new_bound(py, out).to_object(py));
    }
    if let Ok(set) = obj.downcast::<PySet>() {
        let mut out: Vec<PyObject> = Vec::new();
        for item in set.iter() {
            out.push(normalize_python_json(py, &item)?);
        }
        return Ok(PyList::new_bound(py, out).to_object(py));
    }

    if obj.is_instance_of::<PyDateTime>() || obj.is_instance_of::<PyDate>() {
        if let Ok(value) = obj.call_method0("isoformat") {
            return Ok(value.to_object(py));
        }
    }

    if let Ok(bytes) = obj.downcast::<PyBytes>() {
        let encoded = general_purpose::STANDARD.encode(bytes.as_bytes());
        return Ok(encoded.to_object(py));
    }
    if let Ok(bytearray) = obj.downcast::<PyByteArray>() {
        let encoded = general_purpose::STANDARD.encode(unsafe { bytearray.as_bytes() });
        return Ok(encoded.to_object(py));
    }

    if let Ok(enum_mod) = py.import_bound("enum") {
        if let Ok(enum_cls) = enum_mod.getattr("Enum") {
            if obj.is_instance(&enum_cls)? {
                if let Ok(value) = obj.getattr("value") {
                    return normalize_python_json(py, &value);
                }
            }
        }
    }

    if let Ok(decimal_mod) = py.import_bound("decimal") {
        if let Ok(decimal_cls) = decimal_mod.getattr("Decimal") {
            if obj.is_instance(&decimal_cls)? {
                if let Ok(val) = obj.extract::<f64>() {
                    if val.is_finite() {
                        return Ok(val.to_object(py));
                    }
                }
                return Ok(obj.str()?.to_object(py));
            }
        }
    }

    if let Ok(np) = py.import_bound("numpy") {
        if let Ok(np_generic) = np.getattr("generic") {
            if obj.is_instance(&np_generic)? {
                if let Ok(item) = obj.call_method0("item") {
                    return normalize_python_json(py, &item);
                }
            }
        }
        if let Ok(np_array) = np.getattr("ndarray") {
            if obj.is_instance(&np_array)? {
                if let Ok(list) = obj.call_method0("tolist") {
                    return normalize_python_json(py, &list);
                }
            }
        }
    }

    if obj.is_instance_of::<PyBool>() {
        let val: bool = obj.extract()?;
        return Ok(val.to_object(py));
    }
    if obj.is_instance_of::<PyFloat>() {
        let val: f64 = obj.extract()?;
        if !val.is_finite() {
            return Ok(py.None());
        }
        return Ok(val.to_object(py));
    }
    if obj.is_instance_of::<PyLong>() || obj.is_instance_of::<PyString>() {
        return Ok(obj.to_object(py));
    }

    Ok(obj.to_object(py))
}

#[pyfunction]
fn normalize_for_json(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let bound = payload.bind(py);
    normalize_python_json(py, &bound)
}

#[pyfunction]
fn dumps_http_json(py: Python, payload: PyObject) -> PyResult<String> {
    let bound = payload.bind(py);
    let normalized = normalize_python_json(py, &bound)?;
    let value: Value = pythonize::depythonize(normalized.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&value).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn serialize_trace_for_http(py: Python, trace: PyObject) -> PyResult<String> {
    dumps_http_json(py, trace)
}

fn parse_query_params(py: Python, params: Option<PyObject>) -> PyResult<Vec<(String, String)>> {
    let mut out = Vec::new();
    let Some(obj) = params else {
        return Ok(out);
    };

    if let Ok(dict) = obj.bind(py).downcast::<PyDict>() {
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            let value = if let Ok(s) = v.extract::<String>() {
                s
            } else {
                v.str()?.to_string()
            };
            out.push((key, value));
        }
        return Ok(out);
    }

    if let Ok(list) = obj.extract::<Vec<(String, String)>>(py) {
        out.extend(list);
        return Ok(out);
    }

    Err(PyValueError::new_err("params must be dict or list of (str, str) tuples"))
}

fn parse_form_data(py: Python, data: Option<PyObject>) -> PyResult<HashMap<String, String>> {
    let mut out = HashMap::new();
    let Some(obj) = data else {
        return Ok(out);
    };
    let value: Value = pythonize::depythonize(obj.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if let Value::Object(map) = value {
        for (k, v) in map {
            let text = match v {
                Value::String(s) => s,
                other => other.to_string(),
            };
            out.insert(k, text);
        }
        return Ok(out);
    }
    Err(PyValueError::new_err("data must be a dict"))
}

fn parse_multipart_files(py: Python, files: Option<PyObject>) -> PyResult<Vec<RustMultipartFile>> {
    let Some(obj) = files else {
        return Ok(Vec::new());
    };
    let dict = obj.bind(py).downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("files must be a dict"))?;
    let mut out = Vec::new();

    for (k, v) in dict.iter() {
        let field = k.extract::<String>()?;
        if let Ok((filename, bytes, content_type)) =
            v.extract::<(String, Py<PyBytes>, Option<String>)>()
        {
            let bytes_vec = bytes.bind(py).as_bytes().to_vec();
            out.push(RustMultipartFile::new(
                field,
                filename,
                bytes_vec,
                content_type,
            ));
            continue;
        }
        if let Ok((filename, bytes)) = v.extract::<(String, Py<PyBytes>)>() {
            let bytes_vec = bytes.bind(py).as_bytes().to_vec();
            out.push(RustMultipartFile::new(
                field,
                filename,
                bytes_vec,
                None,
            ));
            continue;
        }
        return Err(PyValueError::new_err(
            "file entries must be (filename, bytes[, content_type]) tuples",
        ));
    }

    Ok(out)
}

fn parse_headers(py: Python, headers: Option<PyObject>) -> PyResult<HeaderMap> {
    let mut map = HeaderMap::new();
    let Some(obj) = headers else {
        return Ok(map);
    };

    let dict = obj
        .bind(py)
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("headers must be a dict"))?;

    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let value = if let Ok(s) = v.extract::<String>() {
            s
        } else {
            v.str()?.to_str()?.to_string()
        };
        let name = HeaderName::from_bytes(key.as_bytes())
            .map_err(|_| PyValueError::new_err(format!("invalid header name: {}", key)))?;
        let value = HeaderValue::from_str(&value)
            .map_err(|_| PyValueError::new_err(format!("invalid header value for {}", key)))?;
        map.insert(name, value);
    }

    Ok(map)
}

fn merge_json_map(target: &mut serde_json::Map<String, Value>, value: Value) {
    if let Value::Object(obj) = value {
        for (k, v) in obj {
            target.insert(k, v);
        }
    }
}

#[pyfunction]
fn normalize_backend_base(py: Python, url: &str) -> PyResult<String> {
    core_normalize_backend_base(url)
        .map(|u| u.to_string())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn normalize_inference_base(py: Python, url: &str) -> PyResult<String> {
    core_normalize_inference_base(url)
        .map(|u| u.to_string())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn make_local_api_url(py: Python, host: &str, port: u16) -> PyResult<String> {
    core_make_local_api_url(host, port)
        .map(|u| u.to_string())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn validate_task_app_url(py: Python, url: &str) -> PyResult<String> {
    core_validate_task_app_url(url)
        .map(|u| u.to_string())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn backend_url_base() -> String {
    core_backend_url_base()
}

#[pyfunction]
fn backend_url_api() -> String {
    core_backend_url_api()
}

#[pyfunction]
fn backend_url_synth_research_base() -> String {
    core_backend_url_synth_research_base()
}

#[pyfunction]
fn backend_url_synth_research_openai() -> String {
    core_backend_url_synth_research_openai()
}

#[pyfunction]
fn backend_url_synth_research_anthropic() -> String {
    core_backend_url_synth_research_anthropic()
}

#[pyfunction]
fn frontend_url_base() -> String {
    core_frontend_url_base()
}

#[pyfunction]
fn join_url(base: &str, path: &str) -> String {
    core_join_url(base, path)
}

#[pyfunction]
#[pyo3(signature = (host="localhost", port=8000))]
fn local_backend_url(host: &str, port: u16) -> String {
    core_local_backend_url(host, port)
}

#[pyfunction]
fn backend_health_url(base_url: &str) -> String {
    core_backend_health_url(base_url)
}

#[pyfunction]
fn backend_me_url(base_url: &str) -> String {
    core_backend_me_url(base_url)
}

#[pyfunction]
fn backend_demo_keys_url(base_url: &str) -> String {
    core_backend_demo_keys_url(base_url)
}

// =============================================================================
// Utils
// =============================================================================

#[pyfunction]
fn strip_json_comments(raw: &str) -> String {
    core_utils::strip_json_comments(raw)
}

#[pyfunction]
fn create_and_write_json(py: Python, path: &str, content: PyObject) -> PyResult<()> {
    let value = value_from_pyobject(py, content)?;
    core_utils::create_and_write_json(Path::new(path), &value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_json_to_dict(py: Python, path: &str) -> PyResult<PyObject> {
    let value = core_utils::load_json_to_value(Path::new(path));
    value_to_pyobject(py, &value)
}

#[pyfunction]
fn deep_update(py: Python, base: PyObject, overrides: PyObject) -> PyResult<PyObject> {
    let mut base_value = value_from_pyobject(py, base)?;
    let overrides_value = value_from_pyobject(py, overrides)?;
    synth_ai_core::config::deep_update(&mut base_value, &overrides_value);
    value_to_pyobject(py, &base_value)
}

#[pyfunction]
fn repo_root() -> Option<String> {
    core_utils::repo_root().map(|p| p.to_string_lossy().to_string())
}

#[pyfunction]
fn synth_home_dir() -> String {
    core_utils::synth_home_dir().to_string_lossy().to_string()
}

#[pyfunction]
fn synth_user_config_path() -> String {
    core_utils::synth_user_config_path().to_string_lossy().to_string()
}

#[pyfunction]
fn synth_localapi_config_path() -> String {
    core_utils::synth_localapi_config_path().to_string_lossy().to_string()
}

#[pyfunction]
fn synth_bin_dir() -> String {
    core_utils::synth_bin_dir().to_string_lossy().to_string()
}

#[pyfunction]
fn is_file_type(path: &str, ext: &str) -> bool {
    core_utils::is_file_type(Path::new(path), ext)
}

#[pyfunction]
fn validate_file_type(py: Python, path: &str, ext: &str) -> PyResult<()> {
    core_utils::validate_file_type(Path::new(path), ext)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn is_hidden_path(path: &str, root: &str) -> bool {
    core_utils::is_hidden_path(Path::new(path), Path::new(root))
}

#[pyfunction]
fn get_bin_path(name: &str) -> Option<String> {
    core_utils::get_bin_path(name).map(|p| p.to_string_lossy().to_string())
}

#[pyfunction]
#[pyo3(signature = (dir_name, file_extension="json"))]
fn get_home_config_file_paths(dir_name: &str, file_extension: &str) -> Vec<String> {
    core_utils::get_home_config_file_paths(dir_name, file_extension)
        .into_iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect()
}

#[pyfunction]
fn find_config_path(bin: &str, home_subdir: &str, filename: &str) -> Option<String> {
    core_utils::find_config_path(Path::new(bin), home_subdir, filename)
        .map(|p| p.to_string_lossy().to_string())
}

#[pyfunction]
#[pyo3(signature = (app, repo_root=None))]
fn compute_import_paths(app: &str, repo_root: Option<String>) -> Vec<String> {
    let repo = repo_root.as_ref().map(|r| Path::new(r));
    core_utils::compute_import_paths(Path::new(app), repo)
}

#[pyfunction]
fn cleanup_paths(py: Python, file: &str, dir: &str) -> PyResult<()> {
    core_utils::cleanup_paths(Path::new(file), Path::new(dir))
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn ensure_private_dir(path: &str) -> PyResult<()> {
    core_utils::ensure_private_dir(Path::new(path))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path, content, mode=None))]
fn write_private_text(path: &str, content: &str, mode: Option<u32>) -> PyResult<()> {
    let mode = mode.unwrap_or(core_utils::PRIVATE_FILE_MODE);
    core_utils::write_private_text(Path::new(path), content, mode)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path, data))]
fn write_private_json(py: Python, path: &str, data: PyObject) -> PyResult<()> {
    let value = value_from_pyobject(py, data)?;
    core_utils::write_private_json(Path::new(path), &value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn should_filter_log_line(line: &str) -> bool {
    core_utils::should_filter_log_line(line)
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
        .map_err(|e| map_core_err(py, e))?;

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
#[pyo3(signature = (backend, local_port, api_key=None, backend_url=None, env_api_key=None, verify_local=None, verify_dns=true, progress=false))]
fn tunnel_open(
    py: Python,
    backend: String,
    local_port: u16,
    api_key: Option<String>,
    backend_url: Option<String>,
    env_api_key: Option<String>,
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
    let verify_local = verify_local.unwrap_or(false);
    let result = RUNTIME.block_on(async move {
        tunnels::open_tunnel(
            backend_enum,
            local_port,
            api_key,
            backend_url,
            env_api_key,
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
            process_id: handle.process_id,
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
use std::path::PathBuf;

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
fn get_or_mint_api_key(py: Python, backend_url: Option<String>, allow_mint: bool) -> PyResult<String> {
    RUNTIME.block_on(async {
        auth::get_or_mint_api_key(backend_url.as_deref(), allow_mint).await
    }).map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (frontend_url=None))]
fn init_device_auth(py: Python, frontend_url: Option<String>) -> PyResult<PyObject> {
    let session = RUNTIME.block_on(async {
        auth::init_device_auth(frontend_url.as_deref()).await
    }).map_err(|e| map_core_err(py, e))?;
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
    }).map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &creds).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (backend_url=None, ttl_hours=None))]
fn mint_demo_key(py: Python, backend_url: Option<String>, ttl_hours: Option<u32>) -> PyResult<String> {
    RUNTIME.block_on(async {
        auth::mint_demo_key(backend_url.as_deref(), ttl_hours).await
    }).map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn mask_str(s: &str) -> String {
    auth::mask_str(s)
}

#[pyfunction]
fn auth_config_dir() -> String {
    auth::get_config_dir().to_string_lossy().to_string()
}

#[pyfunction]
fn auth_config_path() -> String {
    auth::get_config_path().to_string_lossy().to_string()
}

#[pyfunction]
#[pyo3(signature = (config_path=None))]
fn auth_load_credentials(py: Python, config_path: Option<String>) -> PyResult<PyObject> {
    let path = config_path.map(PathBuf::from);
    let creds = auth::load_credentials(path.as_deref())
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &creds)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (credentials, config_path=None))]
fn auth_store_credentials(py: Python, credentials: PyObject, config_path: Option<String>) -> PyResult<()> {
    let values: HashMap<String, String> = pythonize::depythonize(credentials.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let path = config_path.map(PathBuf::from);
    auth::store_credentials(&values, path.as_deref())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (credentials, config_path=None))]
fn auth_store_credentials_atomic(py: Python, credentials: PyObject, config_path: Option<String>) -> PyResult<()> {
    let values: HashMap<String, String> = pythonize::depythonize(credentials.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let path = config_path.map(PathBuf::from);
    auth::store_credentials_atomic(&values, path.as_deref())
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (override_env=true))]
fn auth_load_user_env(py: Python, override_env: bool) -> PyResult<PyObject> {
    let values = auth::load_user_env_with(override_env)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &values)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn auth_load_user_config(py: Python) -> PyResult<PyObject> {
    let values = auth::load_user_config()
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &values)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn auth_save_user_config(py: Python, config: PyObject) -> PyResult<()> {
    let values: HashMap<String, Value> = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    auth::save_user_config(&values)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn auth_update_user_config(py: Python, updates: PyObject) -> PyResult<PyObject> {
    let values: HashMap<String, Value> = pythonize::depythonize(updates.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let updated = auth::update_user_config(&values)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &updated)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// LocalAPI auth helpers

#[pyfunction]
fn mint_environment_api_key() -> String {
    core_localapi_auth::mint_environment_api_key()
}

#[pyfunction]
fn encrypt_for_backend(pubkey_b64: &str, secret: &str) -> PyResult<String> {
    core_localapi_auth::encrypt_for_backend(pubkey_b64, secret.as_bytes())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (backend_base, synth_api_key, token=None, timeout=15.0))]
fn setup_environment_api_key(
    py: Python,
    backend_base: String,
    synth_api_key: String,
    token: Option<String>,
    timeout: f64,
) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async move {
        core_localapi_auth::setup_environment_api_key(
            &backend_base,
            &synth_api_key,
            token.as_deref(),
            timeout,
        )
        .await
    });
    let value = result.map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (backend_base=None, synth_api_key=None, upload=true, persist=None))]
fn ensure_localapi_auth(
    py: Python,
    backend_base: Option<String>,
    synth_api_key: Option<String>,
    upload: bool,
    persist: Option<bool>,
) -> PyResult<String> {
    let result = RUNTIME.block_on(async move {
        core_localapi_auth::ensure_localapi_auth(
            backend_base.as_deref(),
            synth_api_key.as_deref(),
            upload,
            persist,
        )
        .await
    });
    result.map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn localapi_normalize_environment_api_key() -> Option<String> {
    core_localapi_auth::normalize_environment_api_key()
}

#[pyfunction]
fn localapi_allowed_environment_api_keys() -> Vec<String> {
    core_localapi_auth::allowed_environment_api_keys()
}

#[pyfunction]
fn localapi_is_api_key_header_authorized(header_values: Vec<String>) -> bool {
    core_localapi_auth::is_api_key_header_authorized(&header_values)
}

#[pyfunction]
fn localapi_normalize_chat_completion_url(url: String) -> String {
    core_localapi_helpers::normalize_chat_completion_url(&url)
}

#[pyfunction]
fn localapi_get_default_max_completion_tokens(model_name: String) -> i32 {
    core_localapi_helpers::get_default_max_completion_tokens(&model_name)
}

// LocalAPI trace helpers

#[pyfunction]
#[pyo3(signature = (inference_url=None))]
fn localapi_extract_trace_correlation_id(inference_url: Option<String>) -> Option<String> {
    core_localapi_trace::extract_trace_correlation_id(inference_url.as_deref())
}

#[pyfunction]
#[pyo3(signature = (trace_correlation_id=None, inference_url=None, fatal=false))]
fn localapi_validate_trace_correlation_id(
    py: Python,
    trace_correlation_id: Option<String>,
    inference_url: Option<String>,
    fatal: bool,
) -> PyResult<Option<String>> {
    core_localapi_trace::validate_trace_correlation_id(
        trace_correlation_id.as_deref(),
        inference_url.as_deref(),
        fatal,
    )
    .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (response_data, trace_correlation_id=None))]
fn localapi_include_trace_correlation_id_in_response(
    py: Python,
    response_data: PyObject,
    trace_correlation_id: Option<String>,
) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, response_data)?;
    let result = core_localapi_trace::include_trace_correlation_id_in_response(
        &value,
        trace_correlation_id.as_deref(),
    );
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (messages, response=None, *, correlation_id=None, session_id=None, metadata=None))]
fn localapi_build_trace_payload(
    py: Python,
    messages: PyObject,
    response: Option<PyObject>,
    correlation_id: Option<String>,
    session_id: Option<String>,
    metadata: Option<PyObject>,
) -> PyResult<PyObject> {
    let messages_value = value_from_pyobject(py, messages)?;
    let response_value = match response {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let metadata_value = match metadata {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let result = core_localapi_trace::build_trace_payload(
        &messages_value,
        response_value.as_ref(),
        correlation_id.as_deref(),
        session_id.as_deref(),
        metadata_value.as_ref(),
    );
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (messages, response=None, *, correlation_id=None, session_id=None, metadata=None))]
fn localapi_build_trajectory_trace(
    py: Python,
    messages: PyObject,
    response: Option<PyObject>,
    correlation_id: Option<String>,
    session_id: Option<String>,
    metadata: Option<PyObject>,
) -> PyResult<PyObject> {
    let messages_value = value_from_pyobject(py, messages)?;
    let response_value = match response {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let metadata_value = match metadata {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let result = core_localapi_trace::build_trajectory_trace(
        &messages_value,
        response_value.as_ref(),
        correlation_id.as_deref(),
        session_id.as_deref(),
        metadata_value.as_ref(),
    );
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (response_data, messages=None, response=None, *, run_id, correlation_id=None))]
fn localapi_include_event_history_in_response(
    py: Python,
    response_data: PyObject,
    messages: Option<PyObject>,
    response: Option<PyObject>,
    run_id: String,
    correlation_id: Option<String>,
) -> PyResult<PyObject> {
    let response_value = value_from_pyobject(py, response_data)?;
    let messages_value = match messages {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let response_payload = match response {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let result = core_localapi_trace::include_event_history_in_response(
        &response_value,
        messages_value.as_ref(),
        response_payload.as_ref(),
        &run_id,
        correlation_id.as_deref(),
    );
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (response_data, messages_by_trajectory=None, responses_by_trajectory=None, *, run_id, correlation_id=None))]
fn localapi_include_event_history_in_trajectories(
    py: Python,
    response_data: PyObject,
    messages_by_trajectory: Option<PyObject>,
    responses_by_trajectory: Option<PyObject>,
    run_id: String,
    correlation_id: Option<String>,
) -> PyResult<PyObject> {
    let response_value = value_from_pyobject(py, response_data)?;
    let messages_value = match messages_by_trajectory {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let responses_value = match responses_by_trajectory {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let result = core_localapi_trace::include_event_history_in_trajectories(
        &response_value,
        messages_value.as_ref(),
        responses_value.as_ref(),
        &run_id,
        correlation_id.as_deref(),
    );
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (response_data, expected_correlation_id=None))]
fn localapi_verify_trace_correlation_id_in_response(
    py: Python,
    response_data: PyObject,
    expected_correlation_id: Option<String>,
) -> PyResult<bool> {
    let value = value_from_pyobject(py, response_data)?;
    Ok(core_localapi_trace::verify_trace_correlation_id_in_response(
        &value,
        expected_correlation_id.as_deref(),
    ))
}

// LocalAPI validation helpers

#[pyfunction]
#[pyo3(signature = (artifact, max_bytes=None))]
fn localapi_validate_artifact_size(
    py: Python,
    artifact: PyObject,
    max_bytes: Option<usize>,
) -> PyResult<()> {
    let value = value_from_pyobject(py, artifact)?;
    let artifact: Artifact = serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let max_bytes = max_bytes.unwrap_or(core_localapi_validation::MAX_INLINE_ARTIFACT_BYTES);
    core_localapi_validation::validate_artifact_size(&artifact, max_bytes)
        .map_err(|e| map_core_err(py, e))?;
    Ok(())
}

#[pyfunction]
fn localapi_validate_artifacts_list(py: Python, artifacts: PyObject) -> PyResult<()> {
    let value = value_from_pyobject(py, artifacts)?;
    let artifacts: Vec<Artifact> = serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    core_localapi_validation::validate_artifacts_list(&artifacts)
        .map_err(|e| map_core_err(py, e))?;
    Ok(())
}

#[pyfunction]
fn localapi_validate_context_overrides(py: Python, overrides: PyObject) -> PyResult<()> {
    let value = value_from_pyobject(py, overrides)?;
    let overrides: Vec<ContextOverride> = serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    core_localapi_validation::validate_context_overrides(&overrides)
        .map_err(|e| map_core_err(py, e))?;
    Ok(())
}

#[pyfunction]
fn localapi_validate_context_snapshot(py: Python, snapshot: PyObject) -> PyResult<()> {
    let value = value_from_pyobject(py, snapshot)?;
    core_localapi_validation::validate_context_snapshot(&value)
        .map_err(|e| map_core_err(py, e))?;
    Ok(())
}

#[pyfunction]
fn localapi_to_jsonable(py: Python, value: PyObject) -> PyResult<PyObject> {
    let mut visited = HashSet::new();
    let bound = value.bind(py);
    to_jsonable_inner(py, &bound, &mut visited, 0)
}

#[pyfunction]
#[pyo3(signature = (headers, policy_config, default_env_keys=None))]
fn localapi_extract_api_key(
    py: Python,
    headers: PyObject,
    policy_config: PyObject,
    default_env_keys: Option<PyObject>,
) -> PyResult<Option<String>> {
    let headers_value = value_from_pyobject(py, headers)?;
    let headers_map: HashMap<String, String> = serde_json::from_value(headers_value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let policy_value = value_from_pyobject(py, policy_config)?;
    let default_map = match default_env_keys {
        Some(obj) => {
            let value = value_from_pyobject(py, obj)?;
            Some(
                serde_json::from_value::<HashMap<String, String>>(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        }
        None => None,
    };
    Ok(core_localapi_helpers::extract_api_key(
        &headers_map,
        &policy_value,
        default_map.as_ref(),
    ))
}

#[pyfunction]
#[pyo3(signature = (response_json, expected_tool_name=None))]
fn localapi_parse_tool_calls_from_response(
    py: Python,
    response_json: PyObject,
    expected_tool_name: Option<String>,
) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, response_json)?;
    let parsed = core_localapi_helpers::parse_tool_calls_from_response(
        &value,
        expected_tool_name.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    value_to_pyobject(py, &Value::Array(parsed))
}

#[pyfunction]
fn localapi_task_app_health(py: Python, task_app_url: String) -> PyResult<PyObject> {
    let result = RUNTIME.block_on(async { core_localapi_health::task_app_health(&task_app_url).await });
    match result {
        Ok(value) => value_to_pyobject(py, &value),
        Err(err) => Err(map_core_err(py, err)),
    }
}

#[pyfunction]
fn localapi_check_url_for_direct_provider_call(url: String) -> bool {
    core_localapi_llm_guards::is_direct_provider_call(&url)
}

#[pyfunction]
fn localapi_get_shared_http_client(py: Python) -> PyResult<PyObject> {
    let mut guard = LOCALAPI_HTTP_CLIENT.lock().unwrap();
    if let Some(client) = guard.as_ref() {
        return Ok(client.clone_ref(py));
    }

    let httpx = py.import_bound("httpx")?;
    let limits_kwargs = PyDict::new_bound(py);
    limits_kwargs.set_item("max_connections", 200)?;
    limits_kwargs.set_item("max_keepalive_connections", 200)?;
    limits_kwargs.set_item("keepalive_expiry", 30.0)?;
    let limits = httpx.getattr("Limits")?.call((), Some(&limits_kwargs))?;

    let timeout_kwargs = PyDict::new_bound(py);
    timeout_kwargs.set_item("connect", 30.0)?;
    timeout_kwargs.set_item("read", 300.0)?;
    timeout_kwargs.set_item("write", 30.0)?;
    timeout_kwargs.set_item("pool", 30.0)?;
    let timeout = httpx.getattr("Timeout")?.call((), Some(&timeout_kwargs))?;

    let client_kwargs = PyDict::new_bound(py);
    client_kwargs.set_item("limits", limits)?;
    client_kwargs.set_item("timeout", timeout)?;
    client_kwargs.set_item("follow_redirects", false)?;

    let client = httpx.getattr("AsyncClient")?.call((), Some(&client_kwargs))?;
    let client_obj = client.to_object(py);
    *guard = Some(client_obj.clone_ref(py));
    Ok(client_obj)
}

#[pyfunction]
fn localapi_reset_shared_http_client(py: Python) -> PyResult<()> {
    let mut guard = LOCALAPI_HTTP_CLIENT.lock().unwrap();
    if let Some(client) = guard.take() {
        if let Ok(asyncio) = py.import_bound("asyncio") {
            if let Ok(loop_obj) = asyncio.call_method0("get_running_loop") {
                if let Ok(aclose) = client.bind(py).getattr("aclose") {
                    if let Ok(coro) = aclose.call0() {
                        let _ = loop_obj.call_method1("create_task", (coro,));
                    }
                }
            }
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (default=false))]
fn localapi_tracing_env_enabled(default: bool) -> bool {
    core_localapi_tracing_utils::tracing_env_enabled(default)
}

#[pyfunction]
fn localapi_resolve_tracing_db_url() -> Option<String> {
    core_localapi_tracing_utils::resolve_tracing_db_url()
}

#[pyfunction]
fn localapi_resolve_sft_output_dir() -> Option<String> {
    core_localapi_tracing_utils::resolve_sft_output_dir()
}

#[pyfunction]
fn localapi_unique_sft_path(base_dir: String, run_id: String) -> Option<String> {
    core_localapi_tracing_utils::unique_sft_path(&base_dir, &run_id)
}

#[pyfunction]
#[pyo3(signature = (agent, global=false))]
fn localapi_get_agent_skills_path(agent: String, global: bool) -> String {
    core_localapi_override::get_agent_skills_path(&agent, global)
}

#[pyfunction]
#[pyo3(signature = (overrides, workspace_dir, allow_global=false, override_bundle_id=None))]
fn localapi_apply_context_overrides(
    py: Python,
    overrides: PyObject,
    workspace_dir: String,
    allow_global: bool,
    override_bundle_id: Option<String>,
) -> PyResult<Vec<ContextOverrideStatusPy>> {
    let value = value_from_pyobject(py, overrides)?;
    let overrides: Vec<ContextOverride> = match value {
        Value::Null => Vec::new(),
        other => serde_json::from_value(other)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    };
    let statuses = core_localapi_override::apply_context_overrides(
        &overrides,
        Path::new(&workspace_dir),
        allow_global,
        override_bundle_id.as_deref(),
    )
    .map_err(|e| map_core_err(py, e))?;
    Ok(statuses
        .into_iter()
        .map(|inner| ContextOverrideStatusPy { inner })
        .collect())
}

#[pyfunction]
fn localapi_get_applied_env_vars(py: Python, overrides: PyObject) -> PyResult<HashMap<String, String>> {
    let value = value_from_pyobject(py, overrides)?;
    let overrides: Vec<ContextOverride> = match value {
        Value::Null => Vec::new(),
        other => serde_json::from_value(other)
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
    };
    Ok(core_localapi_override::get_applied_env_vars(&overrides))
}

#[pyfunction]
#[pyo3(signature = (request, outcome_reward, inference_url=None, trace=None, policy_config=None, artifact=None, success_status=None, status_detail=None, reward_info=None))]
fn localapi_build_rollout_response(
    py: Python,
    request: PyObject,
    outcome_reward: f64,
    inference_url: Option<String>,
    trace: Option<PyObject>,
    policy_config: Option<PyObject>,
    artifact: Option<PyObject>,
    success_status: Option<PyObject>,
    status_detail: Option<String>,
    reward_info: Option<PyObject>,
) -> PyResult<PyObject> {
    let request_value = value_from_pyobject(py, request)?;
    let request: synth_ai_core::localapi::types::RolloutRequest =
        serde_json::from_value(request_value).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let trace_value = match trace {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let policy_value = match policy_config {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };
    let reward_value = match reward_info {
        Some(obj) => Some(value_from_pyobject(py, obj)?),
        None => None,
    };

    let artifact_value = match artifact {
        Some(obj) => {
            let value = value_from_pyobject(py, obj)?;
            Some(
                serde_json::from_value::<Vec<Artifact>>(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        }
        None => None,
    };

    let success_value = match success_status {
        Some(obj) => {
            let value = value_from_pyobject(py, obj)?;
            Some(
                serde_json::from_value::<SuccessStatus>(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        }
        None => None,
    };

    let result = core_localapi_rollout::build_rollout_response(
        &request,
        outcome_reward,
        inference_url.as_deref(),
        trace_value,
        policy_value.as_ref(),
        artifact_value,
        success_value,
        status_detail,
        reward_value.as_ref(),
    )
    .map_err(|e| map_core_err(py, e))?;

    value_to_pyobject(py, &serde_json::to_value(result).unwrap_or(Value::Null))
}

// LocalAPI proxy helpers

#[pyfunction]
#[pyo3(signature = (payload, model=None))]
fn localapi_prepare_for_openai(py: Python, payload: PyObject, model: Option<String>) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, payload)?;
    let result = core_localapi_proxy::prepare_for_openai(model.as_deref(), &value);
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (payload, model=None))]
fn localapi_prepare_for_groq(py: Python, payload: PyObject, model: Option<String>) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, payload)?;
    let result = core_localapi_proxy::prepare_for_groq(model.as_deref(), &value);
    value_to_pyobject(py, &result)
}

#[pyfunction]
#[pyo3(signature = (payload, model=None))]
fn localapi_normalize_response_format_for_groq(
    py: Python,
    payload: PyObject,
    model: Option<String>,
) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, payload)?;
    let mut map = match value {
        Value::Object(map) => map,
        _ => return value_to_pyobject(py, &value),
    };
    core_localapi_proxy::normalize_response_format_for_groq(model.as_deref(), &mut map);
    value_to_pyobject(py, &Value::Object(map))
}

#[pyfunction]
fn localapi_inject_system_hint(py: Python, payload: PyObject, hint: String) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, payload)?;
    let result = core_localapi_proxy::inject_system_hint(&value, &hint);
    value_to_pyobject(py, &result)
}

#[pyfunction]
fn localapi_extract_message_text(py: Python, message: PyObject) -> PyResult<String> {
    let value = value_from_pyobject(py, message)?;
    Ok(core_localapi_proxy::extract_message_text(&value))
}

#[pyfunction]
fn localapi_parse_tool_call_from_text(text: &str) -> (Vec<String>, String) {
    core_localapi_proxy::parse_tool_call_from_text(text)
}

#[pyfunction]
#[pyo3(signature = (payload, fallback_tool_name=None))]
fn localapi_synthesize_tool_call_if_missing(
    py: Python,
    payload: PyObject,
    fallback_tool_name: Option<String>,
) -> PyResult<PyObject> {
    let fallback_tool_name = fallback_tool_name.unwrap_or_else(|| "interact".to_string());
    let value = value_from_pyobject(py, payload)?;
    let result = core_localapi_proxy::synthesize_tool_call_if_missing(&value, &fallback_tool_name);
    value_to_pyobject(py, &result)
}

// LocalAPI validators

#[pyfunction]
fn localapi_validate_rollout_response_for_rl(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let value = value_from_pyobject(py, payload)?;
    let issues = core_localapi_validators::validate_rollout_response_for_rl(&value);
    pythonize::pythonize(py, &issues)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (url=None, default=None))]
fn localapi_normalize_inference_url(
    py: Python,
    url: Option<String>,
    default: Option<String>,
) -> PyResult<String> {
    let default = default.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());
    core_localapi_validators::normalize_inference_url(url.as_deref(), &default)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn localapi_validate_task_app_url(py: Python, url: String) -> PyResult<String> {
    core_localapi_validators::validate_task_app_url(&url)
        .map_err(|e| map_core_err(py, e))
}

// LocalAPI vendor keys

#[pyfunction]
fn localapi_normalize_vendor_keys(py: Python) -> PyResult<PyObject> {
    let result = core_localapi_vendors::normalize_vendor_keys();
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn localapi_get_openai_key() -> Option<String> {
    core_localapi_vendors::get_openai_key()
}

#[pyfunction]
fn localapi_get_groq_key() -> Option<String> {
    core_localapi_vendors::get_groq_key()
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
// HTTP Module - Core HTTP client
// =============================================================================

#[pyclass(name = "HttpClient")]
struct HttpClientPy {
    inner: RustHttpClient,
}

#[pymethods]
impl HttpClientPy {
    #[new]
    #[pyo3(signature = (base_url, api_key, timeout_secs=30))]
    fn new(py: Python, base_url: &str, api_key: &str, timeout_secs: u64) -> PyResult<Self> {
        let client = RustHttpClient::new(base_url, api_key, timeout_secs)
            .map_err(|e| map_core_err(py, CoreError::from(e)))?;
        Ok(Self { inner: client })
    }

    #[pyo3(signature = (path, params=None))]
    fn get_json(&self, py: Python, path: &str, params: Option<PyObject>) -> PyResult<PyObject> {
        let params_vec = parse_query_params(py, params)?;
        let params_refs: Vec<(&str, &str)> = params_vec
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let result: Value = RUNTIME.block_on(async {
            if params_refs.is_empty() {
                self.inner.get_json(path, None).await
            } else {
                self.inner.get_json(path, Some(&params_refs)).await
            }
        }).map_err(|e: RustHttpError| map_core_err(py, CoreError::from(e)))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path, body))]
    fn post_json(&self, py: Python, path: &str, body: PyObject) -> PyResult<PyObject> {
        let body_value: Value = pythonize::depythonize(body.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result: Value = RUNTIME.block_on(async {
            self.inner.post_json(path, &body_value).await
        }).map_err(|e: RustHttpError| map_core_err(py, CoreError::from(e)))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (path, data=None, files=None))]
    fn post_multipart(
        &self,
        py: Python,
        path: &str,
        data: Option<PyObject>,
        files: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let data_map = parse_form_data(py, data)?;
        let file_list = parse_multipart_files(py, files)?;
        let result: Value = RUNTIME.block_on(async {
            self.inner.post_multipart(path, &data_map, &file_list).await
        }).map_err(|e: RustHttpError| map_core_err(py, CoreError::from(e)))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn delete(&self, py: Python, path: &str) -> PyResult<()> {
        RUNTIME
            .block_on(async { self.inner.delete(path).await })
            .map_err(|e: RustHttpError| map_core_err(py, CoreError::from(e)))
    }
}

// =============================================================================
// SSE Module - Server-Sent Events streaming
// =============================================================================

#[pyclass(name = "SseEventIterator")]
struct SseEventIterator {
    inner: Arc<Mutex<Option<CoreSseStream>>>,
}

#[pymethods]
impl SseEventIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&self, py: Python) -> PyResult<Option<PyObject>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("sse stream lock poisoned"))?;
        let stream = match guard.as_mut() {
            Some(stream) => stream,
            None => return Ok(None),
        };

        loop {
            let next = py.allow_threads(|| {
                RUNTIME.block_on(async { stream.as_mut().next().await })
            });

            match next {
                Some(Ok(event)) => {
                    if event.data.trim() == "[DONE]" {
                        *guard = None;
                        return Ok(None);
                    }
                    let parsed: Result<Value, _> = serde_json::from_str(&event.data);
                    if let Ok(value) = parsed {
                        if value.is_object() {
                            return pythonize::pythonize(py, &value)
                                .map(|b| Some(b.unbind()))
                                .map_err(|e| PyValueError::new_err(e.to_string()));
                        }
                    }
                    continue;
                }
                Some(Err(err)) => {
                    return Err(map_core_err(py, err));
                }
                None => {
                    *guard = None;
                    return Ok(None);
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (url, headers=None, method="GET", json_payload=None, timeout=None))]
fn stream_sse_events(
    py: Python,
    url: String,
    headers: Option<PyObject>,
    method: &str,
    json_payload: Option<PyObject>,
    timeout: Option<f64>,
) -> PyResult<SseEventIterator> {
    let headers_map = parse_headers(py, headers)?;
    let method = Method::from_bytes(method.as_bytes())
        .map_err(|_| PyValueError::new_err("invalid HTTP method"))?;
    let payload_value = match json_payload {
        Some(obj) => {
            let value: Value = pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Some(value)
        }
        None => None,
    };
    let timeout = timeout.map(Duration::from_secs_f64);

    let stream = RUNTIME
        .block_on(async {
            core_stream_sse_request(url, method, headers_map, payload_value, timeout).await
        })
        .map_err(|e| map_core_err(py, e))?;

    Ok(SseEventIterator {
        inner: Arc::new(Mutex::new(Some(stream))),
    })
}

// =============================================================================
// Config Module - TOML parsing and config expansion
// =============================================================================

use synth_ai_core::config;

#[pyfunction]
fn parse_toml(py: Python, content: &str) -> PyResult<PyObject> {
    let value = config::parse_toml(content)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &value).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_toml(py: Python, path: &str) -> PyResult<PyObject> {
    let value = config::load_toml(Path::new(path))
        .map_err(|e| map_core_err(py, e))?;
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
#[pyo3(signature = (cli_value=None, env_value=None, config_value=None, default_value=None))]
fn resolve_config_value(
    py: Python,
    cli_value: Option<String>,
    env_value: Option<String>,
    config_value: Option<String>,
    default_value: Option<String>,
) -> PyResult<PyObject> {
    let resolved = config::resolve_config_value(
        cli_value.as_deref(),
        env_value.as_deref(),
        config_value.as_deref(),
        default_value.as_deref(),
    );
    let dict = PyDict::new_bound(py);
    dict.set_item("value", resolved.value)?;
    dict.set_item("cli_value", resolved.cli_value)?;
    dict.set_item("config_value", resolved.config_value)?;
    dict.set_item("cli_overrides_config", resolved.cli_overrides_config)?;
    Ok(dict.unbind().into())
}

#[pyfunction]
fn validate_overrides(py: Python, base: PyObject, overrides: PyObject) -> PyResult<()> {
    let base_value: serde_json::Value = pythonize::depythonize(base.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let overrides_value: serde_json::Value = pythonize::depythonize(overrides.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    config::validate_overrides(&base_value, &overrides_value, "")
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn resolve_seeds(py: Python, seeds: PyObject) -> PyResult<Vec<String>> {
    let seeds_value: serde_json::Value = pythonize::depythonize(seeds.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    config::resolve_seeds(&seeds_value).map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn split_train_validation(seeds: Vec<String>, train_ratio: f64) -> (Vec<String>, Vec<String>) {
    config::split_train_validation(&seeds, train_ratio)
}

#[pyfunction]
fn resolve_seed_spec(py: Python, seeds: PyObject) -> PyResult<Vec<i64>> {
    let seeds_value: serde_json::Value = pythonize::depythonize(seeds.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    config::resolve_seed_spec(&seeds_value).map_err(|e| map_core_err(py, e))
}

// =============================================================================
// Models - Identifier normalization
// =============================================================================

#[pyfunction]
#[pyo3(signature = (model, allow_finetuned_prefixes=false))]
fn normalize_model_identifier(py: Python, model: &str, allow_finetuned_prefixes: bool) -> PyResult<String> {
    core_models::normalize_model_identifier(model, allow_finetuned_prefixes)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn detect_model_provider(py: Python, model: &str) -> PyResult<String> {
    core_models::detect_model_provider(model)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn supported_models(py: Python) -> PyResult<PyObject> {
    let value = core_models::supported_models();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (minimal, defaults=None))]
fn expand_config(py: Python, minimal: PyObject, defaults: Option<PyObject>) -> PyResult<PyObject> {
    let minimal_value: serde_json::Value = pythonize::depythonize(minimal.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let defaults_value: Option<serde_json::Value> = defaults
        .map(|d| pythonize::depythonize(d.bind(py)))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let defaults_struct = if let Some(value) = defaults_value {
        serde_json::from_value::<config::OptimizationDefaults>(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        config::OptimizationDefaults::default()
    };
    let expanded = config::expand_config(&minimal_value, &defaults_struct)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &expanded).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn expand_eval_config(py: Python, minimal: PyObject) -> PyResult<PyObject> {
    let minimal_value: serde_json::Value = pythonize::depythonize(minimal.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let expanded = config::expand_eval_config(&minimal_value)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &expanded).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn expand_gepa_config(py: Python, minimal: PyObject) -> PyResult<PyObject> {
    let minimal_value: serde_json::Value = pythonize::depythonize(minimal.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let expanded = config::expand_gepa_config(&minimal_value)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &expanded).map(|b| b.unbind()).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn is_minimal_config(py: Python, config_value: PyObject) -> PyResult<bool> {
    let value: serde_json::Value = pythonize::depythonize(config_value.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(config::is_minimal_config(&value))
}

// =============================================================================
// Prompt Learning - Payload builder
// =============================================================================

#[pyfunction]
#[pyo3(signature = (config, task_url=None, overrides=None))]
fn build_prompt_learning_payload(
    py: Python,
    config: PyObject,
    task_url: Option<String>,
    overrides: Option<PyObject>,
) -> PyResult<PyObject> {
    let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let overrides_value: serde_json::Value = if let Some(obj) = overrides {
        pythonize::depythonize(obj.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        Value::Object(serde_json::Map::new())
    };

    let (payload, resolved_url) = core_build_prompt_learning_payload(
        &config_value,
        task_url,
        Some(&overrides_value),
    )
    .map_err(|e| map_core_err(py, e))?;

    let payload_py = pythonize::pythonize(py, &payload)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .unbind();
    Ok((payload_py, resolved_url).into_py(py))
}

#[pyfunction]
#[pyo3(signature = (config, config_path=None))]
fn validate_prompt_learning_config(
    py: Python,
    config: PyObject,
    config_path: Option<String>,
) -> PyResult<PyObject> {
    let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = core_validate_prompt_learning_config(&config_value, config_path.as_deref());

    let is_valid = result.is_valid();
    let mut out = serde_json::Map::new();
    out.insert(
        "errors".to_string(),
        Value::Array(result.errors.into_iter().map(Value::String).collect()),
    );
    out.insert(
        "warnings".to_string(),
        Value::Array(result.warnings.into_iter().map(Value::String).collect()),
    );
    out.insert(
        "info".to_string(),
        Value::Array(result.info.into_iter().map(Value::String).collect()),
    );
    out.insert("is_valid".to_string(), Value::Bool(is_valid));

    pythonize::pythonize(py, &Value::Object(out))
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (config))]
fn validate_prompt_learning_config_strict(py: Python, config: PyObject) -> PyResult<PyObject> {
    let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let errors = core_validate_prompt_learning_config_strict(&config_value);
    pythonize::pythonize(py, &errors)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (config, dataset))]
fn validate_graphgen_job_config(
    py: Python,
    config: PyObject,
    dataset: PyObject,
) -> PyResult<PyObject> {
    let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dataset_value: serde_json::Value = pythonize::depythonize(dataset.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = core_validate_graphgen_job_config(&config_value, &dataset_value);

    let mut out = serde_json::Map::new();
    out.insert("errors".to_string(), Value::Array(result.errors));
    out.insert(
        "warnings".to_string(),
        Value::Array(result.warnings.into_iter().map(Value::String).collect()),
    );

    pythonize::pythonize(py, &Value::Object(out))
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn graph_opt_supported_models(py: Python) -> PyResult<PyObject> {
    let value = core_graph_opt_supported_models();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn validate_graphgen_taskset(py: Python, dataset: PyObject) -> PyResult<PyObject> {
    let dataset_value: serde_json::Value = pythonize::depythonize(dataset.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let errors = core_validate_graphgen_taskset(&dataset_value);
    pythonize::pythonize(py, &errors)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn parse_graphgen_taskset(py: Python, dataset: PyObject) -> PyResult<PyObject> {
    let dataset_value: serde_json::Value = pythonize::depythonize(dataset.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed = core_parse_graphgen_taskset(&dataset_value)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_graphgen_taskset(py: Python, path: &str) -> PyResult<PyObject> {
    let parsed = core_load_graphgen_taskset(std::path::Path::new(path))
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (section, base_dir=None))]
fn validate_graph_job_section(
    py: Python,
    section: PyObject,
    base_dir: Option<String>,
) -> PyResult<PyObject> {
    let section_value: serde_json::Value = pythonize::depythonize(section.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let base_path = base_dir.as_deref().map(std::path::Path::new);
    let (result, errors) = core_validate_graph_job_section(&section_value, base_path);

    let mut out = serde_json::Map::new();
    out.insert(
        "errors".to_string(),
        Value::Array(errors),
    );
    if let Some(result) = result {
        out.insert("result".to_string(), result);
    } else {
        out.insert("result".to_string(), Value::Null);
    }
    pythonize::pythonize(py, &Value::Object(out))
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path))]
fn load_graph_job_toml(py: Python, path: &str) -> PyResult<PyObject> {
    let (result, errors) = core_load_graph_job_toml(std::path::Path::new(path));
    let mut out = serde_json::Map::new();
    out.insert("errors".to_string(), Value::Array(errors));
    if let Some(result) = result {
        out.insert("result".to_string(), result);
    } else {
        out.insert("result".to_string(), Value::Null);
    }
    pythonize::pythonize(py, &Value::Object(out))
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (payload))]
fn validate_graph_job_payload(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let errors = core_validate_graph_job_payload(&payload_value);
    pythonize::pythonize(py, &errors)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (source, dataset_name="converted_sft", detect_template=true, max_examples=None))]
fn convert_openai_sft(
    py: Python,
    source: PyObject,
    dataset_name: &str,
    detect_template: bool,
    max_examples: Option<usize>,
) -> PyResult<PyObject> {
    let source_value: serde_json::Value = pythonize::depythonize(source.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = core_convert_openai_sft(
        &source_value,
        Some(dataset_name.to_string()),
        detect_template,
        max_examples,
    )
    .map_err(|e| map_core_err(py, e))?;

    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Graph Evolve - Builders
// =============================================================================

#[pyfunction]
fn parse_graph_evolve_dataset(py: Python, dataset: PyObject) -> PyResult<PyObject> {
    let dataset_value: serde_json::Value = pythonize::depythonize(dataset.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed = core_parse_graph_evolve_dataset(&dataset_value)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_graph_evolve_dataset(py: Python, path: &str) -> PyResult<PyObject> {
    let parsed = core_load_graph_evolve_dataset(path)
        .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn normalize_graph_evolve_policy_models(py: Python, models: PyObject) -> PyResult<Vec<String>> {
    let models_value: Vec<String> = pythonize::depythonize(models.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    core_normalize_graph_evolve_policy_models(models_value)
        .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (policy_models, rollout_budget, proposer_effort, verifier_model=None, verifier_provider=None, population_size=4, num_generations=None, problem_spec=None, target_llm_calls=None, graph_type=None, initial_graph_id=None))]
fn build_graph_evolve_config(
    py: Python,
    policy_models: PyObject,
    rollout_budget: i64,
    proposer_effort: &str,
    verifier_model: Option<String>,
    verifier_provider: Option<String>,
    population_size: i64,
    num_generations: Option<i64>,
    problem_spec: Option<String>,
    target_llm_calls: Option<i64>,
    graph_type: Option<String>,
    initial_graph_id: Option<String>,
) -> PyResult<PyObject> {
    let policy_models_value: Vec<String> = pythonize::depythonize(policy_models.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let config = core_build_graph_evolve_config(
        policy_models_value,
        rollout_budget,
        proposer_effort,
        verifier_model,
        verifier_provider,
        population_size,
        num_generations,
        problem_spec,
        target_llm_calls,
        graph_type,
        initial_graph_id,
    )
    .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &config)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (dataset, config, metadata=None, auto_start=true))]
fn build_graph_evolve_payload(
    py: Python,
    dataset: PyObject,
    config: PyObject,
    metadata: Option<PyObject>,
    auto_start: bool,
) -> PyResult<PyObject> {
    let dataset_value: serde_json::Value = pythonize::depythonize(dataset.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let config_value: serde_json::Value = pythonize::depythonize(config.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let metadata_value: Option<serde_json::Value> = metadata
        .map(|m| pythonize::depythonize(m.bind(py)))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let payload = core_build_graph_evolve_payload(
        &dataset_value,
        &config_value,
        metadata_value.as_ref(),
        auto_start,
    )
    .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &payload)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (prompt_snapshot_id=None, graph_snapshot_id=None))]
fn resolve_graph_evolve_snapshot_id(
    py: Python,
    prompt_snapshot_id: Option<String>,
    graph_snapshot_id: Option<String>,
) -> PyResult<Option<String>> {
    core_resolve_graph_evolve_snapshot_id(
        prompt_snapshot_id.as_deref(),
        graph_snapshot_id.as_deref(),
    )
    .map_err(|e| map_core_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (job_id, prompt_snapshot_id=None, graph_snapshot_id=None))]
fn build_graph_evolve_graph_record_payload(
    py: Python,
    job_id: &str,
    prompt_snapshot_id: Option<String>,
    graph_snapshot_id: Option<String>,
) -> PyResult<PyObject> {
    let payload = core_build_graph_evolve_graph_record_payload(
        job_id,
        prompt_snapshot_id.as_deref(),
        graph_snapshot_id.as_deref(),
    )
    .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &payload)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (job_id, input_data, model=None, prompt_snapshot_id=None, graph_snapshot_id=None))]
fn build_graph_evolve_inference_payload(
    py: Python,
    job_id: &str,
    input_data: PyObject,
    model: Option<String>,
    prompt_snapshot_id: Option<String>,
    graph_snapshot_id: Option<String>,
) -> PyResult<PyObject> {
    let input_value: serde_json::Value = pythonize::depythonize(input_data.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let payload = core_build_graph_evolve_inference_payload(
        job_id,
        &input_value,
        model.as_deref(),
        prompt_snapshot_id.as_deref(),
        graph_snapshot_id.as_deref(),
    )
    .map_err(|e| map_core_err(py, e))?;
    pythonize::pythonize(py, &payload)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn build_graph_evolve_placeholder_dataset(py: Python) -> PyResult<PyObject> {
    let payload = core_build_graph_evolve_placeholder_dataset();
    pythonize::pythonize(py, &payload)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Graph Evolve - API
// =============================================================================

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, payload))]
fn graph_evolve_submit_job(
    py: Python,
    api_key: &str,
    backend_url: &str,
    payload: PyObject,
) -> PyResult<PyObject> {
    let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().submit_job(payload_value).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id))]
fn graph_evolve_get_status(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().get_status(job_id).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id))]
fn graph_evolve_start_job(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().start_job(job_id).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id, since_seq, limit))]
fn graph_evolve_get_events(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
    since_seq: i64,
    limit: i64,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().get_events(job_id, since_seq, limit).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id, query_string=""))]
fn graph_evolve_get_metrics(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
    query_string: &str,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().get_metrics(job_id, query_string).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id))]
fn graph_evolve_download_prompt(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().download_prompt(job_id).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id))]
fn graph_evolve_download_graph_txt(
    _py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
) -> PyResult<String> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    RUNTIME.block_on(async {
        client.graph_evolve().download_graph_txt(job_id).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, payload))]
fn graph_evolve_run_inference(
    py: Python,
    api_key: &str,
    backend_url: &str,
    payload: PyObject,
) -> PyResult<PyObject> {
    let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().run_inference(payload_value).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, payload))]
fn graph_evolve_get_graph_record(
    py: Python,
    api_key: &str,
    backend_url: &str,
    payload: PyObject,
) -> PyResult<PyObject> {
    let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().get_graph_record(payload_value).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id, payload=None))]
fn graph_evolve_cancel_job(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
    payload: Option<PyObject>,
) -> PyResult<PyObject> {
    let payload_value: serde_json::Value = if let Some(obj) = payload {
        pythonize::depythonize(obj.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    } else {
        Value::Object(serde_json::Map::new())
    };
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().cancel_job(job_id, payload_value).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (api_key, backend_url, job_id))]
fn graph_evolve_query_workflow_state(
    py: Python,
    api_key: &str,
    backend_url: &str,
    job_id: &str,
) -> PyResult<PyObject> {
    let client = RustSynthClient::new(api_key, Some(backend_url))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = RUNTIME.block_on(async {
        client.graph_evolve().query_workflow_state(job_id).await
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Graphs - Verifier request builder
// =============================================================================

#[pyfunction]
#[pyo3(signature = (rubric, trace_content=None, trace_ref=None, system_prompt=None, user_prompt=None, options=None, model=None, verifier_shape=None, rlm_impl=None))]
fn build_verifier_request(
    py: Python,
    rubric: PyObject,
    trace_content: Option<PyObject>,
    trace_ref: Option<String>,
    system_prompt: Option<String>,
    user_prompt: Option<String>,
    options: Option<PyObject>,
    model: Option<String>,
    verifier_shape: Option<String>,
    rlm_impl: Option<String>,
) -> PyResult<PyObject> {
    let trace_value: Option<Value> = trace_content
        .map(|t| pythonize::depythonize(t.bind(py)))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rubric_value: Value = pythonize::depythonize(rubric.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let options_value: Option<Value> = options
        .map(|o| pythonize::depythonize(o.bind(py)))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let req = core_build_verifier_request(
        trace_value,
        trace_ref,
        rubric_value,
        system_prompt,
        user_prompt,
        options_value,
        model,
        verifier_shape,
        rlm_impl,
    )
    .map_err(|e| map_core_err(py, e))?;

    pythonize::pythonize(py, &req)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (job_id=None, graph=None))]
fn resolve_graph_job_id(py: Python, job_id: Option<String>, graph: Option<PyObject>) -> PyResult<String> {
    let graph_value: Option<Value> = graph
        .map(|g| pythonize::depythonize(g.bind(py)))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    core_resolve_graph_job_id(job_id, graph_value).map_err(|e| map_core_err(py, e))
}

// =============================================================================
// Trace Upload - Presigned URL helpers
// =============================================================================

#[pyclass(name = "TraceUploadClient")]
struct TraceUploadClientPy {
    inner: RustTraceUploadClient,
}

#[pymethods]
impl TraceUploadClientPy {
    #[new]
    #[pyo3(signature = (base_url, api_key, timeout_secs=120))]
    fn new(py: Python, base_url: &str, api_key: &str, timeout_secs: u64) -> PyResult<Self> {
        let client = RustTraceUploadClient::new(base_url, api_key, timeout_secs)
            .map_err(|e| map_core_err(py, e))?;
        Ok(Self { inner: client })
    }

    #[staticmethod]
    fn trace_size(py: Python, trace: PyObject) -> PyResult<usize> {
        let trace_value: Value = pythonize::depythonize(trace.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        RustTraceUploadClient::trace_size(&trace_value)
            .map_err(|e| map_core_err(py, e))
    }

    #[staticmethod]
    fn should_upload(py: Python, trace: PyObject, threshold_bytes: usize) -> PyResult<bool> {
        let trace_value: Value = pythonize::depythonize(trace.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        RustTraceUploadClient::should_upload(&trace_value, threshold_bytes)
            .map_err(|e| map_core_err(py, e))
    }

    #[pyo3(signature = (content_type=None, expires_in_seconds=None))]
    fn create_upload_url(
        &self,
        py: Python,
        content_type: Option<&str>,
        expires_in_seconds: Option<i64>,
    ) -> PyResult<PyObject> {
        let result: RustUploadUrlResponse = RUNTIME.block_on(async {
            self.inner
                .create_upload_url(content_type, expires_in_seconds)
                .await
        })
        .map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (trace, content_type=None, expires_in_seconds=None))]
    fn upload_trace(
        &self,
        py: Python,
        trace: PyObject,
        content_type: Option<&str>,
        expires_in_seconds: Option<i64>,
    ) -> PyResult<String> {
        let trace_value: Value = pythonize::depythonize(trace.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        RUNTIME.block_on(async {
            self.inner
                .upload_trace(&trace_value, content_type, expires_in_seconds)
                .await
        })
        .map_err(|e| map_core_err(py, e))
    }

    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }
}

// =============================================================================
// Data Model Normalization (Rust-backed validation)
// =============================================================================

#[pyfunction]
fn normalize_rubric(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustRubric>(py, payload)
}

#[pyfunction]
fn normalize_judgement(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustJudgement>(py, payload)
}

#[pyfunction]
fn normalize_objective_spec(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustObjectiveSpec>(py, payload)
}

#[pyfunction]
fn normalize_reward_observation(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustRewardObservation>(py, payload)
}

#[pyfunction]
fn normalize_outcome_reward_record(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustOutcomeRewardRecord>(py, payload)
}

#[pyfunction]
fn normalize_event_reward_record(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustEventRewardRecord>(py, payload)
}

#[pyfunction]
fn normalize_context_override(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustContextOverride>(py, payload)
}

#[pyfunction]
fn normalize_artifact(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustArtifact>(py, payload)
}

#[pyfunction]
fn normalize_trace(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustSessionTrace>(py, payload)
}

#[pyfunction]
fn normalize_llm_call_record(py: Python, payload: PyObject) -> PyResult<PyObject> {
    normalize_value::<RustLLMCallRecord>(py, payload)
}

#[pyfunction]
fn data_enum_values(py: Python) -> PyResult<PyObject> {
    let value = core_data_enum_values();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// =============================================================================
// Data Model PyClasses (Rust-backed)
// =============================================================================

fn kwargs_to_value(_py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Value> {
    if let Some(dict) = kwargs {
        pythonize::depythonize(dict.as_any())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    } else {
        Ok(Value::Object(serde_json::Map::new()))
    }
}

fn value_from_pyobject(py: Python, obj: PyObject) -> PyResult<Value> {
    pythonize::depythonize(obj.bind(py)).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn value_to_pyobject(py: Python, value: &Value) -> PyResult<PyObject> {
    pythonize::pythonize(py, value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn to_jsonable_inner(
    py: Python,
    obj: &Bound<'_, PyAny>,
    visited: &mut HashSet<usize>,
    depth: usize,
) -> PyResult<PyObject> {
    if depth > 32 {
        let type_name = obj.get_type().name()?.to_string();
        let text = format!("<max_depth type={}>", type_name);
        return Ok(PyString::new_bound(py, &text).unbind().into());
    }

    if obj.is_none() {
        return Ok(py.None());
    }

    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBool>()
        || obj.is_instance_of::<PyFloat>()
        || obj.is_instance_of::<PyLong>()
    {
        return Ok(obj.to_object(py));
    }

    if obj.is_instance_of::<PyBytes>() || obj.is_instance_of::<PyByteArray>() {
        let len = obj.len().unwrap_or(0);
        let text = format!("<bytes len={}>", len);
        return Ok(PyString::new_bound(py, &text).unbind().into());
    }

    let obj_id = obj.as_ptr() as usize;
    if visited.contains(&obj_id) {
        let type_name = obj.get_type().name()?.to_string();
        let text = format!("<circular type={}>", type_name);
        return Ok(PyString::new_bound(py, &text).unbind().into());
    }

    // numpy handling (optional)
    if let Ok(numpy) = py.import_bound("numpy") {
        if let Ok(ndarray_type) = numpy.getattr("ndarray") {
            if obj.is_instance(&ndarray_type)? {
                let shape = obj.getattr("shape").ok().map(|v| v.to_object(py)).unwrap_or(py.None());
                let dtype = obj.getattr("dtype").ok().map(|v| v.to_object(py)).unwrap_or(py.None());
                let shape_repr = shape.bind(py).repr().unwrap_or_else(|_| PyString::new_bound(py, "<shape>").into());
                let dtype_repr = dtype.bind(py).repr().unwrap_or_else(|_| PyString::new_bound(py, "<dtype>").into());
                let text = format!(
                    "<ndarray shape={} dtype={}>",
                    shape_repr.to_string_lossy(),
                    dtype_repr.to_string_lossy()
                );
                return Ok(PyString::new_bound(py, &text).unbind().into());
            }
        }
        if let Ok(generic_type) = numpy.getattr("generic") {
            if obj.is_instance(&generic_type)? {
                if let Ok(item) = obj.call_method0("item") {
                    return to_jsonable_inner(py, &item, visited, depth + 1);
                }
            }
        }
    }

    // Enum handling
    if let Ok(enum_mod) = py.import_bound("enum") {
        if let Ok(enum_type) = enum_mod.getattr("Enum") {
            if obj.is_instance(&enum_type)? {
                if let Ok(value) = obj.getattr("value") {
                    return to_jsonable_inner(py, &value, visited, depth + 1);
                }
            }
        }
    }

    // Dataclass handling
    if let Ok(dataclasses) = py.import_bound("dataclasses") {
        if let Ok(is_dc) = dataclasses.call_method1("is_dataclass", (obj,)) {
            if is_dc.is_truthy()? {
                if let Ok(asdict) = dataclasses.call_method1("asdict", (obj,)) {
                    return to_jsonable_inner(py, &asdict, visited, depth + 1);
                }
            }
        }
    }

    // Pydantic / attrs style dumps
    for method in ["model_dump", "dict", "to_dict", "to_json"] {
        if obj.hasattr(method)? {
            let callable = obj.getattr(method)?;
            if callable.is_callable() {
                if let Ok(result) = callable.call0() {
                    return to_jsonable_inner(py, &result, visited, depth + 1);
                }
                let kwargs = PyDict::new_bound(py);
                let _ = kwargs.set_item("exclude_none", false);
                if let Ok(result) = callable.call((), Some(&kwargs)) {
                    return to_jsonable_inner(py, &result, visited, depth + 1);
                }
            }
        }
    }

    // Mapping handling
    if let Ok(dict_obj) = obj.downcast::<PyDict>() {
        visited.insert(obj_id);
        let dict = PyDict::new_bound(py);
        for (key, value) in dict_obj.iter() {
            let key_str = key.str()?.to_string_lossy().to_string();
            let value_obj = to_jsonable_inner(py, &value, visited, depth + 1)?;
            dict.set_item(key_str, value_obj)?;
        }
        visited.remove(&obj_id);
        return Ok(dict.unbind().into());
    }

    if obj.hasattr("items")? {
        if let Ok(items) = obj.call_method0("items") {
            visited.insert(obj_id);
            let dict = PyDict::new_bound(py);
            for item in items.iter()? {
                let pair = item?;
                if let Ok(tuple) = pair.downcast::<PyTuple>() {
                    if tuple.len() == 2 {
                        let key = tuple.get_item(0)?;
                        let value = tuple.get_item(1)?;
                        let key_str = key.str()?.to_string_lossy().to_string();
                        let value_obj = to_jsonable_inner(py, &value, visited, depth + 1)?;
                        dict.set_item(key_str, value_obj)?;
                    }
                }
            }
            visited.remove(&obj_id);
            return Ok(dict.unbind().into());
        }
    }

    // Sequence handling
    if obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>()
        || obj.is_instance_of::<PySet>()
    {
        visited.insert(obj_id);
        let mut items = Vec::new();
        for item in obj.iter()? {
            let item = item?;
            items.push(to_jsonable_inner(py, &item, visited, depth + 1)?);
        }
        visited.remove(&obj_id);
        let list = PyList::new_bound(py, items);
        return Ok(list.unbind().into());
    }

    // __dict__ fallback
    if obj.hasattr("__dict__")? {
        if let Ok(dict_obj) = obj.getattr("__dict__") {
            if dict_obj.is_instance_of::<PyDict>() {
                return to_jsonable_inner(py, &dict_obj, visited, depth + 1);
            }
        }
    }

    let repr = obj.repr()?.to_string_lossy().to_string();
    Ok(PyString::new_bound(py, &repr).unbind().into())
}

macro_rules! define_model_pyclass {
    ($struct_name:ident, $pyname:literal, $rust_type:ty) => {
        #[pyclass(name = $pyname)]
        #[derive(Clone)]
        struct $struct_name {
            inner: $rust_type,
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            #[pyo3(signature = (**kwargs))]
            fn new(py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                let value = kwargs_to_value(py, kwargs)?;
                let parsed: $rust_type = serde_json::from_value(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: parsed })
            }

            #[staticmethod]
            fn from_dict(py: Python, payload: PyObject) -> PyResult<Self> {
                let value = value_from_pyobject(py, payload)?;
                let parsed: $rust_type = serde_json::from_value(value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner: parsed })
            }

            fn to_dict(&self, py: Python) -> PyResult<PyObject> {
                let value = serde_json::to_value(&self.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                value_to_pyobject(py, &value)
            }

            fn model_dump(&self, py: Python) -> PyResult<PyObject> {
                self.to_dict(py)
            }

            fn dict(&self, py: Python) -> PyResult<PyObject> {
                self.to_dict(py)
            }

            fn __repr__(&self) -> PyResult<String> {
                let json = serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string());
                Ok(format!("{}({})", $pyname, json))
            }

            fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
                let value = serde_json::to_value(&self.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                if let Value::Object(map) = value {
                    if let Some(val) = map.get(name) {
                        return value_to_pyobject(py, val);
                    }
                    Err(PyAttributeError::new_err(format!(
                        "{} has no attribute '{}'",
                        $pyname, name
                    )))
                } else {
                    Err(PyAttributeError::new_err(format!(
                        "{} does not expose attributes",
                        $pyname
                    )))
                }
            }

            fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
                let value_json = value_from_pyobject(py, value)?;
                let current = serde_json::to_value(&self.inner)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let Value::Object(mut map) = current else {
                    return Err(PyAttributeError::new_err(format!(
                        "{} does not expose attributes",
                        $pyname
                    )));
                };
                if !map.contains_key(name) {
                    return Err(PyAttributeError::new_err(format!(
                        "{} has no attribute '{}'",
                        $pyname, name
                    )));
                }
                map.insert(name.to_string(), value_json);
                let parsed: $rust_type = serde_json::from_value(Value::Object(map))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                self.inner = parsed;
                Ok(())
            }
        }
    };
}

define_model_pyclass!(CriterionExamplePy, "CriterionExample", RustCriterionExample);
define_model_pyclass!(CriterionPy, "Criterion", RustCriterion);
define_model_pyclass!(RubricPy, "Rubric", RustRubric);
define_model_pyclass!(CriterionScoreDataPy, "CriterionScoreData", RustCriterionScoreData);
define_model_pyclass!(RubricAssignmentPy, "RubricAssignment", RustRubricAssignment);
define_model_pyclass!(JudgementPy, "Judgement", RustJudgement);
define_model_pyclass!(ObjectiveSpecPy, "ObjectiveSpec", RustObjectiveSpec);
define_model_pyclass!(RewardObservationPy, "RewardObservation", RustRewardObservation);
define_model_pyclass!(OutcomeObjectiveAssignmentPy, "OutcomeObjectiveAssignment", RustOutcomeObjectiveAssignment);
define_model_pyclass!(EventObjectiveAssignmentPy, "EventObjectiveAssignment", RustEventObjectiveAssignment);
define_model_pyclass!(InstanceObjectiveAssignmentPy, "InstanceObjectiveAssignment", RustInstanceObjectiveAssignment);
define_model_pyclass!(OutcomeRewardRecordPy, "OutcomeRewardRecord", RustOutcomeRewardRecord);
define_model_pyclass!(EventRewardRecordPy, "EventRewardRecord", RustEventRewardRecord);
define_model_pyclass!(RewardAggregatesPy, "RewardAggregates", RustRewardAggregates);
define_model_pyclass!(CalibrationExamplePy, "CalibrationExample", RustCalibrationExample);
define_model_pyclass!(GoldExamplePy, "GoldExample", RustGoldExample);
define_model_pyclass!(ContextOverridePy, "ContextOverride", RustContextOverride);
define_model_pyclass!(ContextOverrideStatusPy, "ContextOverrideStatus", RustContextOverrideStatus);
define_model_pyclass!(TaskDatasetSpecPy, "TaskDatasetSpec", RustTaskDatasetSpec);
define_model_pyclass!(MutationTypeStatsPy, "MutationTypeStats", RustMutationTypeStats);
define_model_pyclass!(MutationSummaryPy, "MutationSummary", RustMutationSummary);
define_model_pyclass!(SeedAnalysisPy, "SeedAnalysis", RustSeedAnalysis);
define_model_pyclass!(PhaseSummaryPy, "PhaseSummary", RustPhaseSummary);
define_model_pyclass!(StageInfoPy, "StageInfo", RustStageInfo);
define_model_pyclass!(SeedInfoPy, "SeedInfo", RustSeedInfo);
define_model_pyclass!(TokenUsagePy, "TokenUsage", RustSchemaTokenUsage);

#[pyclass(name = "ProgramCandidate")]
#[derive(Clone)]
struct ProgramCandidatePy {
    inner: RustProgramCandidate,
}

#[pymethods]
impl ProgramCandidatePy {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let value = kwargs_to_value(py, kwargs)?;
        let parsed: RustProgramCandidate = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    #[staticmethod]
    fn from_dict(py: Python, payload: PyObject) -> PyResult<Self> {
        let value = value_from_pyobject(py, payload)?;
        let parsed: RustProgramCandidate = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        value_to_pyobject(py, &value)
    }

    fn model_dump(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn dict(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    #[pyo3(signature = (max_length=500))]
    fn get_prompt_summary(&self, max_length: usize) -> PyResult<String> {
        Ok(self.inner.prompt_summary(max_length))
    }

    fn __repr__(&self) -> PyResult<String> {
        let json = serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string());
        Ok(format!("ProgramCandidate({})", json))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Value::Object(map) = value {
            if let Some(val) = map.get(name) {
                return value_to_pyobject(py, val);
            }
            Err(PyAttributeError::new_err(format!(
                "ProgramCandidate has no attribute '{}'",
                name
            )))
        } else {
            Err(PyAttributeError::new_err(
                "ProgramCandidate does not expose attributes",
            ))
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        let value_json = value_from_pyobject(py, value)?;
        let current = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let Value::Object(mut map) = current else {
            return Err(PyAttributeError::new_err(
                "ProgramCandidate does not expose attributes",
            ));
        };
        if !map.contains_key(name) {
            return Err(PyAttributeError::new_err(format!(
                "ProgramCandidate has no attribute '{}'",
                name
            )));
        }
        map.insert(name.to_string(), value_json);
        let parsed: RustProgramCandidate = serde_json::from_value(Value::Object(map))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = parsed;
        Ok(())
    }
}

#[pyclass(name = "BaseJobEvent")]
#[derive(Clone)]
struct BaseJobEventPy {
    inner: RustBaseJobEvent,
}

#[pymethods]
impl BaseJobEventPy {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let value = kwargs_to_value(py, kwargs)?;
        let parsed: RustBaseJobEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    #[staticmethod]
    fn from_dict(py: Python, payload: PyObject) -> PyResult<Self> {
        let value = value_from_pyobject(py, payload)?;
        let parsed: RustBaseJobEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        value_to_pyobject(py, &self.inner.to_dict_value())
    }

    fn model_dump(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn dict(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        let json = serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string());
        Ok(format!("BaseJobEvent({})", json))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Value::Object(map) = value {
            if let Some(val) = map.get(name) {
                return value_to_pyobject(py, val);
            }
            Err(PyAttributeError::new_err(format!(
                "BaseJobEvent has no attribute '{}'",
                name
            )))
        } else {
            Err(PyAttributeError::new_err("BaseJobEvent does not expose attributes"))
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        let value_json = value_from_pyobject(py, value)?;
        let current = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let Value::Object(mut map) = current else {
            return Err(PyAttributeError::new_err(
                "BaseJobEvent does not expose attributes",
            ));
        };
        if !map.contains_key(name) {
            return Err(PyAttributeError::new_err(format!(
                "BaseJobEvent has no attribute '{}'",
                name
            )));
        }
        map.insert(name.to_string(), value_json);
        let parsed: RustBaseJobEvent = serde_json::from_value(Value::Object(map))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = parsed;
        Ok(())
    }
}

#[pyclass(name = "JobEvent")]
#[derive(Clone)]
struct JobEventPy {
    inner: RustJobEvent,
}

#[pymethods]
impl JobEventPy {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let value = kwargs_to_value(py, kwargs)?;
        let parsed: RustJobEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    #[staticmethod]
    fn from_dict(py: Python, payload: PyObject) -> PyResult<Self> {
        let value = value_from_pyobject(py, payload)?;
        let parsed: RustJobEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        value_to_pyobject(py, &self.inner.to_dict_value())
    }

    fn model_dump(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn dict(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        let json = serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string());
        Ok(format!("JobEvent({})", json))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Value::Object(map) = value {
            if let Some(val) = map.get(name) {
                return value_to_pyobject(py, val);
            }
            Err(PyAttributeError::new_err(format!("JobEvent has no attribute '{}'", name)))
        } else {
            Err(PyAttributeError::new_err("JobEvent does not expose attributes"))
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        let value_json = value_from_pyobject(py, value)?;
        let current = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let Value::Object(mut map) = current else {
            return Err(PyAttributeError::new_err("JobEvent does not expose attributes"));
        };
        if !map.contains_key(name) {
            return Err(PyAttributeError::new_err(format!(
                "JobEvent has no attribute '{}'",
                name
            )));
        }
        map.insert(name.to_string(), value_json);
        let parsed: RustJobEvent = serde_json::from_value(Value::Object(map))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = parsed;
        Ok(())
    }
}

#[pyclass(name = "CandidateEvent")]
#[derive(Clone)]
struct CandidateEventPy {
    inner: RustCandidateEvent,
}

#[pymethods]
impl CandidateEventPy {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let value = kwargs_to_value(py, kwargs)?;
        let parsed: RustCandidateEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    #[staticmethod]
    fn from_dict(py: Python, payload: PyObject) -> PyResult<Self> {
        let value = value_from_pyobject(py, payload)?;
        let parsed: RustCandidateEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: parsed })
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        value_to_pyobject(py, &self.inner.to_dict_value())
    }

    fn model_dump(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn dict(&self, py: Python) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        let json = serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string());
        Ok(format!("CandidateEvent({})", json))
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Value::Object(map) = value {
            if let Some(val) = map.get(name) {
                return value_to_pyobject(py, val);
            }
            Err(PyAttributeError::new_err(format!(
                "CandidateEvent has no attribute '{}'",
                name
            )))
        } else {
            Err(PyAttributeError::new_err(
                "CandidateEvent does not expose attributes",
            ))
        }
    }

    fn __setattr__(&mut self, py: Python, name: &str, value: PyObject) -> PyResult<()> {
        let value_json = value_from_pyobject(py, value)?;
        let current = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let Value::Object(mut map) = current else {
            return Err(PyAttributeError::new_err(
                "CandidateEvent does not expose attributes",
            ));
        };
        if !map.contains_key(name) {
            return Err(PyAttributeError::new_err(format!(
                "CandidateEvent has no attribute '{}'",
                name
            )));
        }
        map.insert(name.to_string(), value_json);
        let parsed: RustCandidateEvent = serde_json::from_value(Value::Object(map))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner = parsed;
        Ok(())
    }
}
define_model_pyclass!(ArtifactPy, "Artifact", RustArtifact);
define_model_pyclass!(ArtifactBundlePy, "ArtifactBundle", RustArtifactBundle);

define_model_pyclass!(TimeRecordPy, "TimeRecord", RustTimeRecord);
define_model_pyclass!(MessageContentPy, "SessionMessageContent", RustMessageContent);
define_model_pyclass!(BaseEventFieldsPy, "BaseEventFields", RustBaseEventFields);
define_model_pyclass!(LMCAISEventPy, "LMCAISEvent", RustLMCAISEvent);
define_model_pyclass!(EnvironmentEventPy, "EnvironmentEvent", RustEnvironmentEvent);
define_model_pyclass!(RuntimeEventPy, "RuntimeEvent", RustRuntimeEvent);
define_model_pyclass!(TracingEventPy, "TracingEvent", RustTracingEvent);
define_model_pyclass!(MarkovBlanketMessagePy, "SessionEventMarkovBlanketMessage", RustMarkovBlanketMessage);
define_model_pyclass!(SessionTimeStepPy, "SessionTimeStep", RustSessionTimeStep);
define_model_pyclass!(SessionTracePy, "SessionTrace", RustSessionTrace);

define_model_pyclass!(LLMUsagePy, "LLMUsage", RustLLMUsage);
define_model_pyclass!(LLMRequestParamsPy, "LLMRequestParams", RustLLMRequestParams);
define_model_pyclass!(LLMContentPartPy, "LLMContentPart", RustLLMContentPart);
define_model_pyclass!(LLMMessagePy, "LLMMessage", RustLLMMessage);
define_model_pyclass!(ToolCallSpecPy, "ToolCallSpec", RustToolCallSpec);
define_model_pyclass!(ToolCallResultPy, "ToolCallResult", RustToolCallResult);
define_model_pyclass!(LLMChunkPy, "LLMChunk", RustLLMChunk);
define_model_pyclass!(LLMCallRecordPy, "LLMCallRecord", RustLLMCallRecord);

define_model_pyclass!(LeaseInfoPy, "LeaseInfo", RustLeaseInfo);
define_model_pyclass!(ConnectorStatusPy, "ConnectorStatus", RustConnectorStatus);
define_model_pyclass!(GatewayStatusPy, "GatewayStatus", RustGatewayStatus);
define_model_pyclass!(DiagnosticsPy, "Diagnostics", RustDiagnostics);
define_model_pyclass!(TunnelHandleModelPy, "TunnelHandle", RustTunnelHandle);

// =============================================================================
// Orchestration Events - Parsing helpers
// =============================================================================

#[pyfunction]
fn parse_orchestration_event(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed = RustEventParser::parse(&value);
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn parse_optimization_event(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed = RustEventParser::parse(&value);

    let mut out = serde_json::Map::new();
    out.insert("event_type".to_string(), Value::String(parsed.event_type.clone()));
    out.insert(
        "category".to_string(),
        Value::String(parsed.category.as_str().to_string()),
    );
    out.insert("data".to_string(), parsed.data.clone());
    if let Some(seq) = parsed.seq {
        out.insert("seq".to_string(), Value::Number(seq.into()));
    }
    if let Some(ts) = parsed.timestamp_ms {
        out.insert("timestamp_ms".to_string(), Value::Number(ts.into()));
    }

    match parsed.category {
        synth_ai_core::orchestration::events::EventCategory::Baseline => {
            let detail = RustEventParser::parse_baseline(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Candidate => {
            let detail = RustEventParser::parse_candidate(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Frontier => {
            let detail = RustEventParser::parse_frontier(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Progress => {
            let detail = RustEventParser::parse_progress(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Generation => {
            let detail = RustEventParser::parse_generation(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Complete => {
            let detail = RustEventParser::parse_complete(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Termination => {
            let detail = RustEventParser::parse_termination(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        synth_ai_core::orchestration::events::EventCategory::Usage => {
            let detail = RustEventParser::parse_usage(&parsed);
            merge_json_map(&mut out, serde_json::to_value(detail).unwrap_or(Value::Null));
        }
        _ => {}
    }

    pythonize::pythonize(py, &Value::Object(out))
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (payload, job_id=None))]
fn parse_job_event(py: Python, payload: PyObject, job_id: Option<String>) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let parsed = core_parse_job_event(&value, job_id.as_deref());
    pythonize::pythonize(py, &parsed)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn validate_base_event(py: Python, payload: PyObject) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(payload.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = core_validate_base_event(&value);
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (seed, score=None))]
fn orchestration_seed_score_entry(
    py: Python,
    seed: i64,
    score: Option<PyObject>,
) -> PyResult<PyObject> {
    let score_value: Option<Value> = match score {
        Some(obj) => Some(
            pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    let value = core_seed_score_entry(seed, score_value.as_ref());
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (candidate, require_stages=false, candidate_id=None))]
fn orchestration_extract_stages_from_candidate(
    py: Python,
    candidate: PyObject,
    require_stages: bool,
    candidate_id: Option<String>,
) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(candidate.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = core_extract_stages_from_candidate(&value, require_stages, candidate_id.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match result {
        Some(stages) => {
            let stages_value = serde_json::to_value(stages)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            pythonize::pythonize(py, &stages_value)
                .map(|b| b.unbind())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn orchestration_extract_program_candidate_content(py: Python, candidate: PyObject) -> PyResult<String> {
    let value: Value = pythonize::depythonize(candidate.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(core_extract_program_candidate_content(&value))
}

#[pyfunction]
fn orchestration_normalize_transformation(py: Python, transformation: PyObject) -> PyResult<PyObject> {
    let value: Value = pythonize::depythonize(transformation.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    match core_normalize_transformation(&value) {
        Some(result) => pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string())),
        None => Ok(py.None()),
    }
}

#[pyfunction]
#[pyo3(signature = (candidate, candidate_id=None, seed_info=None, token_usage=None, cost_usd=None, timestamp_ms=None))]
fn orchestration_build_program_candidate(
    py: Python,
    candidate: PyObject,
    candidate_id: Option<String>,
    seed_info: Option<PyObject>,
    token_usage: Option<PyObject>,
    cost_usd: Option<f64>,
    timestamp_ms: Option<i64>,
) -> PyResult<PyObject> {
    let candidate_value: Value = pythonize::depythonize(candidate.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let seed_info_value: Option<Value> = match seed_info {
        Some(obj) => Some(
            pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    let token_usage_value: Option<Value> = match token_usage {
        Some(obj) => Some(
            pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ),
        None => None,
    };
    let result = core_build_program_candidate(
        &candidate_value,
        candidate_id.as_deref(),
        seed_info_value.as_ref(),
        token_usage_value.as_ref(),
        cost_usd,
        timestamp_ms,
    );
    pythonize::pythonize(py, &result)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn orchestration_max_instruction_length() -> usize {
    RUST_MAX_INSTRUCTION_LENGTH
}

#[pyfunction]
fn orchestration_max_rollout_samples() -> usize {
    RUST_MAX_ROLLOUT_SAMPLES
}

#[pyfunction]
fn orchestration_max_seed_info_count() -> usize {
    RUST_MAX_SEED_INFO_COUNT
}

#[pyfunction]
fn orchestration_event_enum_values(py: Python) -> PyResult<PyObject> {
    let value = core_event_enum_values();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn orchestration_is_valid_event_type(event_type: &str) -> bool {
    core_is_valid_event_type(event_type)
}

#[pyfunction]
fn orchestration_validate_event_type(py: Python, event_type: &str) -> PyResult<String> {
    core_validate_event_type(event_type).map_err(|e| map_core_err(py, e))
}

#[pyfunction]
fn orchestration_base_event_schemas(py: Python) -> PyResult<PyObject> {
    let value = core_base_event_schemas();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn orchestration_base_job_event_schema(py: Python) -> PyResult<PyObject> {
    let value = core_base_job_event_schema();
    pythonize::pythonize(py, &value)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn orchestration_get_base_schema(py: Python, name: &str) -> PyResult<PyObject> {
    match core_get_base_schema(name) {
        Some(value) => pythonize::pythonize(py, &value)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string())),
        None => Ok(py.None()),
    }
}

#[pyfunction]
fn orchestration_merge_event_schema(
    py: Python,
    base: PyObject,
    extension: PyObject,
    algorithm: &str,
    event_type: &str,
) -> PyResult<PyObject> {
    let base_value: Value = pythonize::depythonize(base.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let extension_value: Value = pythonize::depythonize(extension.bind(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let merged = core_merge_event_schema(&base_value, &extension_value, algorithm, event_type);
    pythonize::pythonize(py, &merged)
        .map(|b| b.unbind())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "ProgressTracker")]
struct ProgressTrackerPy {
    inner: Arc<Mutex<RustProgressTracker>>,
}

fn progress_tracker_state(tracker: &RustProgressTracker) -> Result<Value, String> {
    let mut map = serde_json::Map::new();
    map.insert(
        "progress".to_string(),
        serde_json::to_value(&tracker.progress).map_err(|e| e.to_string())?,
    );
    map.insert(
        "candidates".to_string(),
        serde_json::to_value(&tracker.candidates).map_err(|e| e.to_string())?,
    );
    map.insert(
        "baseline".to_string(),
        serde_json::to_value(&tracker.baseline).map_err(|e| e.to_string())?,
    );
    map.insert(
        "frontier".to_string(),
        serde_json::to_value(&tracker.frontier).map_err(|e| e.to_string())?,
    );
    map.insert(
        "frontier_history".to_string(),
        serde_json::to_value(&tracker.frontier_history).map_err(|e| e.to_string())?,
    );
    map.insert(
        "generation_history".to_string(),
        serde_json::to_value(&tracker.generation_history).map_err(|e| e.to_string())?,
    );
    map.insert(
        "last_seq".to_string(),
        serde_json::to_value(&tracker.last_seq).map_err(|e| e.to_string())?,
    );
    Ok(Value::Object(map))
}

#[pymethods]
impl ProgressTrackerPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RustProgressTracker::new())),
        }
    }

    fn update(&self, py: Python, payload: PyObject) -> PyResult<PyObject> {
        let value: Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let parsed = RustEventParser::parse(&value);
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        guard.update(&parsed);
        let state = progress_tracker_state(&guard).map_err(PyValueError::new_err)?;
        pythonize::pythonize(py, &state)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn state(&self, py: Python) -> PyResult<PyObject> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyValueError::new_err("progress tracker lock poisoned"))?;
        let state = progress_tracker_state(&guard).map_err(PyValueError::new_err)?;
        pythonize::pythonize(py, &state)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// API Client - SynthClient
// =============================================================================

use synth_ai_core::api::{
    SynthClient as RustSynthClient,
    GepaJobRequest, MiproJobRequest, EvalJobRequest,
    GraphCompletionRequest, VerifierOptions,
    build_verifier_request as core_build_verifier_request,
    resolve_graph_job_id as core_resolve_graph_job_id,
};
use synth_ai_core::orchestration::{
    PromptLearningJob as RustPromptLearningJob,
    GraphEvolveJob as RustGraphEvolveJob,
    ProgressTracker as RustProgressTracker,
    parse_job_event as core_parse_job_event,
    validate_base_event as core_validate_base_event,
    build_prompt_learning_payload as core_build_prompt_learning_payload,
    validate_prompt_learning_config as core_validate_prompt_learning_config,
    validate_prompt_learning_config_strict as core_validate_prompt_learning_config_strict,
    validate_graphgen_job_config as core_validate_graphgen_job_config,
    graph_opt_supported_models as core_graph_opt_supported_models,
    validate_graphgen_taskset as core_validate_graphgen_taskset,
    parse_graphgen_taskset as core_parse_graphgen_taskset,
    load_graphgen_taskset as core_load_graphgen_taskset,
    validate_graph_job_section as core_validate_graph_job_section,
    load_graph_job_toml as core_load_graph_job_toml,
    validate_graph_job_payload as core_validate_graph_job_payload,
    convert_openai_sft as core_convert_openai_sft,
    seed_score_entry as core_seed_score_entry,
    extract_stages_from_candidate as core_extract_stages_from_candidate,
    extract_program_candidate_content as core_extract_program_candidate_content,
    normalize_transformation as core_normalize_transformation,
    build_program_candidate as core_build_program_candidate,
    event_enum_values as core_event_enum_values,
    is_valid_event_type as core_is_valid_event_type,
    validate_event_type as core_validate_event_type,
    base_event_schemas as core_base_event_schemas,
    base_job_event_schema as core_base_job_event_schema,
    get_base_schema as core_get_base_schema,
    merge_event_schema as core_merge_event_schema,
    parse_graph_evolve_dataset as core_parse_graph_evolve_dataset,
    load_graph_evolve_dataset as core_load_graph_evolve_dataset,
    normalize_graph_evolve_policy_models as core_normalize_graph_evolve_policy_models,
    build_graph_evolve_config as core_build_graph_evolve_config,
    build_graph_evolve_payload as core_build_graph_evolve_payload,
    resolve_graph_evolve_snapshot_id as core_resolve_graph_evolve_snapshot_id,
    build_graph_evolve_graph_record_payload as core_build_graph_evolve_graph_record_payload,
    build_graph_evolve_inference_payload as core_build_graph_evolve_inference_payload,
    build_graph_evolve_placeholder_dataset as core_build_graph_evolve_placeholder_dataset,
};
use synth_ai_core::orchestration::events::EventParser as RustEventParser;

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

    #[pyo3(signature = (job_id, reason=None))]
    fn cancel_eval(&self, py: Python, job_id: &str, reason: Option<&str>) -> PyResult<PyObject> {
        let reason_owned = reason.map(|s| s.to_string());
        let result = RUNTIME.block_on(async {
            self.inner.eval().cancel(job_id, reason_owned).await
        }).map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -------------------------------------------------------------------------
    // Inference API
    // -------------------------------------------------------------------------

    #[pyo3(signature = (request))]
    fn inference_chat_completion(&self, py: Python, request: PyObject) -> PyResult<PyObject> {
        let body: Value = pythonize::depythonize(request.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid request: {}", e)))?;
        let result = RUNTIME
            .block_on(async { self.inner.inference().chat_completion(body).await })
            .map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn get_eval_results(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().get_results(job_id).await
        }).map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (job_id))]
    fn download_eval_traces(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let bytes = RUNTIME.block_on(async {
            self.inner.eval().download_traces(job_id).await
        }).map_err(|e| map_core_err(py, e))?;
        Ok(PyBytes::new_bound(py, &bytes).into_py(py))
    }

    #[pyo3(signature = (job_id))]
    fn query_eval_workflow_state(&self, py: Python, job_id: &str) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.eval().query_workflow_state(job_id).await
        }).map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -------------------------------------------------------------------------
    // Graphs API
    // -------------------------------------------------------------------------

    #[pyo3(signature = (kind=None, limit=None))]
    fn list_graphs(&self, py: Python, kind: Option<&str>, limit: Option<i32>) -> PyResult<PyObject> {
        let result = RUNTIME.block_on(async {
            self.inner.graphs().list_graphs(kind, limit).await
        }).map_err(|e| map_core_err(py, e))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

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
// Orchestration - GraphEvolveJob
// =============================================================================

#[pyclass]
struct GraphEvolveJob {
    inner: std::sync::Mutex<RustGraphEvolveJob>,
}

#[pymethods]
impl GraphEvolveJob {
    #[staticmethod]
    #[pyo3(signature = (payload, api_key=None, base_url=None))]
    fn from_payload(py: Python, payload: PyObject, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = RustGraphEvolveJob::from_payload(payload_value, api_key, base_url)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[staticmethod]
    #[pyo3(signature = (job_id, api_key=None, base_url=None))]
    fn from_job_id(job_id: &str, api_key: Option<&str>, base_url: Option<&str>) -> PyResult<Self> {
        let job = RustGraphEvolveJob::from_job_id(job_id, api_key, base_url)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: std::sync::Mutex::new(job) })
    }

    #[getter]
    fn job_id(&self) -> Option<String> {
        self.inner.lock().unwrap().job_id().map(|s| s.to_string())
    }

    #[getter]
    fn legacy_graphgen_job_id(&self) -> Option<String> {
        self.inner.lock().unwrap().legacy_job_id().map(|s| s.to_string())
    }

    fn submit(&self, py: Python) -> PyResult<PyObject> {
        let mut job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.submit().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_status(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.get_status().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn start(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.start().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (since_seq=0, limit=1000))]
    fn get_events(&self, py: Python, since_seq: i64, limit: i64) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.get_events(since_seq, limit).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (query_string=""))]
    fn get_metrics(&self, py: Python, query_string: &str) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.get_metrics(query_string).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn download_prompt(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.download_prompt().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn download_graph_txt(&self) -> PyResult<String> {
        let job = self.inner.lock().unwrap();
        RUNTIME.block_on(async { job.download_graph_txt().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (payload))]
    fn run_inference(&self, py: Python, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.run_inference(payload_value).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (payload))]
    fn get_graph_record(&self, py: Python, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: serde_json::Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?;
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.get_graph_record(payload_value).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (payload=None))]
    fn cancel(&self, py: Python, payload: Option<PyObject>) -> PyResult<PyObject> {
        let payload_value: serde_json::Value = if let Some(obj) = payload {
            pythonize::depythonize(obj.bind(py))
                .map_err(|e| PyValueError::new_err(format!("invalid payload: {}", e)))?
        } else {
            Value::Object(serde_json::Map::new())
        };
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.cancel(payload_value).await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn query_workflow_state(&self, py: Python) -> PyResult<PyObject> {
        let job = self.inner.lock().unwrap();
        let result = RUNTIME.block_on(async { job.query_workflow_state().await })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (model_name=None))]
fn tracing_detect_provider(_py: Python, model_name: Option<String>) -> PyResult<String> {
    Ok(synth_ai_core::tracing::detect_provider(model_name.as_deref()))
}

#[pyfunction]
fn tracing_calculate_cost(
    _py: Python,
    model_name: &str,
    input_tokens: i64,
    output_tokens: i64,
) -> PyResult<Option<f64>> {
    Ok(synth_ai_core::tracing::calculate_cost(
        model_name,
        input_tokens,
        output_tokens,
    ))
}

// =============================================================================
// Tracing - SessionTracer + Libsql storage
// =============================================================================

fn py_to_map(py: Python, obj: Option<PyObject>) -> PyResult<HashMap<String, Value>> {
    if let Some(val) = obj {
        pythonize::depythonize(val.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    } else {
        Ok(HashMap::new())
    }
}

fn parse_tracing_query_params(py: Python, obj: Option<PyObject>) -> PyResult<RustQueryParams> {
    let Some(val) = obj else {
        return Ok(RustQueryParams::None);
    };
    if let Ok(map) = pythonize::depythonize::<HashMap<String, Value>>(val.bind(py)) {
        let named = map
            .into_iter()
            .map(|(key, value)| {
                let name = if key.starts_with(':') || key.starts_with('@') || key.starts_with('$') {
                    key
                } else {
                    format!(":{key}")
                };
                (name, value)
            })
            .collect();
        return Ok(RustQueryParams::Named(named));
    }
    if let Ok(values) = pythonize::depythonize::<Vec<Value>>(val.bind(py)) {
        return Ok(RustQueryParams::Positional(values));
    }
    Err(PyValueError::new_err(
        "query params must be a dict of named params or a list of positional params",
    ))
}

fn coerce_tracing_event(value: Value) -> Result<RustTracingEvent, String> {
    let mut value = value;
    if value.get("event_type").is_none() {
        let event_type = if value.get("model_name").is_some()
            || value.get("call_records").is_some()
            || value.get("input_tokens").is_some()
            || value.get("output_tokens").is_some()
        {
            "cais"
        } else if value.get("reward").is_some()
            || value.get("terminated").is_some()
            || value.get("truncated").is_some()
        {
            "environment"
        } else if value.get("actions").is_some() {
            "runtime"
        } else {
            return Err("unable to infer tracing event_type".to_string());
        };
        if let Some(obj) = value.as_object_mut() {
            obj.insert("event_type".to_string(), Value::String(event_type.to_string()));
        }
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

#[pyclass(name = "LibsqlTraceStorage")]
#[derive(Clone)]
struct LibsqlTraceStoragePy {
    inner: Arc<LibsqlTraceStorage>,
}

#[pymethods]
impl LibsqlTraceStoragePy {
    #[staticmethod]
    fn memory() -> PyResult<Self> {
        let storage = RUNTIME
            .block_on(LibsqlTraceStorage::new_memory())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(storage),
        })
    }

    #[staticmethod]
    fn file(path: &str) -> PyResult<Self> {
        let storage = RUNTIME
            .block_on(LibsqlTraceStorage::new_file(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(storage),
        })
    }

    #[staticmethod]
    fn turso(url: &str, auth_token: &str) -> PyResult<Self> {
        let config = RustStorageConfig::turso(url, auth_token);
        let storage = RUNTIME
            .block_on(LibsqlTraceStorage::new(config))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(storage),
        })
    }

    fn initialize(&self) -> PyResult<()> {
        RUNTIME
            .block_on(self.inner.initialize())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn close(&self) -> PyResult<()> {
        RUNTIME
            .block_on(self.inner.close())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (sql, params=None))]
    fn query(&self, py: Python, sql: &str, params: Option<PyObject>) -> PyResult<PyObject> {
        let params_value = parse_tracing_query_params(py, params)?;
        let result = RUNTIME
            .block_on(self.inner.query(sql, params_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "SessionTracer")]
struct SessionTracerPy {
    inner: Arc<RustSessionTracer>,
}

#[pymethods]
impl SessionTracerPy {
    #[new]
    #[pyo3(signature = (storage=None, auto_save=None))]
    fn new(storage: Option<LibsqlTraceStoragePy>, auto_save: Option<bool>) -> PyResult<Self> {
        let storage = match storage {
            Some(s) => s.inner.clone(),
            None => Arc::new(
                RUNTIME
                    .block_on(LibsqlTraceStorage::new_memory())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
        };
        let mut tracer = RustSessionTracer::new(storage);
        if let Some(enabled) = auto_save {
            tracer.set_auto_save(enabled);
        }
        Ok(Self {
            inner: Arc::new(tracer),
        })
    }

    #[staticmethod]
    fn memory() -> PyResult<Self> {
        Self::new(None, None)
    }

    #[staticmethod]
    fn file(path: &str) -> PyResult<Self> {
        let storage = LibsqlTraceStoragePy::file(path)?;
        Self::new(Some(storage), None)
    }

    #[staticmethod]
    fn turso(url: &str, auth_token: &str) -> PyResult<Self> {
        let storage = LibsqlTraceStoragePy::turso(url, auth_token)?;
        Self::new(Some(storage), None)
    }

    #[pyo3(signature = (session_id=None, metadata=None))]
    fn start_session(&self, py: Python, session_id: Option<&str>, metadata: Option<PyObject>) -> PyResult<String> {
        let metadata_value = py_to_map(py, metadata)?;
        RUNTIME
            .block_on(self.inner.start_session(session_id, metadata_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (step_id, turn_number=None, metadata=None))]
    fn start_timestep(
        &self,
        py: Python,
        step_id: &str,
        turn_number: Option<i32>,
        metadata: Option<PyObject>,
    ) -> PyResult<()> {
        let metadata_value = py_to_map(py, metadata)?;
        RUNTIME
            .block_on(self.inner.start_timestep(step_id, turn_number, metadata_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn end_timestep(&self) -> PyResult<()> {
        RUNTIME
            .block_on(self.inner.end_timestep())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (event))]
    fn record_event(&self, py: Python, event: PyObject) -> PyResult<Option<i64>> {
        let value: Value = pythonize::depythonize(event.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let event = coerce_tracing_event(value).map_err(PyValueError::new_err)?;
        RUNTIME
            .block_on(self.inner.record_event(event))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (message))]
    fn record_message(&self, py: Python, message: PyObject) -> PyResult<Option<i64>> {
        let value: Value = pythonize::depythonize(message.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let message: RustMarkovBlanketMessage =
            serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        RUNTIME
            .block_on(self.inner.record_message(
                message.content,
                &message.message_type,
                message.metadata,
            ))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (reward))]
    fn record_outcome_reward(&self, py: Python, reward: PyObject) -> PyResult<Option<i64>> {
        let value: Value = pythonize::depythonize(reward.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reward: RustOutcomeReward =
            serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        RUNTIME
            .block_on(self.inner.record_outcome_reward(reward))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (event_id, reward, message_id=None, turn_number=None))]
    fn record_event_reward(
        &self,
        py: Python,
        event_id: i64,
        reward: PyObject,
        message_id: Option<i64>,
        turn_number: Option<i32>,
    ) -> PyResult<Option<i64>> {
        let value: Value = pythonize::depythonize(reward.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reward: RustEventReward =
            serde_json::from_value(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let _ = (message_id, turn_number);
        RUNTIME
            .block_on(self.inner.record_event_reward(event_id, reward))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (save=true))]
    fn end_session(&self, py: Python, save: bool) -> PyResult<PyObject> {
        let trace: RustSessionTrace = RUNTIME
            .block_on(self.inner.end_session(save))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &trace)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn current_session_id(&self) -> PyResult<Option<String>> {
        Ok(RUNTIME.block_on(self.inner.current_session_id()))
    }

    fn current_step_id(&self) -> PyResult<Option<String>> {
        Ok(RUNTIME.block_on(self.inner.current_step_id()))
    }

    #[pyo3(signature = (session_id))]
    fn get_session(&self, py: Python, session_id: &str) -> PyResult<PyObject> {
        let session = RUNTIME
            .block_on(self.inner.get_session(session_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &session)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (session_id))]
    fn delete_session(&self, session_id: &str) -> PyResult<bool> {
        RUNTIME
            .block_on(self.inner.delete_session(session_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (sql, params=None))]
    fn query(&self, py: Python, sql: &str, params: Option<PyObject>) -> PyResult<PyObject> {
        let params_value = parse_tracing_query_params(py, params)?;
        let result = RUNTIME
            .block_on(self.inner.query(sql, params_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// Streaming - StreamConfig / StreamEndpoints / JobStreamer
// =============================================================================

fn parse_stream_type(name: &str) -> Option<RustStreamType> {
    match name {
        "status" => Some(RustStreamType::Status),
        "events" => Some(RustStreamType::Events),
        "metrics" => Some(RustStreamType::Metrics),
        "timeline" => Some(RustStreamType::Timeline),
        _ => None,
    }
}

#[pyclass(name = "StreamConfig")]
#[derive(Clone)]
struct StreamConfigPy {
    inner: RustStreamConfig,
}

#[pymethods]
impl StreamConfigPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustStreamConfig::default(),
        }
    }

    #[staticmethod]
    fn all() -> Self {
        Self {
            inner: RustStreamConfig::all(),
        }
    }

    #[staticmethod]
    fn minimal() -> Self {
        Self {
            inner: RustStreamConfig::minimal(),
        }
    }

    #[staticmethod]
    fn errors_only() -> Self {
        Self {
            inner: RustStreamConfig::errors_only(),
        }
    }

    #[staticmethod]
    fn metrics_only() -> Self {
        Self {
            inner: RustStreamConfig::metrics_only(),
        }
    }

    fn enable_stream(&mut self, stream_type: &str) -> PyResult<()> {
        let parsed = parse_stream_type(stream_type)
            .ok_or_else(|| PyValueError::new_err("unknown stream type"))?;
        self.inner = self.inner.clone().enable_stream(parsed);
        Ok(())
    }

    fn disable_stream(&mut self, stream_type: &str) -> PyResult<()> {
        let parsed = parse_stream_type(stream_type)
            .ok_or_else(|| PyValueError::new_err("unknown stream type"))?;
        self.inner = self.inner.clone().disable_stream(parsed);
        Ok(())
    }

    fn include_event_type(&mut self, event_type: &str) {
        self.inner = self.inner.clone().include_event_type(event_type);
    }

    fn exclude_event_type(&mut self, event_type: &str) {
        self.inner = self.inner.clone().exclude_event_type(event_type);
    }

    fn with_levels(&mut self, levels: Vec<String>) {
        let level_refs: Vec<&str> = levels.iter().map(|s| s.as_str()).collect();
        self.inner = self.inner.clone().with_levels(level_refs);
    }

    fn with_interval(&mut self, seconds: f64) {
        self.inner = self.inner.clone().with_interval(seconds);
    }

    fn with_sample_rate(&mut self, rate: f64) {
        self.inner = self.inner.clone().with_sample_rate(rate);
    }

    fn without_deduplication(&mut self) {
        self.inner = self.inner.clone().without_deduplication();
    }

    #[pyo3(signature = (max_events=None))]
    fn set_max_events_per_poll(&mut self, max_events: Option<usize>) {
        self.inner.max_events_per_poll = max_events;
    }
}

#[pyclass(name = "StreamEndpoints")]
#[derive(Clone)]
struct StreamEndpointsPy {
    inner: RustStreamEndpoints,
}

#[pymethods]
impl StreamEndpointsPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustStreamEndpoints::default(),
        }
    }

    #[staticmethod]
    fn learning(job_id: &str) -> Self {
        Self {
            inner: RustStreamEndpoints::learning(job_id),
        }
    }

    #[staticmethod]
    fn prompt_learning(job_id: &str) -> Self {
        Self {
            inner: RustStreamEndpoints::prompt_learning(job_id),
        }
    }

    #[staticmethod]
    fn eval(job_id: &str) -> Self {
        Self {
            inner: RustStreamEndpoints::eval(job_id),
        }
    }

    #[staticmethod]
    fn sft(job_id: &str) -> Self {
        Self {
            inner: RustStreamEndpoints::sft(job_id),
        }
    }

    #[staticmethod]
    fn graph_optimization(job_id: &str) -> Self {
        Self {
            inner: RustStreamEndpoints::graph_optimization(job_id),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (status=None, events=None, metrics=None, timeline=None))]
    fn custom(
        status: Option<String>,
        events: Option<String>,
        metrics: Option<String>,
        timeline: Option<String>,
    ) -> Self {
        Self {
            inner: RustStreamEndpoints::custom(status, events, metrics, timeline),
        }
    }

    fn with_status_fallback(&mut self, endpoint: &str) {
        self.inner = self.inner.clone().with_status_fallback(endpoint);
    }

    fn with_event_fallback(&mut self, endpoint: &str) {
        self.inner = self.inner.clone().with_event_fallback(endpoint);
    }

    fn events_stream_url(&self) -> Option<String> {
        self.inner.events_stream_url()
    }

    #[getter]
    fn status(&self) -> Option<String> {
        self.inner.status.clone()
    }

    #[getter]
    fn events(&self) -> Option<String> {
        self.inner.events.clone()
    }

    #[getter]
    fn metrics(&self) -> Option<String> {
        self.inner.metrics.clone()
    }

    #[getter]
    fn timeline(&self) -> Option<String> {
        self.inner.timeline.clone()
    }
}

#[pyclass(name = "JobStreamer")]
struct JobStreamerPy {
    inner: Mutex<RustJobStreamer>,
}

#[pymethods]
impl JobStreamerPy {
    #[new]
    #[pyo3(signature = (base_url, api_key, job_id, endpoints=None, config=None))]
    fn new(
        base_url: String,
        api_key: String,
        job_id: String,
        endpoints: Option<StreamEndpointsPy>,
        config: Option<StreamConfigPy>,
    ) -> Self {
        let mut streamer = RustJobStreamer::new(base_url, api_key, job_id);
        if let Some(e) = endpoints {
            streamer = streamer.with_endpoints(e.inner);
        }
        if let Some(c) = config {
            streamer = streamer.with_config(c.inner);
        }
        Self {
            inner: Mutex::new(streamer),
        }
    }

    fn poll_status(&self, py: Python) -> PyResult<PyObject> {
        let mut streamer = self.inner.lock().unwrap();
        let status = RUNTIME
            .block_on(streamer.poll_status())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &status)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn poll_events(&self, py: Python) -> PyResult<PyObject> {
        let mut streamer = self.inner.lock().unwrap();
        let events = RUNTIME
            .block_on(streamer.poll_events())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &events)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn poll_metrics(&self, py: Python) -> PyResult<PyObject> {
        let mut streamer = self.inner.lock().unwrap();
        let metrics = RUNTIME
            .block_on(streamer.poll_metrics())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &metrics)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn stream_until_terminal(&self, py: Python) -> PyResult<PyObject> {
        let mut streamer = self.inner.lock().unwrap();
        let status = RUNTIME
            .block_on(streamer.stream_until_terminal())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &status)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// LocalAPI - TaskAppClient / EnvClient
// =============================================================================

#[pyclass(name = "TaskAppClient")]
#[derive(Clone)]
struct TaskAppClientPy {
    inner: Arc<RustTaskAppClient>,
}

#[pyclass(name = "EnvClient")]
#[derive(Clone)]
struct EnvClientPy {
    inner: Arc<RustTaskAppClient>,
}

#[pymethods]
impl TaskAppClientPy {
    #[new]
    #[pyo3(signature = (base_url, api_key=None, timeout_secs=None))]
    fn new(base_url: &str, api_key: Option<&str>, timeout_secs: Option<u64>) -> Self {
        let client = match timeout_secs {
            Some(timeout) => RustTaskAppClient::with_timeout(base_url, api_key, timeout),
            None => RustTaskAppClient::new(base_url, api_key),
        };
        Self {
            inner: Arc::new(client),
        }
    }

    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    fn health(&self, py: Python) -> PyResult<PyObject> {
        let response = RUNTIME
            .block_on(self.inner.health())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &response)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn info(&self, py: Python) -> PyResult<PyObject> {
        let response = RUNTIME
            .block_on(self.inner.info())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &response)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (seeds=None))]
    fn task_info(&self, py: Python, seeds: Option<Vec<i64>>) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(self.inner.task_info(seeds.as_deref()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn taskset_info(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(self.inner.taskset_info())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rollout(&self, py: Python, request: PyObject) -> PyResult<PyObject> {
        let req: synth_ai_core::localapi::RolloutRequest =
            pythonize::depythonize(request.bind(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(self.inner.rollout(&req))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn done(&self, py: Python) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(self.inner.done())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get(&self, py: Python, path: &str) -> PyResult<PyObject> {
        let result = RUNTIME
            .block_on(self.inner.get(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn post(&self, py: Python, path: &str, body: PyObject) -> PyResult<PyObject> {
        let payload: Value = pythonize::depythonize(body.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = RUNTIME
            .block_on(self.inner.post(path, &payload))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn wait_for_healthy(&self, timeout_seconds: f64, poll_interval_seconds: f64) -> PyResult<()> {
        RUNTIME
            .block_on(self.inner.wait_for_healthy(timeout_seconds, poll_interval_seconds))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn env(&self) -> EnvClientPy {
        EnvClientPy {
            inner: self.inner.clone(),
        }
    }
}

#[pymethods]
impl EnvClientPy {
    fn initialize(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let env = RustEnvClient::new(&self.inner);
        let result = RUNTIME
            .block_on(env.initialize(env_name, &payload_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn step(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let env = RustEnvClient::new(&self.inner);
        let result = RUNTIME
            .block_on(env.step(env_name, &payload_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn terminate(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let env = RustEnvClient::new(&self.inner);
        let result = RUNTIME
            .block_on(env.terminate(env_name, &payload_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn reset(&self, py: Python, env_name: &str, payload: PyObject) -> PyResult<PyObject> {
        let payload_value: Value = pythonize::depythonize(payload.bind(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let env = RustEnvClient::new(&self.inner);
        let result = RUNTIME
            .block_on(env.reset(env_name, &payload_value))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        pythonize::pythonize(py, &result)
            .map(|b| b.unbind())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// =============================================================================
// LocalAPI - Dataset Registry
// =============================================================================

#[pyclass(name = "TaskDatasetRegistry")]
struct TaskDatasetRegistryPy {
    entries: HashMap<String, (RustTaskDatasetSpec, PyObject, bool)>,
    cache: HashMap<(String, Option<String>, Option<String>), PyObject>,
}

#[pymethods]
impl TaskDatasetRegistryPy {
    #[new]
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    #[pyo3(signature = (spec, loader, cache=true))]
    fn register(&mut self, py: Python, spec: PyObject, loader: PyObject, cache: bool) -> PyResult<()> {
        let value = value_from_pyobject(py, spec)?;
        let parsed: RustTaskDatasetSpec = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        parsed.validate().map_err(|e| map_core_err(py, e))?;
        let key = parsed.id.clone();
        self.entries.insert(key, (parsed, loader, cache));
        Ok(())
    }

    fn describe(&self, dataset_id: String) -> PyResult<TaskDatasetSpecPy> {
        let (spec, _, _) = self
            .entries
            .get(&dataset_id)
            .ok_or_else(|| PyValueError::new_err(format!("Dataset not registered: {dataset_id}")))?;
        Ok(TaskDatasetSpecPy { inner: spec.clone() })
    }

    fn list(&self) -> PyResult<Vec<TaskDatasetSpecPy>> {
        Ok(self
            .entries
            .values()
            .map(|(spec, _, _)| TaskDatasetSpecPy { inner: spec.clone() })
            .collect())
    }

    fn get(&mut self, py: Python, spec: PyObject) -> PyResult<PyObject> {
        let bound = spec.bind(py);
        let (effective_spec, loader, cache_enabled) = if let Ok(dataset_id) = bound.extract::<String>() {
            let (base, loader, cache_enabled) = self
                .entries
                .get(&dataset_id)
                .ok_or_else(|| PyValueError::new_err(format!("Dataset not registered: {dataset_id}")))?;
            (base.clone(), loader.clone_ref(py), *cache_enabled)
        } else {
            let value = value_from_pyobject(py, spec)?;
            let parsed: RustTaskDatasetSpec = serde_json::from_value(value)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let (base, loader, cache_enabled) = self
                .entries
                .get(&parsed.id)
                .ok_or_else(|| PyValueError::new_err(format!("Dataset not registered: {}", parsed.id)))?;
            (base.merge_with(&parsed), loader.clone_ref(py), *cache_enabled)
        };

        let cache_key = (
            effective_spec.id.clone(),
            effective_spec.version.clone(),
            effective_spec.default_split.clone(),
        );
        if cache_enabled {
            if let Some(cached) = self.cache.get(&cache_key) {
                return Ok(cached.clone_ref(py));
            }
        }

        let spec_obj = Py::new(py, TaskDatasetSpecPy { inner: effective_spec })?;
        let result = loader.bind(py).call1((spec_obj,))?.to_object(py);

        if cache_enabled {
            self.cache.insert(cache_key, result.clone_ref(py));
        }

        Ok(result)
    }

    #[staticmethod]
    #[pyo3(signature = (spec, split=None))]
    fn ensure_split(py: Python, spec: PyObject, split: Option<String>) -> PyResult<String> {
        let value = value_from_pyobject(py, spec)?;
        let parsed: RustTaskDatasetSpec = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        core_localapi_datasets::ensure_split(&parsed, split.as_deref())
            .map_err(|e| map_core_err(py, e))
    }

    #[staticmethod]
    #[pyo3(signature = (seed, cardinality=None))]
    fn normalise_seed(seed: i64, cardinality: Option<i64>) -> i64 {
        core_localapi_datasets::normalise_seed(seed, cardinality)
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
    m.add_function(wrap_pyfunction!(backend_url_base, m)?)?;
    m.add_function(wrap_pyfunction!(backend_url_api, m)?)?;
    m.add_function(wrap_pyfunction!(backend_url_synth_research_base, m)?)?;
    m.add_function(wrap_pyfunction!(backend_url_synth_research_openai, m)?)?;
    m.add_function(wrap_pyfunction!(backend_url_synth_research_anthropic, m)?)?;
    m.add_function(wrap_pyfunction!(frontend_url_base, m)?)?;
    m.add_function(wrap_pyfunction!(join_url, m)?)?;
    m.add_function(wrap_pyfunction!(local_backend_url, m)?)?;
    m.add_function(wrap_pyfunction!(backend_health_url, m)?)?;
    m.add_function(wrap_pyfunction!(backend_me_url, m)?)?;
    m.add_function(wrap_pyfunction!(backend_demo_keys_url, m)?)?;
    // Utils
    m.add_function(wrap_pyfunction!(strip_json_comments, m)?)?;
    m.add_function(wrap_pyfunction!(create_and_write_json, m)?)?;
    m.add_function(wrap_pyfunction!(load_json_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(deep_update, m)?)?;
    m.add_function(wrap_pyfunction!(repo_root, m)?)?;
    m.add_function(wrap_pyfunction!(synth_home_dir, m)?)?;
    m.add_function(wrap_pyfunction!(synth_user_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(synth_localapi_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(synth_bin_dir, m)?)?;
    m.add_function(wrap_pyfunction!(is_file_type, m)?)?;
    m.add_function(wrap_pyfunction!(validate_file_type, m)?)?;
    m.add_function(wrap_pyfunction!(is_hidden_path, m)?)?;
    m.add_function(wrap_pyfunction!(get_bin_path, m)?)?;
    m.add_function(wrap_pyfunction!(get_home_config_file_paths, m)?)?;
    m.add_function(wrap_pyfunction!(find_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(compute_import_paths, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup_paths, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_private_dir, m)?)?;
    m.add_function(wrap_pyfunction!(write_private_text, m)?)?;
    m.add_function(wrap_pyfunction!(write_private_json, m)?)?;
    m.add_function(wrap_pyfunction!(should_filter_log_line, m)?)?;

    // Events
    m.add_function(wrap_pyfunction!(poll_events, m)?)?;

    // HTTP
    m.add_class::<HttpClientPy>()?;
    m.add_class::<SseEventIterator>()?;
    m.add_function(wrap_pyfunction!(stream_sse_events, m)?)?;

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
    m.add_function(wrap_pyfunction!(mint_demo_key, m)?)?;
    m.add_function(wrap_pyfunction!(mask_str, m)?)?;
    m.add_function(wrap_pyfunction!(auth_config_dir, m)?)?;
    m.add_function(wrap_pyfunction!(auth_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(auth_load_credentials, m)?)?;
    m.add_function(wrap_pyfunction!(auth_store_credentials, m)?)?;
    m.add_function(wrap_pyfunction!(auth_store_credentials_atomic, m)?)?;
    m.add_function(wrap_pyfunction!(auth_load_user_env, m)?)?;
    m.add_function(wrap_pyfunction!(auth_load_user_config, m)?)?;
    m.add_function(wrap_pyfunction!(auth_save_user_config, m)?)?;
    m.add_function(wrap_pyfunction!(auth_update_user_config, m)?)?;
    m.add_function(wrap_pyfunction!(mint_environment_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(encrypt_for_backend, m)?)?;
    m.add_function(wrap_pyfunction!(setup_environment_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_localapi_auth, m)?)?;

    // LocalAPI (NEW)
    m.add_function(wrap_pyfunction!(localapi_normalize_environment_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_allowed_environment_api_keys, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_is_api_key_header_authorized, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_normalize_chat_completion_url, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_default_max_completion_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_extract_trace_correlation_id, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_trace_correlation_id, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_include_trace_correlation_id_in_response, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_build_trace_payload, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_build_trajectory_trace, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_include_event_history_in_response, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_include_event_history_in_trajectories, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_verify_trace_correlation_id_in_response, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_artifact_size, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_artifacts_list, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_context_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_context_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_to_jsonable, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_extract_api_key, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_parse_tool_calls_from_response, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_task_app_health, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_check_url_for_direct_provider_call, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_shared_http_client, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_reset_shared_http_client, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_tracing_env_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_resolve_tracing_db_url, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_resolve_sft_output_dir, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_unique_sft_path, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_agent_skills_path, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_apply_context_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_applied_env_vars, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_build_rollout_response, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_prepare_for_openai, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_prepare_for_groq, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_normalize_response_format_for_groq, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_inject_system_hint, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_extract_message_text, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_parse_tool_call_from_text, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_synthesize_tool_call_if_missing, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_rollout_response_for_rl, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_normalize_inference_url, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_validate_task_app_url, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_normalize_vendor_keys, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_openai_key, m)?)?;
    m.add_function(wrap_pyfunction!(localapi_get_groq_key, m)?)?;

    // Polling (NEW)
    m.add_function(wrap_pyfunction!(calculate_backoff, m)?)?;

    // Config (NEW)
    m.add_function(wrap_pyfunction!(parse_toml, m)?)?;
    m.add_function(wrap_pyfunction!(load_toml, m)?)?;
    m.add_function(wrap_pyfunction!(deep_merge, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_config_value, m)?)?;
    m.add_function(wrap_pyfunction!(validate_overrides, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(split_train_validation, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_seed_spec, m)?)?;
    m.add_function(wrap_pyfunction!(expand_config, m)?)?;
    m.add_function(wrap_pyfunction!(expand_eval_config, m)?)?;
    m.add_function(wrap_pyfunction!(expand_gepa_config, m)?)?;
    m.add_function(wrap_pyfunction!(is_minimal_config, m)?)?;
    m.add_function(wrap_pyfunction!(build_prompt_learning_payload, m)?)?;
    m.add_function(wrap_pyfunction!(validate_prompt_learning_config, m)?)?;
    m.add_function(wrap_pyfunction!(validate_prompt_learning_config_strict, m)?)?;
    m.add_function(wrap_pyfunction!(validate_graphgen_job_config, m)?)?;
    m.add_function(wrap_pyfunction!(graph_opt_supported_models, m)?)?;
    m.add_function(wrap_pyfunction!(validate_graphgen_taskset, m)?)?;
    m.add_function(wrap_pyfunction!(parse_graphgen_taskset, m)?)?;
    m.add_function(wrap_pyfunction!(load_graphgen_taskset, m)?)?;
    m.add_function(wrap_pyfunction!(validate_graph_job_section, m)?)?;
    m.add_function(wrap_pyfunction!(load_graph_job_toml, m)?)?;
    m.add_function(wrap_pyfunction!(validate_graph_job_payload, m)?)?;
    m.add_function(wrap_pyfunction!(parse_graph_evolve_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(load_graph_evolve_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_graph_evolve_policy_models, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_evolve_config, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_evolve_payload, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_graph_evolve_snapshot_id, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_evolve_graph_record_payload, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_evolve_inference_payload, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph_evolve_placeholder_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_submit_job, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_get_status, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_start_job, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_get_events, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_get_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_download_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_download_graph_txt, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_run_inference, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_get_graph_record, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_cancel_job, m)?)?;
    m.add_function(wrap_pyfunction!(graph_evolve_query_workflow_state, m)?)?;
    m.add_function(wrap_pyfunction!(convert_openai_sft, m)?)?;
    m.add_function(wrap_pyfunction!(build_verifier_request, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_graph_job_id, m)?)?;

    // Models (NEW)
    m.add_function(wrap_pyfunction!(normalize_model_identifier, m)?)?;
    m.add_function(wrap_pyfunction!(detect_model_provider, m)?)?;
    m.add_function(wrap_pyfunction!(supported_models, m)?)?;

    // Trace upload (NEW)
    m.add_class::<TraceUploadClientPy>()?;

    // Data normalization (NEW)
    m.add_function(wrap_pyfunction!(normalize_rubric, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_judgement, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_objective_spec, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_reward_observation, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_outcome_reward_record, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_event_reward_record, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_context_override, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_trace, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_llm_call_record, m)?)?;
    m.add_function(wrap_pyfunction!(data_enum_values, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_for_json, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_http_json, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_trace_for_http, m)?)?;

    // Data models (NEW)
    m.add_class::<CriterionExamplePy>()?;
    m.add_class::<CriterionPy>()?;
    m.add_class::<RubricPy>()?;
    m.add_class::<CriterionScoreDataPy>()?;
    m.add_class::<RubricAssignmentPy>()?;
    m.add_class::<JudgementPy>()?;
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
    m.add_class::<ContextOverridePy>()?;
    m.add_class::<ContextOverrideStatusPy>()?;
    m.add_class::<TaskDatasetSpecPy>()?;
    m.add_class::<MutationTypeStatsPy>()?;
    m.add_class::<MutationSummaryPy>()?;
    m.add_class::<SeedAnalysisPy>()?;
    m.add_class::<PhaseSummaryPy>()?;
    m.add_class::<StageInfoPy>()?;
    m.add_class::<SeedInfoPy>()?;
    m.add_class::<TokenUsagePy>()?;
    m.add_class::<ProgramCandidatePy>()?;
    m.add_class::<BaseJobEventPy>()?;
    m.add_class::<JobEventPy>()?;
    m.add_class::<CandidateEventPy>()?;
    m.add_class::<ArtifactPy>()?;
    m.add_class::<ArtifactBundlePy>()?;
    m.add_class::<TimeRecordPy>()?;
    m.add_class::<MessageContentPy>()?;
    m.add_class::<BaseEventFieldsPy>()?;
    m.add_class::<LMCAISEventPy>()?;
    m.add_class::<EnvironmentEventPy>()?;
    m.add_class::<RuntimeEventPy>()?;
    m.add_class::<TracingEventPy>()?;
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
    m.add_class::<LeaseInfoPy>()?;
    m.add_class::<LeaseClientPy>()?;
    m.add_class::<ConnectorStatusPy>()?;
    m.add_class::<GatewayStatusPy>()?;
    m.add_class::<DiagnosticsPy>()?;
    m.add_class::<TunnelHandleModelPy>()?;

    // Orchestration events (NEW)
    m.add_function(wrap_pyfunction!(parse_orchestration_event, m)?)?;
    m.add_function(wrap_pyfunction!(parse_optimization_event, m)?)?;
    m.add_function(wrap_pyfunction!(parse_job_event, m)?)?;
    m.add_function(wrap_pyfunction!(validate_base_event, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_seed_score_entry, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_extract_stages_from_candidate, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_extract_program_candidate_content, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_normalize_transformation, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_build_program_candidate, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_max_instruction_length, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_max_rollout_samples, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_max_seed_info_count, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_event_enum_values, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_is_valid_event_type, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_validate_event_type, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_base_event_schemas, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_base_job_event_schema, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_get_base_schema, m)?)?;
    m.add_function(wrap_pyfunction!(orchestration_merge_event_schema, m)?)?;
    m.add_class::<ProgressTrackerPy>()?;

    // API Client (NEW)
    m.add_class::<SynthClient>()?;

    // Orchestration (NEW)
    m.add_class::<PromptLearningJob>()?;
    m.add_class::<GraphEvolveJob>()?;

    // Tracing (NEW)
    m.add_function(wrap_pyfunction!(tracing_detect_provider, m)?)?;
    m.add_function(wrap_pyfunction!(tracing_calculate_cost, m)?)?;
    m.add_class::<LibsqlTraceStoragePy>()?;
    m.add_class::<SessionTracerPy>()?;

    // Streaming (NEW)
    m.add_class::<StreamConfigPy>()?;
    m.add_class::<StreamEndpointsPy>()?;
    m.add_class::<JobStreamerPy>()?;

    // LocalAPI (NEW)
    m.add_class::<TaskAppClientPy>()?;
    m.add_class::<EnvClientPy>()?;
    m.add_class::<TaskDatasetRegistryPy>()?;

    Ok(())
}
