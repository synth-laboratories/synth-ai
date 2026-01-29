from __future__ import annotations

from typing import Any, Dict

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.graph_evolve_service.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "graph_evolve_submit_job"):
        raise RuntimeError("Rust core Graph Evolve service required; synth_ai_py is unavailable.")
    return synth_ai_py


def submit_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_submit_job(api_key, backend_url, payload)


def get_graph_evolve_status(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_get_status(api_key, backend_url, job_id)


def start_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_start_job(api_key, backend_url, job_id)


def get_graph_evolve_events(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    since_seq: int,
    limit: int,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_get_events(api_key, backend_url, job_id, since_seq, limit)


def get_graph_evolve_metrics(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    query_string: str,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_get_metrics(api_key, backend_url, job_id, query_string)


def download_graph_evolve_prompt(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_download_prompt(api_key, backend_url, job_id)


def download_graph_evolve_graph_txt(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> str:
    rust = _require_rust()
    return rust.graph_evolve_download_graph_txt(api_key, backend_url, job_id)


def run_graph_evolve_inference(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_run_inference(api_key, backend_url, payload)


def get_graph_evolve_graph_record(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_get_graph_record(api_key, backend_url, payload)


def cancel_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_cancel_job(api_key, backend_url, job_id, payload)


def query_graph_evolve_workflow_state(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    rust = _require_rust()
    return rust.graph_evolve_query_workflow_state(api_key, backend_url, job_id)
