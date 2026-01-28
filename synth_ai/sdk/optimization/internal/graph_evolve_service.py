from __future__ import annotations

from typing import Any, Dict, Optional

from synth_ai.core.rust_core.urls import ensure_api_base

from .utils import http_get, http_post, parse_json_response


def _normalize_backend_url(backend_url: str) -> str:
    return ensure_api_base(backend_url).rstrip("/")


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_with_fallback(
    *,
    primary_url: str,
    legacy_url: Optional[str],
    api_key: str,
    timeout: float,
) -> Any:
    resp = http_get(primary_url, headers=_headers(api_key), timeout=timeout)
    if resp.status_code == 404 and legacy_url:
        resp = http_get(legacy_url, headers=_headers(api_key), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed: {resp.status_code} - {resp.text[:500]}")
    return parse_json_response(resp, context=f"GET {primary_url}")


def _post_with_fallback(
    *,
    primary_url: str,
    legacy_url: Optional[str],
    api_key: str,
    payload: Any,
    timeout: float,
    error_context: str,
) -> Any:
    resp = http_post(primary_url, headers=_headers(api_key), json_body=payload, timeout=timeout)
    if resp.status_code == 404 and legacy_url:
        resp = http_post(legacy_url, headers=_headers(api_key), json_body=payload, timeout=timeout)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"{error_context}: {resp.status_code} - {resp.text[:500]}")
    return parse_json_response(resp, context=error_context)


def submit_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    create_url = f"{base_url}/graph-evolve/jobs"
    legacy_url = f"{base_url}/graphgen/jobs"

    resp = http_post(create_url, headers=_headers(api_key), json_body=payload, timeout=180.0)
    if resp.status_code == 404:
        resp = http_post(legacy_url, headers=_headers(api_key), json_body=payload, timeout=180.0)

    if resp.status_code not in (200, 201):
        error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
        if resp.status_code == 404:
            error_msg += (
                f"\n\nPossible causes:"
                f"\n1. Backend route /api/graph-evolve/jobs not registered"
                f"\n2. Graph Evolve feature may not be enabled on this backend"
                f"\n3. Verify backend is running at: {backend_url}"
            )
        raise RuntimeError(error_msg)

    return parse_json_response(resp, context="Graph evolve submission")


def get_graph_evolve_status(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    primary_url = f"{base_url}/graph-evolve/jobs/{job_id}"
    legacy_url = f"{base_url}/graphgen/jobs/{job_id}"
    return _get_with_fallback(
        primary_url=primary_url,
        legacy_url=legacy_url,
        api_key=api_key,
        timeout=30.0,
    )


def start_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    primary_url = f"{base_url}/graph-evolve/jobs/{job_id}/start"
    legacy_url = f"{base_url}/graphgen/jobs/{job_id}/start"
    return _post_with_fallback(
        primary_url=primary_url,
        legacy_url=legacy_url,
        api_key=api_key,
        payload=None,
        timeout=60.0,
        error_context="Failed to start job",
    )


def get_graph_evolve_events(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    since_seq: int,
    limit: int,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    base = f"{base_url}/graph-evolve/jobs/{job_id}/events"
    primary_url = f"{base}?since_seq={since_seq}&limit={limit}"
    legacy_base = f"{base_url}/graphgen/jobs/{job_id}/events"
    legacy_url = f"{legacy_base}?since_seq={since_seq}&limit={limit}"
    return _get_with_fallback(
        primary_url=primary_url,
        legacy_url=legacy_url,
        api_key=api_key,
        timeout=30.0,
    )


def get_graph_evolve_metrics(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    query_string: str,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    primary_url = f"{base_url}/graph-evolve/jobs/{job_id}/metrics?{query_string}"
    legacy_url = f"{base_url}/graphgen/jobs/{job_id}/metrics?{query_string}"
    return _get_with_fallback(
        primary_url=primary_url,
        legacy_url=legacy_url,
        api_key=api_key,
        timeout=30.0,
    )


def download_graph_evolve_prompt(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    primary_url = f"{base_url}/graph-evolve/jobs/{job_id}/download"
    legacy_url = f"{base_url}/graphgen/jobs/{job_id}/download"
    return _get_with_fallback(
        primary_url=primary_url,
        legacy_url=legacy_url,
        api_key=api_key,
        timeout=30.0,
    )


def download_graph_evolve_graph_txt(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> str:
    base_url = _normalize_backend_url(backend_url)
    primary_url = f"{base_url}/graph-evolve/jobs/{job_id}/graph.txt"
    legacy_url = f"{base_url}/graphgen/jobs/{job_id}/graph.txt"
    resp = http_get(primary_url, headers=_headers(api_key), timeout=30.0)
    if resp.status_code == 404:
        resp = http_get(legacy_url, headers=_headers(api_key), timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download graph export: {resp.status_code} - {resp.text[:500]}"
        )
    return resp.text


def run_graph_evolve_inference(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    url = f"{base_url}/graphgen/graph/completions"
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=60.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Inference failed: {resp.status_code} - {resp.text[:500]}")
    return parse_json_response(resp, context="Graph evolve inference")


def get_graph_evolve_graph_record(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    url = f"{base_url}/graphgen/graph/record"
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get graph record: {resp.status_code} - {resp.text[:500]}")
    return parse_json_response(resp, context="Graph evolve graph record")


def cancel_graph_evolve_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    url = f"{base_url}/jobs/{job_id}/cancel"
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    return parse_json_response(resp, context="Graph evolve cancel")


def query_graph_evolve_workflow_state(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    base_url = _normalize_backend_url(backend_url)
    url = f"{base_url}/jobs/{job_id}/workflow-state"
    resp = http_get(url, headers=_headers(api_key), timeout=10.0)
    if resp.status_code != 200:
        return {
            "job_id": job_id,
            "workflow_state": None,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
        }
    return parse_json_response(resp, context="Graph evolve workflow state")
