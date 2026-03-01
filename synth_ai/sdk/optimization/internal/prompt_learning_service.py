from __future__ import annotations

from typing import Any

from .utils import ensure_api_base, http_get, http_post, parse_json_response


def _headers(api_key: str, *, worker_token: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    if worker_token and worker_token.strip():
        headers["X-SynthTunnel-Worker-Token"] = worker_token.strip()
    return headers


def cancel_prompt_learning_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    api_base = ensure_api_base(backend_url)
    url = f"{api_base}/jobs/{job_id}/cancel"
    payload: dict[str, Any] = {}
    if reason:
        payload["reason"] = reason
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    return parse_json_response(resp, context="Prompt learning cancel")


def pause_prompt_learning_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    api_base = ensure_api_base(backend_url)
    url = f"{api_base}/jobs/{job_id}/pause"
    payload: dict[str, Any] = {}
    if reason:
        payload["reason"] = reason
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    return parse_json_response(resp, context="Prompt learning pause")


def resume_prompt_learning_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    api_base = ensure_api_base(backend_url)
    url = f"{api_base}/jobs/{job_id}/resume"
    payload: dict[str, Any] = {}
    if reason:
        payload["reason"] = reason
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    return parse_json_response(resp, context="Prompt learning resume")


def query_prompt_learning_workflow_state(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> dict[str, Any]:
    url = f"{ensure_api_base(backend_url)}/jobs/{job_id}/workflow-state"
    resp = http_get(url, headers=_headers(api_key), timeout=10.0)
    if resp.status_code != 200:
        return {
            "job_id": job_id,
            "workflow_state": None,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
        }
    return parse_json_response(resp, context="Prompt learning workflow state")
