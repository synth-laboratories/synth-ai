from __future__ import annotations

from typing import Any, Dict, Optional

from .utils import ensure_api_base, http_get, http_post, parse_json_response


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }


def submit_prompt_learning_job(
    *,
    backend_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    create_url = f"{ensure_api_base(backend_url)}/prompt-learning/online/jobs"

    resp = http_post(create_url, headers=_headers(api_key), json_body=payload)
    if resp.status_code not in (200, 201):
        error_msg = f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
        if resp.status_code == 404:
            error_msg += (
                f"\n\nPossible causes:"
                f"\n1. Backend route /api/prompt-learning/online/jobs not registered"
                f"\n2. Backend server needs restart (lazy import may have failed)"
                f"\n3. Check backend logs for: 'Failed to import prompt_learning_online_router'"
                f"\n4. Verify backend is running at: {backend_url}"
            )
        raise RuntimeError(error_msg)
    return parse_json_response(resp, context="Prompt learning submission")


def cancel_prompt_learning_job(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{ensure_api_base(backend_url)}/prompt-learning/online/jobs/{job_id}/cancel"
    payload: Dict[str, Any] = {}
    if reason:
        payload["reason"] = reason
    resp = http_post(url, headers=_headers(api_key), json_body=payload, timeout=30.0)
    return parse_json_response(resp, context="Prompt learning cancel")


def query_prompt_learning_workflow_state(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
) -> Dict[str, Any]:
    url = f"{ensure_api_base(backend_url)}/jobs/{job_id}/workflow-state"
    resp = http_get(url, headers=_headers(api_key), timeout=10.0)
    if resp.status_code != 200:
        return {
            "job_id": job_id,
            "workflow_state": None,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
        }
    return parse_json_response(resp, context="Prompt learning workflow state")
