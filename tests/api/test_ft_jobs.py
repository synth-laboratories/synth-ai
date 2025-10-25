import pytest
import httpx


@pytest.mark.slow
def test_ft_jobs_list_and_get_stub(base_url: str, auth_headers: dict):
    # Use a random (non-existent) job id to hit the GET job status stub
    api_base = base_url.replace("/v1", "")
    job_id = "ftjob-deadbeefdeadbeefdeadbeefdeadbeef"
    get_url = f"{api_base}/fine_tuning/jobs/{job_id}"
    r2 = httpx.get(get_url, headers=auth_headers, timeout=30)
    # API should respond 200 with queued stub or 401 if unauthorized
    assert r2.status_code in (200, 401), r2.text
    if r2.status_code == 200:
        data = r2.json()
        assert data.get("id") == job_id
        assert data.get("status") in (
            "queued",
            "running",
            "succeeded",
            "failed",
            "validating_files",
        )
