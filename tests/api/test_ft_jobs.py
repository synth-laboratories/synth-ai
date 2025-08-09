import httpx
import time


def test_ft_jobs_list_and_get_stub(base_url: str, auth_headers: dict):
    # List jobs (should not error)
    list_url = f"{base_url}/learning_v2/fine_tuning/jobs"
    r = httpx.get(list_url, headers=auth_headers, timeout=30)
    assert r.status_code in (200, 401)

    # Create a stubbed GET by using a random (non-existent) job id should now return queued stub
    # thanks to API behavior which avoids early 404s.
    job_id = "ftjob-deadbeefdeadbeefdeadbeefdeadbeef"
    get_url = f"{base_url}/learning_v2/fine_tuning/jobs/{job_id}"
    r2 = httpx.get(get_url, headers=auth_headers, timeout=30)
    # API should respond 200 with queued stub or 401 if unauthorized
    assert r2.status_code in (200, 401), r2.text
    if r2.status_code == 200:
        data = r2.json()
        assert data.get("id") == job_id
        assert data.get("status") in ("queued", "running", "succeeded", "failed", "validating_files")


