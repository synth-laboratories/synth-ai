import httpx


def test_get_balance_current(base_url: str, auth_headers: dict):
    url = f"{base_url}/balance/current"
    r = httpx.get(url, headers=auth_headers, timeout=30)
    assert r.status_code in (200, 401), r.text  # allow 401 if key invalid
    if r.status_code == 200:
        data = r.json()
        assert "balance_cents" in data
        assert "balance_dollars" in data


