from __future__ import annotations

import pytest


@pytest.mark.unit
def test_synth_ai_py_exports_expected_symbols() -> None:
    import synth_ai_py

    assert hasattr(synth_ai_py, "synth_tunnel_create_lease")
    assert hasattr(synth_ai_py, "synth_tunnel_close_lease")

    assert hasattr(synth_ai_py, "EnvironmentPoolsClient")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "get_json")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "post_json")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "put_json")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "delete")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "get_bytes")
    assert hasattr(synth_ai_py.EnvironmentPoolsClient, "stream_rollout_events")

    assert hasattr(synth_ai_py, "HttpClient")
    assert hasattr(synth_ai_py.HttpClient, "get_json")
    assert hasattr(synth_ai_py.HttpClient, "post_json")
    assert hasattr(synth_ai_py.HttpClient, "put_json")
    assert hasattr(synth_ai_py.HttpClient, "delete")

    assert hasattr(synth_ai_py, "env_pools_check_plan_access")
