from __future__ import annotations

import pytest

from synth_ai.core.utils.urls import is_synthtunnel_url


def test_is_synthtunnel_url_rejects_path_only_match_on_untrusted_host() -> None:
    assert not is_synthtunnel_url("https://attacker.example/s/rt_fake123")


def test_is_synthtunnel_url_accepts_trusted_relay_hosts() -> None:
    assert is_synthtunnel_url("https://st.usesynth.ai/s/rt_abc")
    assert is_synthtunnel_url("https://dev.st.usesynth.ai/s/rt_abc")
    assert is_synthtunnel_url("https://infra-api-dev.usesynth.ai/s/rt_abc/health")
    assert is_synthtunnel_url("http://127.0.0.1:8080/s/rt_local/rollout")


def test_is_synthtunnel_url_rejects_non_route_paths_on_trusted_hosts() -> None:
    assert not is_synthtunnel_url("https://st.usesynth.ai/api/v1/jobs")
    assert not is_synthtunnel_url("https://api-dev.usesynth.ai/api/v1/health")


def test_is_synthtunnel_url_accepts_env_extended_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SYNTH_TUNNEL_TRUSTED_HOSTS",
        "relay.dev.example.com,*.edge.dev.example.com",
    )
    assert is_synthtunnel_url("https://relay.dev.example.com/s/rt_abc")
    assert is_synthtunnel_url("https://foo.edge.dev.example.com/s/rt_abc/health")
