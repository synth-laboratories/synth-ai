from __future__ import annotations

import asyncio

import pytest

from synth_ai.core.tunnels import TunnelBackend, TunnelProvider, TunneledContainer
from synth_ai.core.tunnels.errors import (
    TunnelErrorCode,
    TunnelProviderError,
    map_problem_to_tunnel_error_code,
)


def test_localhost_provider_requires_no_api_key() -> None:
    tunnel = asyncio.run(
        TunneledContainer.create(local_port=8114, provider=TunnelProvider.Localhost)
    )
    assert tunnel.backend == TunnelBackend.Localhost
    assert tunnel.url == "http://localhost:8114"


def test_provider_backend_conflict_fails() -> None:
    with pytest.raises(TunnelProviderError) as exc:
        asyncio.run(
            TunneledContainer.create(
                local_port=8114,
                provider=TunnelProvider.Ngrok,
                backend=TunnelBackend.Localhost,
            )
        )
    assert exc.value.code == TunnelErrorCode.PROVIDER_INVALID


def test_problem_code_mapping_uses_tunnel_taxonomy() -> None:
    assert (
        map_problem_to_tunnel_error_code(
            "bad_request",
            "No default managed ngrok URL configured for this org.",
        )
        == TunnelErrorCode.URL_REQUIRED
    )
    assert map_problem_to_tunnel_error_code("unauthorized", "") == TunnelErrorCode.AUTH_INVALID
    assert map_problem_to_tunnel_error_code("rate_limited", "") == TunnelErrorCode.CAPACITY_EXCEEDED
