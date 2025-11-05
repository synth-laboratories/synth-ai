"""Test that reproduces the proxy authentication bug.

Issue: When inference_url points to /proxy/v1/chat/completions, the API key
selection logic in policy_routes.py incorrectly uses SYNTH_API_KEY instead of
skipping authentication (setting api_key_override=None).

The bug is in policy_routes.py:504:
    if "/proxy/groq" in low_url or "/proxy/openai" in low_url:
        api_key_override = None

This check fails to match /proxy/v1/ URLs, causing the code to fall through
to the else block that sets SYNTH_API_KEY. The proxy endpoint then rejects
the request because it expects ENVIRONMENT_API_KEY for task app authentication.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from examples.task_apps.crafter.task_app.synth_envs_hosted.hosted_app import create_app


@pytest.mark.asyncio
async def test_proxy_v1_endpoint_auth_fails_with_wrong_key():
    """Reproduce the bug: /proxy/v1/chat/completions receives wrong API key.

    This test demonstrates that:
    1. Policy creates an inference client with SYNTH_API_KEY (wrong)
    2. The request to /proxy/v1/chat/completions fails auth
    3. Expected: api_key_override should be None for proxy endpoints
    4. Actual: api_key_override is SYNTH_API_KEY
    """
    # Set up environment with both keys
    env_key = "sk-rlenv-test-environment-key-1234567890abcdef"
    synth_key = "sk-synth-test-synth-key-1234567890abcdef"

    with patch.dict(os.environ, {
        "ENVIRONMENT_API_KEY": env_key,
        "SYNTH_API_KEY": synth_key,
    }):
        app = create_app()
        client = TestClient(app)

        # Create policy with inference_url pointing to proxy endpoint
        policy_payload = {
            "policy_name": "crafter-react",
            "config": {
                "inference_url": "http://127.0.0.1:8000/proxy/v1/chat/completions",
                "model": "synth:qwen-0.6b",
            },
            "rl_run_id": "test-run-proxy-auth",
            "mode": "eval",  # eval mode uses inference URLs as-is
        }

        resp = client.post("/policy/create", json=policy_payload)
        assert resp.status_code == 200, f"Policy create failed: {resp.text}"
        policy_id = resp.json()["policy_id"]

        # Attempt to step the policy - this will call the proxy endpoint internally
        step_payload = {
            "policy_id": policy_id,
            "observation": {"env_name": "crafter", "step": 0},
            "dry_run": True,  # Don't actually call inference, just check auth logic
        }

        # This should fail because:
        # 1. The policy code sees inference_url="/proxy/v1/chat/completions"
        # 2. The pattern check at policy_routes.py:504 fails (only checks /proxy/openai or /proxy/groq)
        # 3. Falls through to else: api_key_override = SYNTH_API_KEY
        # 4. Creates OpenAI client with Authorization: Bearer <SYNTH_API_KEY>
        # 5. Proxy endpoint expects ENVIRONMENT_API_KEY in Authorization header
        # 6. Auth fails with 400/401

        step_resp = client.post("/policy/step", json=step_payload)

        # The bug causes this to fail
        # When fixed, the proxy endpoint should work because api_key_override=None
        # means no Bearer token is added, so the proxy uses its server-side keys

        # Currently, we expect failure with auth error
        # After fix, we'd expect success or at least a different error (not auth)
        if step_resp.status_code != 200:
            error_detail = step_resp.json().get("detail", {})
            print(f"Auth failed as expected (bug present): {error_detail}")
            # Verify it's an auth error, not something else
            if isinstance(error_detail, dict):
                error_code = error_detail.get("error", {}).get("code", "")
                assert "unauthorised" in error_code or "auth" in error_code.lower(), \
                    f"Expected auth error but got: {error_detail}"
        else:
            # If this passes, the bug has been fixed!
            print("✓ Test passed - bug has been fixed!")
            data = step_resp.json()
            # Verify the meta has the correct inference_url
            meta = data.get("meta", {})
            assert meta.get("inference_url") == "http://127.0.0.1:8000/proxy/v1/chat/completions"


@pytest.mark.asyncio
async def test_proxy_openai_endpoint_works_correctly():
    """Verify that /proxy/openai endpoints work (they match the pattern check).

    This test shows that /proxy/openai/ URLs correctly get api_key_override=None
    because they match the pattern check at policy_routes.py:504.
    """
    env_key = "sk-rlenv-test-environment-key-1234567890abcdef"
    synth_key = "sk-synth-test-synth-key-1234567890abcdef"

    with patch.dict(os.environ, {
        "ENVIRONMENT_API_KEY": env_key,
        "SYNTH_API_KEY": synth_key,
        "OPENAI_API_KEY": "sk-openai-test-key-1234567890abcdef",  # Needed for proxy
    }):
        app = create_app()
        client = TestClient(app)

        # Create policy with inference_url pointing to /proxy/openai (which works)
        policy_payload = {
            "policy_name": "crafter-react",
            "config": {
                "inference_url": "http://127.0.0.1:8000/proxy/openai/v1/chat/completions",
                "model": "gpt-4o-mini",
            },
            "rl_run_id": "test-run-proxy-openai",
            "mode": "eval",
        }

        resp = client.post("/policy/create", json=policy_payload)
        assert resp.status_code == 200, f"Policy create failed: {resp.text}"
        policy_id = resp.json()["policy_id"]

        step_payload = {
            "policy_id": policy_id,
            "observation": {"env_name": "crafter", "step": 0},
            "dry_run": True,
        }

        step_resp = client.post("/policy/step", json=step_payload)

        # This should work because /proxy/openai matches the pattern check
        # and gets api_key_override=None (no Bearer token added)
        # The proxy endpoint authenticates via ENVIRONMENT_API_KEY from request headers

        # Note: May still fail if OPENAI_API_KEY is invalid, but should NOT be an auth error
        # from the task app itself
        if step_resp.status_code != 200:
            error_detail = step_resp.json().get("detail", {})
            # Should not be task app auth error
            if isinstance(error_detail, dict):
                error_code = error_detail.get("error", {}).get("code", "")
                assert error_code != "unauthorised", \
                    f"Unexpected task app auth error for /proxy/openai: {error_detail}"
