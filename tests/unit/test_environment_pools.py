"""Unit tests for environment_pools SDK."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestURLBuilding:
    """Test URL construction helpers."""

    def test_resolve_base_url_with_explicit(self):
        from synth_ai.sdk.environment_pools import _resolve_base_url

        result = _resolve_base_url("https://custom.api.com", default="https://default.com")
        # normalize_backend_base adds trailing slash
        assert result == "https://custom.api.com/" or result == "https://custom.api.com"

    def test_resolve_base_url_uses_default(self):
        from synth_ai.sdk.environment_pools import _resolve_base_url

        result = _resolve_base_url(None, default="https://default.com")
        assert "default.com" in result

    def test_resolve_base_url_strips_whitespace(self):
        from synth_ai.sdk.environment_pools import _resolve_base_url

        result = _resolve_base_url("  ", default="https://default.com")
        assert "default.com" in result

    def test_url_builds_v1_path(self):
        from synth_ai.sdk.environment_pools import _url

        result = _url("https://api.example.com", "rollouts", api_version="v1")
        assert result == "https://api.example.com/v1/rollouts"

    def test_url_builds_legacy_path(self):
        from synth_ai.sdk.environment_pools import _url

        result = _url("https://api.example.com", "rollouts", api_version="legacy")
        assert "/api/v1/environment-pools/rollouts" in result

    def test_cred_url_builds_correctly(self):
        from synth_ai.sdk.environment_pools import _cred_url

        result = _cred_url("https://api.example.com", "cred123")
        assert result == "https://api.example.com/api/v1/credentials/cred123"

    def test_cred_url_no_path(self):
        from synth_ai.sdk.environment_pools import _cred_url

        result = _cred_url("https://api.example.com")
        assert result == "https://api.example.com/api/v1/credentials"


class TestAPIKeyResolution:
    """Test API key resolution logic."""

    def test_resolve_api_key_explicit(self):
        from synth_ai.sdk.environment_pools import _resolve_api_key

        result = _resolve_api_key("sk_test_123")
        assert result == "sk_test_123"

    def test_resolve_api_key_from_env(self):
        from synth_ai.sdk.environment_pools import _resolve_api_key

        with patch.dict(os.environ, {"SYNTH_API_KEY": "sk_env_456"}):
            result = _resolve_api_key(None)
            assert result == "sk_env_456"

    def test_resolve_api_key_missing_raises(self):
        from synth_ai.sdk.environment_pools import _resolve_api_key

        # Mock get_api_key to raise and clear env var
        old_value = os.environ.pop("SYNTH_API_KEY", None)
        try:
            with patch("synth_ai.sdk.environment_pools.get_api_key", side_effect=Exception("not found")):
                with pytest.raises(ValueError, match="api_key is required"):
                    _resolve_api_key(None)
        finally:
            if old_value is not None:
                os.environ["SYNTH_API_KEY"] = old_value

    def test_resolve_api_key_strips_whitespace(self):
        from synth_ai.sdk.environment_pools import _resolve_api_key

        result = _resolve_api_key("  sk_test_789  ")
        assert result == "  sk_test_789  "  # Note: explicit key not stripped


class TestHeaders:
    """Test header construction."""

    def test_auth_headers(self):
        from synth_ai.sdk.environment_pools import _auth_headers

        headers = _auth_headers("sk_test")
        assert headers == {"Authorization": "Bearer sk_test"}

    def test_request_headers_without_idempotency(self):
        from synth_ai.sdk.environment_pools import _request_headers

        headers = _request_headers("sk_test", None)
        assert headers == {"Authorization": "Bearer sk_test"}

    def test_request_headers_with_idempotency(self):
        from synth_ai.sdk.environment_pools import _request_headers

        headers = _request_headers("sk_test", "idem-key-123")
        assert headers == {
            "Authorization": "Bearer sk_test",
            "Idempotency-Key": "idem-key-123",
        }


class TestSchemas:
    """Test Pydantic schema definitions."""

    def test_task_ref_creation(self):
        from synth_ai.sdk.environment_pools import TaskRef

        ref = TaskRef(dataset="my-dataset", task_id="task-1")
        assert ref.dataset == "my-dataset"
        assert ref.task_id == "task-1"
        assert ref.version is None

    def test_task_ref_with_version(self):
        from synth_ai.sdk.environment_pools import TaskRef

        ref = TaskRef(dataset="my-dataset", task_id="task-1", version="v2")
        assert ref.version == "v2"

    def test_agent_spec_creation(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec(harness="claude-code", model_id="claude-sonnet-4.5")
        assert spec.harness == "claude-code"
        assert spec.model_id == "claude-sonnet-4.5"
        assert spec.harness_version is None

    def test_agent_spec_with_version(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec(harness="opencode", harness_version="1.2.3", model_id="claude-sonnet-4.5")
        assert spec.harness_version == "1.2.3"

    def test_environment_spec(self):
        from synth_ai.sdk.environment_pools import EnvironmentSpec

        spec = EnvironmentSpec(backend="harbor", docker_image="my-image:latest")
        assert spec.backend == "harbor"
        assert spec.docker_image == "my-image:latest"

    def test_timeout_spec(self):
        from synth_ai.sdk.environment_pools import TimeoutSpec

        spec = TimeoutSpec(agent_sec=300, verifier_sec=60)
        assert spec.agent_sec == 300
        assert spec.verifier_sec == 60

    def test_pool_config_v1(self):
        from synth_ai.sdk.environment_pools import PoolConfigV1

        config = PoolConfigV1(
            pool_id="pool-123",
            pool_type="sandbox",
            capacity=10,
            concurrency=5,
            spend_limit_usd=100.0,
            idle_timeout_sec=300,
            prewarm=2,
        )
        assert config.pool_id == "pool-123"
        assert config.capacity == 10
        assert config.spend_limit_usd == 100.0
        assert config.prewarm == 2


class TestRolloutHandle:
    """Test RolloutHandle ergonomics."""

    def test_rollout_handle_creation(self):
        from synth_ai.sdk.environment_pools import RolloutHandle

        handle = RolloutHandle(
            "rollout-123",
            backend_base="https://api.example.com",
            api_key="sk_test",
        )
        assert handle.rollout_id == "rollout-123"
        assert handle.backend_base == "https://api.example.com"
        assert handle.api_key == "sk_test"

    @patch("synth_ai.sdk.environment_pools.get_rollout")
    def test_rollout_handle_get(self, mock_get):
        from synth_ai.sdk.environment_pools import RolloutHandle

        mock_get.return_value = {"rollout_id": "rollout-123", "status": "running"}
        handle = RolloutHandle("rollout-123", api_key="sk_test")
        result = handle.get()
        assert result["status"] == "running"
        mock_get.assert_called_once()

    @patch("synth_ai.sdk.environment_pools.get_rollout")
    def test_rollout_handle_wait_success(self, mock_get):
        from synth_ai.sdk.environment_pools import RolloutHandle

        mock_get.side_effect = [
            {"rollout_id": "rollout-123", "status": "running"},
            {"rollout_id": "rollout-123", "status": "succeeded"},
        ]
        handle = RolloutHandle("rollout-123", api_key="sk_test")
        result = handle.wait(poll_interval=0.01, timeout=1.0)
        assert result["status"] == "succeeded"


class TestPayloadConversion:
    """Test payload conversion from request objects."""

    def test_payload_from_dict(self):
        from synth_ai.sdk.environment_pools import _payload_from_request

        request = {"task_ref": {"dataset": "test", "task_id": "t1"}}
        result = _payload_from_request(request)
        assert result == request

    def test_payload_from_pydantic(self):
        from synth_ai.sdk.environment_pools import AgentSpec, _payload_from_request

        spec = AgentSpec(harness="claude-code", model_id="claude-sonnet-4.5")
        result = _payload_from_request(spec)
        assert result["harness"] == "claude-code"
        assert result["model_id"] == "claude-sonnet-4.5"

    def test_payload_adds_kind_from_harness(self):
        from synth_ai.sdk.environment_pools import _payload_from_request

        request = {"agent": {"harness": "opencode", "model_id": "test"}}
        result = _payload_from_request(request)
        assert result["agent"]["kind"] == "opencode"


class TestRolloutIdExtraction:
    """Test rollout_id extraction from payloads."""

    def test_extracts_rollout_id(self):
        from synth_ai.sdk.environment_pools import _rollout_id_from_payload

        result = _rollout_id_from_payload({"rollout_id": "r-123"})
        assert result == "r-123"

    def test_extracts_trial_id_fallback(self):
        from synth_ai.sdk.environment_pools import _rollout_id_from_payload

        result = _rollout_id_from_payload({"trial_id": "t-456"})
        assert result == "t-456"

    def test_returns_none_for_missing(self):
        from synth_ai.sdk.environment_pools import _rollout_id_from_payload

        result = _rollout_id_from_payload({})
        assert result is None

    def test_returns_none_for_empty_string(self):
        from synth_ai.sdk.environment_pools import _rollout_id_from_payload

        result = _rollout_id_from_payload({"rollout_id": "  "})
        assert result is None


class TestAgentSpecPresets:
    """Test AgentSpec preset convenience constructors."""

    def test_preset_claude_code_default_model(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.preset("claude-code")
        assert spec.harness == "claude-code"
        assert spec.model_id == "claude-sonnet-4.5"
        assert spec.harness_version is None

    def test_preset_with_harness_version(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.preset("claude-code", harness_version="1.0.0")
        assert spec.harness_version == "1.0.0"

    def test_preset_opencode_default_model(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.preset("opencode")
        assert spec.harness == "opencode"
        assert spec.model_id == "claude-sonnet-4.5"

    def test_preset_codex_default_model(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.preset("codex")
        assert spec.harness == "codex"
        assert spec.model_id == "gpt-5.2-codex"

    def test_preset_codex_explicit_model(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.preset("codex", model_id="gpt-5.1-codex-mini")
        assert spec.model_id == "gpt-5.1-codex-mini"

    def test_preset_invalid_model_raises(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        with pytest.raises(ValueError, match="not allowed for harness"):
            AgentSpec.preset("claude-code", model_id="gpt-5.2-codex")

    def test_preset_unknown_harness_raises(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        with pytest.raises(ValueError, match="Unknown preset"):
            AgentSpec.preset("unknown-harness")

    def test_claude_code_classmethod(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.claude_code()
        assert spec.harness == "claude-code"
        assert spec.model_id == "claude-sonnet-4.5"

    def test_opencode_classmethod(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.opencode()
        assert spec.harness == "opencode"
        assert spec.model_id == "claude-sonnet-4.5"

    def test_codex_classmethod(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.codex()
        assert spec.harness == "codex"
        assert spec.model_id == "gpt-5.2-codex"

    def test_codex_invalid_model_raises(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        with pytest.raises(ValueError, match="not allowed for harness"):
            AgentSpec.codex(model_id="claude-sonnet-4.5")

    def test_claude_code_with_version(self):
        from synth_ai.sdk.environment_pools import AgentSpec

        spec = AgentSpec.claude_code(harness_version="2.0.0")
        assert spec.harness_version == "2.0.0"


class TestPoolTaskFactory:
    """Test PoolTask factory methods."""

    def test_from_docker(self):
        from synth_ai.sdk.environment_pools import PoolTask

        task = PoolTask.from_docker("task-1", "my-image:latest")
        assert task.task_id == "task-1"
        assert task.backend == "harbor"
        assert task.docker_image == "my-image:latest"

    def test_from_docker_with_options(self):
        from synth_ai.sdk.environment_pools import PoolTask

        task = PoolTask.from_docker(
            "task-2",
            "my-image:v2",
            env={"FOO": "bar"},
            resources={"cpus": 4, "memory_mb": 8192},
            registry_credential="cred-1",
            task_path="/workspace",
        )
        assert task.env_vars == {"FOO": "bar"}
        assert task.resources is not None
        assert task.resources.cpus == 4
        assert task.resources.memory_mb == 8192
        assert task.registry_credential == "cred-1"
        assert task.task_path == "/workspace"

    def test_from_openenv(self):
        from synth_ai.sdk.environment_pools import PoolTask

        task = PoolTask.from_openenv("task-3", "https://app.example.com")
        assert task.task_id == "task-3"
        assert task.backend == "openenv"
        assert task.openenv_deployment == {"task_app_url": "https://app.example.com"}

    def test_from_openenv_with_custom_deployment(self):
        from synth_ai.sdk.environment_pools import PoolTask

        custom = {"task_app_url": "https://app.example.com", "version": "v2"}
        task = PoolTask.from_openenv("task-4", "https://app.example.com", openenv_deployment=custom)
        assert task.openenv_deployment == custom

    def test_from_browser(self):
        from synth_ai.sdk.environment_pools import PoolTask

        task = PoolTask.from_browser("task-5", "https://app.example.com")
        assert task.task_id == "task-5"
        assert task.backend == "browser"
        assert task.browser == {"task_app_url": "https://app.example.com"}

    def test_from_browser_with_profile(self):
        from synth_ai.sdk.environment_pools import PoolTask

        task = PoolTask.from_browser("task-6", "https://app.example.com", profile="chrome-default")
        assert task.browser is not None
        assert task.browser["profile"] == "chrome-default"

    def test_from_docker_with_pool_resources_object(self):
        from synth_ai.sdk.environment_pools import PoolResources, PoolTask

        res = PoolResources(cpus=2, memory_mb=4096)
        task = PoolTask.from_docker("task-7", "img:latest", resources=res)
        assert task.resources is not None
        assert task.resources.cpus == 2


class TestEnvironmentPoolsClient:
    """Test EnvironmentPoolsClient instantiation and namespace delegation."""

    def test_client_instantiation(self):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        client = EnvironmentPoolsClient(
            api_key="sk_test", backend_base="https://api.example.com", skip_plan_check=True
        )
        assert client._api_key == "sk_test"
        assert client._backend_base == "https://api.example.com"

    def test_client_has_namespaces(self):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        assert hasattr(client, "rollouts")
        assert hasattr(client, "pools")
        assert hasattr(client, "credentials")
        assert hasattr(client, "capabilities")

    def test_pools_has_tasks_sub_namespace(self):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        assert hasattr(client.pools, "tasks")

    @patch("synth_ai.sdk.environment_pools.get_rollout")
    def test_rollouts_get_delegates(self, mock_get):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        mock_get.return_value = {"rollout_id": "r-123", "status": "running"}
        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        handle = client.rollouts.get("r-123")
        assert handle.rollout_id == "r-123"
        mock_get.assert_called_once_with(
            "r-123",
            backend_base=None,
            api_key="sk_test",
            timeout=30.0,
            api_version=None,
        )

    @patch("synth_ai.sdk.environment_pools.list_rollouts")
    def test_rollouts_list_delegates(self, mock_list):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        mock_list.return_value = {"rollouts": [], "cursor": None}
        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        result = client.rollouts.list(limit=10)
        assert result == {"rollouts": [], "cursor": None}
        mock_list.assert_called_once()

    @patch("synth_ai.sdk.environment_pools.get_capabilities")
    def test_capabilities_get_delegates(self, mock_caps):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        mock_caps.return_value = {"agents": ["claude-code"]}
        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        result = client.capabilities.get()
        assert result == {"agents": ["claude-code"]}

    @patch("synth_ai.sdk.environment_pools.list_pools")
    def test_pools_list_delegates(self, mock_list):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        mock_list.return_value = [{"pool_id": "p-1"}]
        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        result = client.pools.list()
        assert result == [{"pool_id": "p-1"}]


class TestPlanGating:
    """Test plan gating infrastructure."""

    def test_plan_gating_error_message(self):
        from synth_ai.core.errors import PlanGatingError

        err = PlanGatingError(feature="environment_pools", current_plan="free")
        msg = str(err)
        assert "Pro" in msg or "pro" in msg
        assert "team" in msg or "Team" in msg
        assert "free" in msg
        assert "usesynth.ai/pricing" in msg

    def test_check_plan_access_pro_allowed(self):
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"plan": "pro", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "pro"

    def test_check_plan_access_team_allowed(self):
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"plan": "team", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "team"

    def test_check_plan_access_free_rejected(self):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"plan": "free", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError) as exc_info:
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )
            assert exc_info.value.current_plan == "free"

    def test_check_plan_access_trial_rejected(self):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"plan": "trial", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError):
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )

    def test_check_plan_access_demo_allowed(self):
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"plan": "demo", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "demo"

    def test_check_plan_access_me_endpoint_unreachable_falls_through(self):
        from synth_ai.sdk.environment_pools import _check_plan_access

        with patch("httpx.get", side_effect=Exception("connection refused")):
            # Should not raise — falls through gracefully
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result == {}

    def test_check_plan_access_uses_tier_field(self):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"tier": "free", "org_id": "org-1"}

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError) as exc_info:
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )
            assert exc_info.value.current_plan == "free"

    @patch("synth_ai.sdk.environment_pools._check_plan_access")
    def test_client_checks_plan_on_init(self, mock_check):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        mock_check.side_effect = PlanGatingError(
            feature="environment_pools", current_plan="free"
        )
        with pytest.raises(PlanGatingError):
            EnvironmentPoolsClient(api_key="sk_test")

    @patch("synth_ai.sdk.environment_pools._check_plan_access")
    def test_client_skip_plan_check(self, mock_check):
        from synth_ai.sdk.environment_pools import EnvironmentPoolsClient

        client = EnvironmentPoolsClient(api_key="sk_test", skip_plan_check=True)
        mock_check.assert_not_called()
        assert client._account_info == {}

    def test_raise_for_status_with_plan_check_converts_403(self):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _raise_for_status_with_plan_check

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {
            "error": {
                "code": "plan_required",
                "message": "Pro plan required",
                "current_plan": "free",
            }
        }
        with pytest.raises(PlanGatingError) as exc_info:
            _raise_for_status_with_plan_check(mock_resp)
        assert exc_info.value.current_plan == "free"

    def test_raise_for_status_with_plan_check_403_upgrade_message(self):
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _raise_for_status_with_plan_check

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {
            "error": {
                "code": "forbidden",
                "message": "Please upgrade your plan to access this feature",
            }
        }
        with pytest.raises(PlanGatingError):
            _raise_for_status_with_plan_check(mock_resp)

    def test_raise_for_status_with_plan_check_403_non_plan(self):
        from synth_ai.sdk.environment_pools import _raise_for_status_with_plan_check

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {
            "error": {"code": "forbidden", "message": "insufficient permissions"}
        }
        mock_resp.raise_for_status.side_effect = Exception("HTTP 403")
        with pytest.raises(Exception, match="HTTP 403"):
            _raise_for_status_with_plan_check(mock_resp)

    def test_raise_for_status_with_plan_check_passes_non_403(self):
        from synth_ai.sdk.environment_pools import _raise_for_status_with_plan_check

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        # Should not raise
        _raise_for_status_with_plan_check(mock_resp)

    # --- Feature flag override tests ---

    def test_free_plan_with_feature_flag_dict_bool_allowed(self):
        """Free-tier org with feature_flags: {environment_pools: true} should be allowed."""
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": {"environment_pools": True},
        }

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "free"

    def test_free_plan_with_feature_flag_dict_object_allowed(self):
        """Free-tier org with feature_flags: {environment_pools: {enabled: true}} should be allowed."""
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": {
                "environment_pools": {"enabled": True, "reason": "beta tester"}
            },
        }

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "free"

    def test_free_plan_with_feature_flag_list_dict_allowed(self):
        """Free-tier org with feature_flags as list of dicts should be allowed."""
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": [
                {"feature": "environment_pools", "enabled": True},
            ],
        }

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "free"

    def test_free_plan_with_feature_flag_list_string_allowed(self):
        """Free-tier org with feature_flags as list of strings should be allowed."""
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": ["environment_pools", "other_feature"],
        }

        with patch("httpx.get", return_value=mock_resp):
            result = _check_plan_access(
                api_key="sk_test", backend_base="https://api.example.com"
            )
            assert result["plan"] == "free"

    def test_free_plan_with_feature_flag_disabled_rejected(self):
        """Free-tier org with environment_pools flag set to false should be rejected."""
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": {"environment_pools": False},
        }

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError):
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )

    def test_free_plan_with_wrong_feature_flag_rejected(self):
        """Free-tier org flagged for a different feature should still be rejected."""
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": {"other_feature": True},
        }

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError):
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )

    def test_free_plan_no_feature_flags_rejected(self):
        """Free-tier org with empty feature_flags should be rejected."""
        from synth_ai.core.errors import PlanGatingError
        from synth_ai.sdk.environment_pools import _check_plan_access

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "plan": "free",
            "org_id": "org-1",
            "feature_flags": {},
        }

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(PlanGatingError):
                _check_plan_access(
                    api_key="sk_test", backend_base="https://api.example.com"
                )
