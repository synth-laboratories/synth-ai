"""Unit tests for OpenAPI ↔ Pydantic schema validation.

Tests the validation script that ensures task_app.yaml stays in sync
with Pydantic models in synth_ai/task/contracts.py.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

from synth_ai.sdk.task.contracts import (  # noqa: E402
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)


class TestOpenAPIPydanticValidation:
    """Tests for OpenAPI/Pydantic schema synchronization."""

    def test_openapi_spec_exists(self):
        """Test that OpenAPI spec file exists."""
        spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
        assert spec_path.exists(), f"OpenAPI spec not found at {spec_path}"

    def test_openapi_spec_valid_yaml(self):
        """Test that OpenAPI spec is valid YAML."""
        spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        assert isinstance(spec, dict)
        assert "openapi" in spec or "swagger" in spec
        assert "components" in spec
        assert "schemas" in spec["components"]

    def test_required_schemas_in_openapi(self):
        """Test that all required schemas exist in OpenAPI spec."""
        spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        schemas = spec.get("components", {}).get("schemas", {})

        required_schemas = [
            "RolloutRequest",
            "RolloutResponse",
            "RolloutMetrics",
            "TaskInfo",
        ]

        for schema_name in required_schemas:
            assert schema_name in schemas, (
                f"Required schema {schema_name} missing from OpenAPI spec"
            )

    def test_pydantic_models_exist(self):
        """Test that all required Pydantic models exist."""
        models = [
            RolloutRequest,
            RolloutResponse,
            RolloutMetrics,
            TaskInfo,
        ]

        for model in models:
            assert model is not None, f"Pydantic model {model.__name__} not found"

    def test_pydantic_models_have_json_schema(self):
        """Test that Pydantic models can generate JSON schemas."""
        models = [
            RolloutRequest,
            RolloutResponse,
            RolloutMetrics,
            TaskInfo,
        ]

        for model in models:
            schema = model.model_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema or "$defs" in schema

    def test_rollout_request_schema_structure(self):
        """Test that RolloutRequest has expected structure."""
        schema = RolloutRequest.model_json_schema()

        # Check for required fields
        assert "properties" in schema
        props = schema["properties"]

        # RolloutRequest should have these fields
        assert "run_id" in props
        assert "env" in props
        assert "policy" in props
        assert "mode" in props

    def test_rollout_response_schema_structure(self):
        """Test that RolloutResponse has expected structure."""
        schema = RolloutResponse.model_json_schema()

        assert "properties" in schema
        props = schema["properties"]

        # RolloutResponse should have these fields
        assert "run_id" in props
        assert "trace" in props
        assert "metrics" in props

    def test_rollout_metrics_has_outcome_reward(self):
        """Test that RolloutMetrics has outcome_reward field (required)."""
        schema = RolloutMetrics.model_json_schema()

        assert "properties" in schema
        props = schema["properties"]

        # outcome_reward is the required field for scoring
        assert "outcome_reward" in props, "RolloutMetrics must have outcome_reward field"

        # Check that outcome_reward is required
        required = schema.get("required", [])
        assert "outcome_reward" in required, "outcome_reward must be a required field"

    def test_validation_script_runs(self):
        """Test that validation script can be imported and run."""
        script_path = PROJECT_ROOT / "scripts" / "validate_openapi_pydantic.py"
        assert script_path.exists(), f"Validation script not found at {script_path}"

        # Test that script can be imported
        import importlib.util

        spec = importlib.util.spec_from_file_location("validate_openapi_pydantic", script_path)
        assert spec is not None
        assert spec.loader is not None

    def test_validation_script_main_function(self):
        """Test that validation script has main function."""
        script_path = PROJECT_ROOT / "scripts" / "validate_openapi_pydantic.py"

        import importlib.util

        spec = importlib.util.spec_from_file_location("validate_openapi_pydantic", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "main"), "Validation script must have main() function"
        assert callable(module.main), "main() must be callable"

    @pytest.mark.parametrize(
        "model_name",
        [
            "RolloutRequest",
            "RolloutResponse",
            "RolloutMetrics",
            "TaskInfo",
        ],
    )
    def test_schema_model_mapping_exists(self, model_name):
        """Test that schema-to-model mapping exists for all key schemas."""
        script_path = PROJECT_ROOT / "scripts" / "validate_openapi_pydantic.py"

        import importlib.util

        spec = importlib.util.spec_from_file_location("validate_openapi_pydantic", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "SCHEMA_MODEL_MAP"), "Script must have SCHEMA_MODEL_MAP"
        assert model_name in module.SCHEMA_MODEL_MAP, f"{model_name} missing from SCHEMA_MODEL_MAP"

    def test_openapi_spec_has_paths(self):
        """Test that OpenAPI spec defines required endpoints."""
        spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        paths = spec.get("paths", {})

        # Required endpoints
        assert "/health" in paths, "OpenAPI spec must define /health endpoint"
        assert "/rollout" in paths, "OpenAPI spec must define /rollout endpoint"

        # Optional but recommended
        assert "/task_info" in paths or "/info" in paths, (
            "OpenAPI spec should define /task_info or /info endpoint"
        )

    def test_rollout_endpoint_is_post(self):
        """Test that /rollout endpoint uses POST method."""
        spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
        with open(spec_path) as f:
            spec = yaml.safe_load(f)

        paths = spec.get("paths", {})
        rollout_path = paths.get("/rollout", {})

        assert "post" in rollout_path, "/rollout must support POST method"

    def test_contracts_module_importable(self):
        """Test that contracts module can be imported."""
        from synth_ai.contracts import TASK_APP_CONTRACT_PATH, get_task_app_contract

        assert callable(get_task_app_contract)
        assert TASK_APP_CONTRACT_PATH.exists()
