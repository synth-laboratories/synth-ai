"""Unit tests for verifier/rubric configuration validation."""

import pytest
from synth_ai.sdk.api.train.validation import (
    InvalidRubricConfigError,
    InvalidVerifierConfigError,
    VerifierConfig,
    extract_and_validate_verifier_rubric,
    validate_rubric_config,
    validate_verifier_config,
)


class TestRubricValidation:
    """Test rubric configuration validation."""

    def test_default_rubric_disabled(self):
        """Empty config should default to disabled."""
        config = validate_rubric_config({})
        assert config.enabled is False
        assert config.weights.env == 1.0
        assert config.weights.event == 0.0
        assert config.weights.outcome == 0.0

    def test_valid_rubric_config(self):
        """Valid rubric config should parse correctly."""
        config = validate_rubric_config(
            {
                "enabled": True,
                "weights": {
                    "env": 0.2,
                    "event": 0.4,
                    "outcome": 0.4,
                },
            }
        )
        assert config.enabled is True
        assert config.weights.env == 0.2
        assert config.weights.event == 0.4
        assert config.weights.outcome == 0.4

    def test_rubric_weights_sum_zero_fails(self):
        """All zero weights should fail validation."""
        with pytest.raises(InvalidRubricConfigError, match="(?i)at least one"):
            validate_rubric_config(
                {
                    "enabled": True,
                    "weights": {
                        "env": 0.0,
                        "event": 0.0,
                        "outcome": 0.0,
                    },
                }
            )

    def test_rubric_negative_weight_fails(self):
        """Negative weights should fail validation."""
        with pytest.raises(InvalidRubricConfigError):
            validate_rubric_config(
                {
                    "enabled": True,
                    "weights": {
                        "env": -0.1,
                        "event": 0.5,
                        "outcome": 0.5,
                    },
                }
            )

    def test_deprecated_rubric_fields_rejected(self):
        """Deprecated fields should be rejected."""
        with pytest.raises(InvalidRubricConfigError, match="deprecated"):
            validate_rubric_config(
                {
                    "enabled": True,
                    "model": "openai/gpt-oss-120b",  # Deprecated
                    "weights": {"env": 1.0},
                }
            )

    def test_deprecated_rubric_event_section_rejected(self):
        """Deprecated [rubric.event] section should be rejected."""
        with pytest.raises(InvalidRubricConfigError, match="rubric.event"):
            validate_rubric_config(
                {
                    "enabled": True,
                    "weights": {"env": 1.0},
                    "event": {  # Deprecated section
                        "rubric_id": "crafter/event@v1",
                        "criteria": [],
                    },
                }
            )


class TestVerifierValidation:
    """Test verifier configuration validation."""

    def test_empty_verifier_config_returns_none(self):
        """Empty config should return None."""
        config = validate_verifier_config({})
        assert config is None

    def test_valid_verifier_config(self):
        """Valid verifier config should parse correctly."""
        config = validate_verifier_config(
            {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "rubric_id": "task@v1",
                    "event": True,
                    "outcome": True,
                },
            }
        )
        assert config is not None
        assert config.options.provider == "openai"
        assert config.options.model == "gpt-5"
        assert config.options.rubric_id == "task@v1"
        assert config.options.event is True
        assert config.options.outcome is True

    def test_verifier_missing_options_fails(self):
        """Verifier config without options should fail."""
        with pytest.raises(InvalidVerifierConfigError, match="options.*required"):
            validate_verifier_config({"other": "value"})

    def test_verifier_invalid_provider_fails(self):
        """Invalid provider should fail validation."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config(
                {
                    "options": {
                        "provider": "invalid_provider",
                        "model": "gpt-5",
                    },
                }
            )

    def test_verifier_both_disabled_fails(self):
        """Both event and outcome disabled should fail."""
        with pytest.raises(InvalidVerifierConfigError, match="(?i)at least one"):
            validate_verifier_config(
                {
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                        "event": False,
                        "outcome": False,
                    },
                }
            )

    def test_deprecated_verifier_type_rejected(self):
        """Deprecated verifier.type should be rejected."""
        with pytest.raises(InvalidVerifierConfigError, match="deprecated"):
            validate_verifier_config(
                {
                    "type": "groq",  # Deprecated
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                    },
                }
            )

    def test_deprecated_max_concurrency_rejected(self):
        """Deprecated max_concurrency should be rejected."""
        with pytest.raises(InvalidVerifierConfigError, match="deprecated"):
            validate_verifier_config(
                {
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                        "max_concurrency": 10,  # Deprecated
                    },
                }
            )

    def test_timeout_rejected(self):
        """Top-level verifier.timeout_s should be rejected."""
        with pytest.raises(InvalidVerifierConfigError, match="deprecated"):
            validate_verifier_config(
                {
                    "timeout_s": 60,  # Deprecated location
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                    },
                }
            )


class TestVerifierRubricIntegration:
    """Test integrated verifier/rubric validation."""

    def test_extract_both_valid(self):
        """Both rubric and verifier valid should parse."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": True,
                    "outcome": True,
                },
            },
        }
        rubric, verifier = extract_and_validate_verifier_rubric(toml_config)
        assert rubric.enabled is True
        assert verifier is not None

    def test_rubric_enabled_without_verifier_rejected(self):
        """Rubric enabled but no verifier should be rejected."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
        }
        with pytest.raises(InvalidVerifierConfigError, match="requires a \\[verifier\\]"):
            extract_and_validate_verifier_rubric(toml_config)

    def test_event_weight_without_event_verification_rejected(self):
        """Event weight > 0 but event verification disabled should be rejected."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": False,  # Disabled
                    "outcome": True,
                },
            },
        }
        with pytest.raises(InvalidVerifierConfigError, match=r"event.*requires"):
            extract_and_validate_verifier_rubric(toml_config)

    def test_outcome_weight_without_outcome_verification_rejected(self):
        """Outcome weight > 0 but outcome verification disabled should be rejected."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": True,
                    "outcome": False,  # Disabled
                },
            },
        }
        with pytest.raises(InvalidVerifierConfigError, match=r"outcome.*requires"):
            extract_and_validate_verifier_rubric(toml_config)


class TestBuildHTTPOptions:
    """Test building HTTP request options."""

    def test_build_minimal_options(self):
        """Minimal options should build correctly."""
        from synth_ai.sdk.api.train.validation import build_verifier_http_options

        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
            }
        )
        options = build_verifier_http_options(config.options)

        assert options["provider"] == "openai"
        assert options["model"] == "gpt-5"
        assert options["event"] is True  # Default
        assert options["outcome"] is True  # Default
        assert "rubric_id" not in options  # Not present
        assert "timeout_s" not in options  # Not present

    def test_build_full_options(self):
        """Full options should build correctly."""
        from synth_ai.sdk.api.train.validation import build_verifier_http_options

        config = VerifierConfig(
            options={
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "rubric_id": "crafter/bundle@v1",
                "event": False,
                "outcome": True,
                "timeout_s": 60.0,
                "metadata": {"async": True},
                "rubric_overrides": {"event": {}},
            }
        )
        options = build_verifier_http_options(config.options)

        assert options["provider"] == "groq"
        assert options["model"] == "openai/gpt-oss-120b"
        assert options["rubric_id"] == "crafter/bundle@v1"
        assert options["event"] is False
        assert options["outcome"] is True
        assert options["timeout_s"] == 60.0
        assert options["metadata"] == {"async": True}
        assert options["rubric_overrides"] == {"event": {}}

    def test_task_info_overrides_static(self):
        """TaskInfo overrides should take priority over static config."""
        from synth_ai.sdk.api.train.validation import build_verifier_http_options

        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": {"event": {"static": True}},
            }
        )

        task_info_overrides = {"event": {"dynamic": True}}
        options = build_verifier_http_options(
            config.options,
            rubric_overrides_from_task_info=task_info_overrides,
        )

        # TaskInfo overrides should replace static config
        assert options["rubric_overrides"] == {"event": {"dynamic": True}}
