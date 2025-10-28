"""Unit tests for judge/rubric configuration validation."""

from __future__ import annotations

import pytest

from synth_ai.cli.commands.train import (
    InvalidJudgeConfigError,
    InvalidRubricConfigError,
    RubricConfig,
    JudgeConfig,
    extract_and_validate_judge_rubric,
    validate_judge_config,
    validate_rubric_config,
    check_for_deprecated_fields,
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
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": 0.2,
                "event": 0.4,
                "outcome": 0.4,
            },
        })
        assert config.enabled is True
        assert config.weights.env == 0.2
        assert config.weights.event == 0.4
        assert config.weights.outcome == 0.4

    def test_rubric_weights_sum_zero_fails(self):
        """All zero weights should fail validation."""
        with pytest.raises(InvalidRubricConfigError, match="(?i)at least one"):
            validate_rubric_config({
                "enabled": True,
                "weights": {
                    "env": 0.0,
                    "event": 0.0,
                    "outcome": 0.0,
                },
            })

    def test_rubric_negative_weight_fails(self):
        """Negative weights should fail validation."""
        with pytest.raises(InvalidRubricConfigError):
            validate_rubric_config({
                "enabled": True,
                "weights": {
                    "env": -0.1,
                    "event": 0.5,
                    "outcome": 0.5,
                },
            })

    def test_deprecated_rubric_fields_warned(self):
        """Deprecated fields should trigger warnings."""
        with pytest.warns(DeprecationWarning, match="model"):
            validate_rubric_config({
                "enabled": True,
                "model": "openai/gpt-oss-120b",  # Deprecated
                "weights": {"env": 1.0},
            })

    def test_deprecated_rubric_event_section_warned(self):
        """Deprecated [rubric.event] section should trigger warning."""
        with pytest.warns(DeprecationWarning, match="rubric.event"):
            validate_rubric_config({
                "enabled": True,
                "weights": {"env": 1.0},
                "event": {  # Deprecated section
                    "rubric_id": "crafter/event@v1",
                    "criteria": [],
                },
            })


class TestJudgeValidation:
    """Test judge configuration validation."""

    def test_empty_judge_config_returns_none(self):
        """Empty config should return None."""
        config = validate_judge_config({})
        assert config is None

    def test_valid_judge_config(self):
        """Valid judge config should parse correctly."""
        config = validate_judge_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "rubric_id": "task@v1",
                "event": True,
                "outcome": True,
            },
        })
        assert config is not None
        assert config.options.provider == "openai"
        assert config.options.model == "gpt-5"
        assert config.options.rubric_id == "task@v1"
        assert config.options.event is True
        assert config.options.outcome is True

    def test_judge_missing_options_fails(self):
        """Judge config without options should fail."""
        with pytest.raises(InvalidJudgeConfigError, match="options.*required"):
            validate_judge_config({"other": "value"})

    def test_judge_invalid_provider_fails(self):
        """Invalid provider should fail validation."""
        with pytest.raises(InvalidJudgeConfigError):
            validate_judge_config({
                "options": {
                    "provider": "invalid_provider",
                    "model": "gpt-5",
                },
            })

    def test_judge_both_disabled_fails(self):
        """Both event and outcome disabled should fail."""
        with pytest.raises(InvalidJudgeConfigError, match="(?i)at least one"):
            validate_judge_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": False,
                    "outcome": False,
                },
            })

    def test_deprecated_judge_type_warned(self):
        """Deprecated judge.type should trigger warning."""
        with pytest.warns(DeprecationWarning, match="(?i)deprecated.*fields"):
            validate_judge_config({
                "type": "groq",  # Deprecated
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                },
            })

    def test_deprecated_max_concurrency_warned(self):
        """Deprecated max_concurrency should trigger warning."""
        with pytest.warns(DeprecationWarning, match="max_concurrency"):
            validate_judge_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "max_concurrency": 10,  # Deprecated
                },
            })

    def test_timeout_migration(self):
        """judge.timeout_s should auto-migrate to judge.options.timeout_s."""
        with pytest.warns(DeprecationWarning, match="timeout_s"):
            config = validate_judge_config({
                "timeout_s": 60,  # Deprecated location
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                },
            })
        assert config is not None
        assert config.options.timeout_s == 60


class TestJudgeRubricIntegration:
    """Test integrated judge/rubric validation."""

    def test_extract_both_valid(self):
        """Both rubric and judge valid should parse."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "judge": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": True,
                    "outcome": True,
                },
            },
        }
        rubric, judge = extract_and_validate_judge_rubric(toml_config)
        assert rubric.enabled is True
        assert judge is not None

    def test_rubric_enabled_without_judge_warned(self):
        """Rubric enabled but no judge should warn and disable rubric."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
        }
        with pytest.warns(UserWarning, match="rubric.*enabled.*judge.*missing"):
            rubric, judge = extract_and_validate_judge_rubric(toml_config)
        assert rubric.enabled is False
        assert judge is None

    def test_event_weight_without_event_judging_warned(self):
        """Event weight > 0 but event judging disabled should warn."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "judge": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": False,  # Disabled
                    "outcome": True,
                },
            },
        }
        with pytest.warns(UserWarning, match=r"(?i)event.*>.*0.*but.*event=false"):
            extract_and_validate_judge_rubric(toml_config)

    def test_outcome_weight_without_outcome_judging_warned(self):
        """Outcome weight > 0 but outcome judging disabled should warn."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
            },
            "judge": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": True,
                    "outcome": False,  # Disabled
                },
            },
        }
        with pytest.warns(UserWarning, match=r"(?i)outcome.*>.*0.*but.*outcome=false"):
            extract_and_validate_judge_rubric(toml_config)


class TestDeprecatedFieldsChecker:
    """Test deprecated fields detection."""

    def test_no_deprecated_fields(self):
        """Clean config should return empty dict."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 1.0},
            },
            "judge": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                },
            },
        }
        deprecated = check_for_deprecated_fields(toml_config)
        assert deprecated == {}

    def test_deprecated_rubric_fields_detected(self):
        """Deprecated rubric fields should be detected."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "model": "gpt-5",  # Deprecated
                "api_base": "https://api.openai.com",  # Deprecated
                "weights": {"env": 1.0},
            },
        }
        deprecated = check_for_deprecated_fields(toml_config)
        assert "rubric" in deprecated
        assert "model" in deprecated["rubric"]
        assert "api_base" in deprecated["rubric"]

    def test_deprecated_rubric_sections_detected(self):
        """Deprecated rubric sections should be detected."""
        toml_config = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 1.0},
                "event": {"rubric_id": "test"},  # Deprecated section
                "outcome": {"rubric_id": "test"},  # Deprecated section
            },
        }
        deprecated = check_for_deprecated_fields(toml_config)
        assert "rubric" in deprecated
        assert any("event" in field for field in deprecated["rubric"])
        assert any("outcome" in field for field in deprecated["rubric"])

    def test_deprecated_judge_options_detected(self):
        """Deprecated judge.options fields should be detected."""
        toml_config = {
            "judge": {
                "type": "groq",  # Deprecated
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "max_concurrency": 10,  # Deprecated
                    "tracks": ["process"],  # Deprecated
                },
            },
        }
        deprecated = check_for_deprecated_fields(toml_config)
        assert "judge" in deprecated
        assert "type" in deprecated["judge"]
        assert "judge.options" in deprecated
        assert "max_concurrency" in deprecated["judge.options"]
        assert "tracks" in deprecated["judge.options"]


class TestBuildHTTPOptions:
    """Test building HTTP request options."""

    def test_build_minimal_options(self):
        """Minimal options should build correctly."""
        from synth_ai.cli.commands.train import build_judge_http_options
        
        config = JudgeConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
            }
        )
        options = build_judge_http_options(config.options)
        
        assert options["provider"] == "openai"
        assert options["model"] == "gpt-5"
        assert options["event"] is True  # Default
        assert options["outcome"] is True  # Default
        assert "rubric_id" not in options  # Not present
        assert "timeout_s" not in options  # Not present

    def test_build_full_options(self):
        """Full options should build correctly."""
        from synth_ai.cli.commands.train import build_judge_http_options
        
        config = JudgeConfig(
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
        options = build_judge_http_options(config.options)
        
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
        from synth_ai.cli.commands.train import build_judge_http_options
        
        config = JudgeConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": {"event": {"static": True}},
            }
        )
        
        task_info_overrides = {"event": {"dynamic": True}}
        options = build_judge_http_options(
            config.options,
            rubric_overrides_from_task_info=task_info_overrides,
        )
        
        # TaskInfo overrides should replace static config
        assert options["rubric_overrides"] == {"event": {"dynamic": True}}

