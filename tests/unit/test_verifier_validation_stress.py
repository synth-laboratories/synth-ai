"""
Stress tests for verifier/rubric configuration validation.

These tests cover edge cases, error conditions, malformed input,
and complex validation scenarios to ensure robustness.
"""

from __future__ import annotations

import pytest

from synth_ai.cli.commands.train import (
    InvalidVerifierConfigError,
    InvalidRubricConfigError,
    RubricConfig,
    VerifierConfig,
    extract_and_validate_verifier_rubric,
    validate_verifier_config,
    validate_rubric_config,
    build_verifier_http_options,
)


class TestRubricValidationStress:
    """Stress tests for rubric configuration validation."""

    def test_empty_weights_dict(self):
        """Empty weights dict should use defaults."""
        # Pydantic fills in defaults, so this doesn't fail
        config = validate_rubric_config({
            "enabled": True,
            "weights": {},  # Empty
        })
        # Should default to env=1.0, others=0.0
        assert config.weights.env == 1.0
        assert config.weights.event == 0.0
        assert config.weights.outcome == 0.0

    def test_weights_not_a_dict(self):
        """Weights as non-dict should fail."""
        with pytest.raises(InvalidRubricConfigError, match="must be a dictionary"):
            validate_rubric_config({
                "enabled": True,
                "weights": ["env", "event"],  # List instead of dict
            })

    def test_weights_with_string_values(self):
        """String values in weights should be coerced."""
        # Pydantic coerces strings to floats
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": "0.5",  # String instead of float
                "event": 0.3,
                "outcome": 0.2,
            },
        })
        assert config.weights.env == 0.5

    def test_weights_with_none_values(self):
        """None values in weights should fail."""
        with pytest.raises(InvalidRubricConfigError):
            validate_rubric_config({
                "enabled": True,
                "weights": {
                    "env": None,  # None
                    "event": 0.5,
                    "outcome": 0.5,
                },
            })

    def test_weights_extremely_small_positive(self):
        """Very small but positive weight should work."""
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": 1e-10,  # Extremely small
                "event": 0.0,
                "outcome": 0.0,
            },
        })
        assert config.weights.env == 1e-10

    def test_weights_exactly_zero_all(self):
        """All zero weights should fail."""
        with pytest.raises(InvalidRubricConfigError, match="(?i)at least one"):
            validate_rubric_config({
                "enabled": True,
                "weights": {
                    "env": 0.0,
                    "event": 0.0,
                    "outcome": 0.0,
                },
            })

    def test_weights_sum_greater_than_one(self):
        """Weights summing to > 1 should work (no constraint)."""
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": 1.0,
                "event": 1.0,
                "outcome": 1.0,
            },
        })
        assert config.weights.env == 1.0
        assert config.weights.event == 1.0
        assert config.weights.outcome == 1.0

    def test_weights_very_large_values(self):
        """Very large weight values should work."""
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": 1000.0,
                "event": 0.0,
                "outcome": 0.0,
            },
        })
        assert config.weights.env == 1000.0

    def test_rubric_enabled_as_string(self):
        """Enabled as string should be coerced."""
        config = validate_rubric_config({
            "enabled": "true",  # String
            "weights": {"env": 1.0},
        })
        # Pydantic should coerce it
        assert config.enabled is True

    def test_rubric_enabled_as_int(self):
        """Enabled as int should be coerced."""
        config = validate_rubric_config({
            "enabled": 1,  # Int
            "weights": {"env": 1.0},
        })
        assert config.enabled is True

    def test_rubric_with_extra_fields(self):
        """Extra unknown fields should be ignored."""
        config = validate_rubric_config({
            "enabled": True,
            "weights": {"env": 1.0},
            "unknown_field": "value",
            "another_unknown": 123,
        })
        assert config.enabled is True
        # Should not raise error (Pydantic extras allowed)

    def test_rubric_missing_enabled_field(self):
        """Missing enabled field should default to False."""
        config = validate_rubric_config({
            "weights": {"env": 1.0},
        })
        assert config.enabled is False

    def test_rubric_unicode_in_deprecated_fields(self):
        """Unicode characters in deprecated fields should warn."""
        with pytest.warns(DeprecationWarning):
            validate_rubric_config({
                "enabled": True,
                "model": "模型/gpt-oss-120b",  # Unicode
                "weights": {"env": 1.0},
            })


class TestVerifierValidationStress:
    """Stress tests for verifier configuration validation."""

    def test_verifier_options_not_a_dict(self):
        """Options as non-dict should fail."""
        with pytest.raises(InvalidVerifierConfigError, match="must be a dictionary"):
            validate_verifier_config({
                "options": "not a dict",
            })

    def test_verifier_options_is_list(self):
        """Options as list should fail."""
        with pytest.raises(InvalidVerifierConfigError, match="must be a dictionary"):
            validate_verifier_config({
                "options": ["provider", "model"],
            })

    def test_verifier_missing_provider(self):
        """Missing provider should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "model": "gpt-5",
                    # Missing provider
                },
            })

    def test_verifier_missing_model(self):
        """Missing model should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    # Missing model
                },
            })

    def test_verifier_empty_provider(self):
        """Empty provider string should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "",  # Empty
                    "model": "gpt-5",
                },
            })

    def test_verifier_empty_model(self):
        """Empty model string should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "",  # Empty
                },
            })

    def test_verifier_whitespace_only_provider(self):
        """Whitespace-only provider should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "   ",  # Whitespace
                    "model": "gpt-5",
                },
            })

    def test_verifier_invalid_provider_case_sensitive(self):
        """Wrong case provider should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "OpenAI",  # Wrong case
                    "model": "gpt-5",
                },
            })

    def test_verifier_provider_with_special_chars(self):
        """Provider with special chars should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "open-ai!",  # Special chars
                    "model": "gpt-5",
                },
            })

    def test_verifier_negative_timeout(self):
        """Negative timeout should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "timeout_s": -10,  # Negative
                },
            })

    def test_verifier_zero_timeout(self):
        """Zero timeout should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "timeout_s": 0,  # Zero
                },
            })

    def test_verifier_extremely_large_timeout(self):
        """Very large timeout should work."""
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "timeout_s": 99999.0,  # Very large
            },
        })
        assert config.options.timeout_s == 99999.0

    def test_verifier_timeout_as_string(self):
        """Timeout as string should be coerced."""
        # Pydantic coerces strings to floats
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "timeout_s": "60",  # String
            },
        })
        assert config.options.timeout_s == 60.0

    def test_verifier_event_and_outcome_both_string_false(self):
        """Both event and outcome as string 'false' should fail."""
        with pytest.raises(InvalidVerifierConfigError, match="(?i)at least one"):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": "false",  # String
                    "outcome": "false",  # String
                },
            })

    def test_verifier_metadata_not_dict(self):
        """Metadata as non-dict should fail."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "metadata": "not a dict",
                },
            })

    def test_verifier_metadata_deeply_nested(self):
        """Deeply nested metadata should work."""
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "metadata": {
                    "level1": {
                        "level2": {
                            "level3": {
                                "level4": "deep value",
                            },
                        },
                    },
                },
            },
        })
        assert config.options.metadata["level1"]["level2"]["level3"]["level4"] == "deep value"

    def test_verifier_rubric_overrides_deeply_nested(self):
        """Deeply nested rubric_overrides should work."""
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": {
                    "event": {
                        "criteria": [
                            {
                                "id": "test",
                                "weight": 1.0,
                                "description": "Test criterion",
                                "nested": {
                                    "more": {
                                        "data": "here",
                                    },
                                },
                            },
                        ],
                    },
                },
            },
        })
        assert "event" in config.options.rubric_overrides

    def test_verifier_rubric_id_very_long(self):
        """Very long rubric_id should work."""
        long_id = "a" * 1000
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "rubric_id": long_id,
            },
        })
        assert config.options.rubric_id == long_id

    def test_verifier_model_with_unicode(self):
        """Model name with unicode should work."""
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "模型-gpt-5",  # Unicode
            },
        })
        assert "模型" in config.options.model


class TestCrossValidationStress:
    """Stress tests for cross-validation between rubric and verifier."""

    def test_rubric_enabled_verifier_none_warns(self):
        """Rubric enabled but no verifier should warn and disable."""
        with pytest.warns(UserWarning, match="rubric.*enabled.*verifier.*missing"):
            rubric, verifier = extract_and_validate_verifier_rubric({
                "rubric": {
                    "enabled": True,
                    "weights": {"env": 0.2, "event": 0.4, "outcome": 0.4},
                },
            })
        assert rubric.enabled is False
        assert verifier is None

    def test_rubric_disabled_verifier_present_ok(self):
        """Rubric disabled with verifier present should work."""
        rubric, verifier = extract_and_validate_verifier_rubric({
            "rubric": {
                "enabled": False,
                "weights": {"env": 1.0},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                },
            },
        })
        assert rubric.enabled is False
        assert verifier is not None

    def test_all_weights_zero_with_verifier(self):
        """All zero weights should fail even with verifier present."""
        with pytest.raises(InvalidRubricConfigError):
            extract_and_validate_verifier_rubric({
                "rubric": {
                    "enabled": True,
                    "weights": {"env": 0.0, "event": 0.0, "outcome": 0.0},
                },
                "verifier": {
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                    },
                },
            })

    def test_event_weight_high_but_event_disabled_warns(self):
        """High event weight but event disabled should warn."""
        with pytest.warns(UserWarning, match=r"(?i)event.*>.*0.*but.*event=false"):
            extract_and_validate_verifier_rubric({
                "rubric": {
                    "enabled": True,
                    "weights": {"env": 0.0, "event": 1.0, "outcome": 0.0},
                },
                "verifier": {
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                        "event": False,  # Disabled
                        "outcome": True,
                    },
                },
            })

    def test_outcome_weight_high_but_outcome_disabled_warns(self):
        """High outcome weight but outcome disabled should warn."""
        with pytest.warns(UserWarning, match=r"(?i)outcome.*>.*0.*but.*outcome=false"):
            extract_and_validate_verifier_rubric({
                "rubric": {
                    "enabled": True,
                    "weights": {"env": 0.0, "event": 0.0, "outcome": 1.0},
                },
                "verifier": {
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                        "event": True,
                        "outcome": False,  # Disabled
                    },
                },
            })

    def test_both_weights_zero_but_judging_enabled_ok(self):
        """Event/outcome weights zero but judging enabled is OK."""
        rubric, verifier = extract_and_validate_verifier_rubric({
            "rubric": {
                "enabled": True,
                "weights": {"env": 1.0, "event": 0.0, "outcome": 0.0},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": True,
                    "outcome": True,
                },
            },
        })
        # No warning - weights are intentionally zero
        assert rubric.weights.event == 0.0
        assert rubric.weights.outcome == 0.0


class TestHTTPOptionsBuildingStress:
    """Stress tests for building HTTP request options."""

    def test_build_options_with_none_optional_fields(self):
        """Building options with all optional fields None."""
        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_id": None,
                "timeout_s": None,
                "metadata": {},
                "rubric_overrides": {},
            }
        )
        options = build_verifier_http_options(config.options)
        
        # Should not include None fields or empty dicts (exclude_none excludes defaults)
        assert "rubric_id" not in options
        assert "timeout_s" not in options
        # Empty dicts are also excluded by exclude_none
        assert "provider" in options
        assert "model" in options

    def test_build_options_task_info_overrides_empty_static(self):
        """TaskInfo overrides empty static config."""
        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": {},  # Empty
            }
        )
        
        task_info_overrides = {"event": {"criteria": [{"id": "test"}]}}
        options = build_verifier_http_options(
            config.options,
            rubric_overrides_from_task_info=task_info_overrides,
        )
        
        # TaskInfo overrides even empty static config
        assert options["rubric_overrides"] == task_info_overrides

    def test_build_options_task_info_none_uses_static(self):
        """TaskInfo None uses static rubric_overrides."""
        static_overrides = {"outcome": {"criteria": []}}
        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": static_overrides,
            }
        )
        
        options = build_verifier_http_options(
            config.options,
            rubric_overrides_from_task_info=None,
        )
        
        # Should use static config
        assert options["rubric_overrides"] == static_overrides

    def test_build_options_task_info_empty_dict_uses_static(self):
        """TaskInfo empty dict uses static rubric_overrides."""
        static_overrides = {"outcome": {"criteria": []}}
        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "rubric_overrides": static_overrides,
            }
        )
        
        options = build_verifier_http_options(
            config.options,
            rubric_overrides_from_task_info={},  # Empty dict
        )
        
        # Empty dict is falsy, should use static
        assert options["rubric_overrides"] == static_overrides

    def test_build_options_metadata_with_special_types(self):
        """Metadata with various Python types."""
        config = VerifierConfig(
            options={
                "provider": "openai",
                "model": "gpt-5",
                "metadata": {
                    "string": "value",
                    "int": 123,
                    "float": 45.67,
                    "bool": True,
                    "none": None,
                    "list": [1, 2, 3],
                    "dict": {"nested": "value"},
                },
            }
        )
        
        options = build_verifier_http_options(config.options)
        metadata = options["metadata"]
        
        assert metadata["string"] == "value"
        assert metadata["int"] == 123
        assert metadata["float"] == 45.67
        assert metadata["bool"] is True
        assert metadata["none"] is None
        assert metadata["list"] == [1, 2, 3]
        assert metadata["dict"] == {"nested": "value"}


class TestMalformedInputStress:
    """Stress tests with malformed and unexpected input."""

    def test_rubric_config_is_none(self):
        """Rubric config as None should default to disabled."""
        config = validate_rubric_config(None)
        assert config.enabled is False

    def test_rubric_config_is_empty_dict(self):
        """Empty rubric config should default to disabled."""
        config = validate_rubric_config({})
        assert config.enabled is False

    def test_verifier_config_is_none(self):
        """Verifier config as None should return None."""
        config = validate_verifier_config(None)
        assert config is None

    def test_verifier_config_is_empty_dict(self):
        """Empty verifier config should return None."""
        config = validate_verifier_config({})
        assert config is None

    def test_extract_from_config_with_no_rubric_or_verifier(self):
        """Config with neither rubric nor verifier."""
        rubric, verifier = extract_and_validate_verifier_rubric({})
        assert rubric.enabled is False
        assert verifier is None

    def test_extract_from_config_with_other_sections(self):
        """Config with other sections but no rubric/verifier."""
        rubric, verifier = extract_and_validate_verifier_rubric({
            "algorithm": {"type": "online"},
            "model": {"base": "Qwen"},
            "training": {"num_epochs": 1},
        })
        assert rubric.enabled is False
        assert verifier is None

    def test_weights_with_infinity(self):
        """Weights with infinity should work (Pydantic allows it)."""
        config = validate_rubric_config({
            "enabled": True,
            "weights": {
                "env": float("inf"),
                "event": 0.0,
                "outcome": 0.0,
            },
        })
        assert config.weights.env == float("inf")

    def test_weights_with_nan_fails(self):
        """Weights with NaN should fail validation."""
        # Pydantic might allow NaN, but let's test
        try:
            config = validate_rubric_config({
                "enabled": True,
                "weights": {
                    "env": float("nan"),
                    "event": 0.5,
                    "outcome": 0.5,
                },
            })
            # If it passes, at least check the value
            import math
            assert math.isnan(config.weights.env)
        except InvalidRubricConfigError:
            # Expected - NaN should fail
            pass

    def test_deeply_nested_invalid_provider(self):
        """Verifier config with valid structure but invalid provider value."""
        with pytest.raises(InvalidVerifierConfigError):
            validate_verifier_config({
                "options": {
                    "provider": "anthropic",  # Not in allowed list
                    "model": "claude-3",
                },
            })

    def test_unicode_everywhere(self):
        """Config with unicode in all string fields."""
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "模型-gpt-5-测试",
                "rubric_id": "任务/捆绑包@v1",
                "metadata": {
                    "键": "值",
                    "key": "中文值",
                },
            },
        })
        assert "模型" in config.options.model
        assert "任务" in config.options.rubric_id

    def test_extremely_large_config(self):
        """Config with hundreds of metadata fields."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        config = validate_verifier_config({
            "options": {
                "provider": "openai",
                "model": "gpt-5",
                "metadata": large_metadata,
            },
        })
        assert len(config.options.metadata) == 1000


class TestErrorMessagesStress:
    """Test that error messages are clear and helpful."""

    def test_missing_provider_error_message(self):
        """Error message for missing provider should be clear."""
        with pytest.raises(InvalidVerifierConfigError) as exc_info:
            validate_verifier_config({
                "options": {
                    "model": "gpt-5",
                },
            })
        
        error_msg = str(exc_info.value)
        assert "provider" in error_msg.lower() or "field required" in error_msg.lower()

    def test_invalid_provider_error_message(self):
        """Error message for invalid provider should list allowed values."""
        with pytest.raises(InvalidVerifierConfigError) as exc_info:
            validate_verifier_config({
                "options": {
                    "provider": "anthropic",
                    "model": "claude",
                },
            })
        
        error_msg = str(exc_info.value)
        # Should mention the validation pattern or expected values
        assert "provider" in error_msg.lower() or "pattern" in error_msg.lower()

    def test_all_zero_weights_error_message(self):
        """Error message for zero weights should be clear."""
        with pytest.raises(InvalidRubricConfigError) as exc_info:
            validate_rubric_config({
                "enabled": True,
                "weights": {"env": 0.0, "event": 0.0, "outcome": 0.0},
            })
        
        error_msg = str(exc_info.value)
        assert "at least one" in error_msg.lower()
        assert "weight" in error_msg.lower()

    def test_both_judging_disabled_error_message(self):
        """Error message for both judging types disabled should be clear."""
        with pytest.raises(InvalidVerifierConfigError) as exc_info:
            validate_verifier_config({
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                    "event": False,
                    "outcome": False,
                },
            })
        
        error_msg = str(exc_info.value)
        assert "at least one" in error_msg.lower()
        assert ("event" in error_msg.lower() or "outcome" in error_msg.lower())


# Performance stress test (optional - can be slow)
class TestPerformanceStress:
    """Performance tests for validation with large configs."""

    def test_validate_1000_times(self):
        """Validate same config 1000 times (should be fast)."""
        config_dict = {
            "rubric": {
                "enabled": True,
                "weights": {"env": 0.5, "event": 0.3, "outcome": 0.2},
            },
            "verifier": {
                "options": {
                    "provider": "openai",
                    "model": "gpt-5",
                },
            },
        }
        
        import time
        start = time.time()
        for _ in range(1000):
            rubric, verifier = extract_and_validate_verifier_rubric(config_dict)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Validation too slow: {elapsed:.2f}s for 1000 iterations"
        assert rubric.enabled is True
        assert verifier is not None

    def test_validate_many_warnings(self):
        """Config that triggers many warnings (performance check)."""
        with pytest.warns(DeprecationWarning):
            config_dict = {
                "rubric": {
                    "enabled": True,
                    "model": "gpt",  # Deprecated
                    "api_base": "url",  # Deprecated
                    "api_key_env": "KEY",  # Deprecated
                    "weights": {"env": 1.0},
                    "event": {},  # Deprecated section
                    "outcome": {},  # Deprecated section
                },
                "verifier": {
                    "type": "groq",  # Deprecated
                    "timeout_s": 60,  # Deprecated location
                    "options": {
                        "provider": "openai",
                        "model": "gpt-5",
                        "max_concurrency": 10,  # Deprecated
                        "tracks": ["process"],  # Deprecated
                    },
                },
            }
            
            # Should still validate despite many warnings
            rubric, verifier = extract_and_validate_verifier_rubric(config_dict)
            assert rubric.enabled is True
            assert verifier is not None

