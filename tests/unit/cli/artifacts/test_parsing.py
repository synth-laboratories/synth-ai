"""Unit tests for artifact ID parsing utilities.

Comprehensive test coverage for all parsing edge cases, error conditions,
and failure modes.
"""

from __future__ import annotations

import pytest

from synth_ai.cli.commands.artifacts.parsing import (
    ParsedModelId,
    ParsedPromptId,
    detect_artifact_type,
    is_model_id,
    is_prompt_id,
    parse_model_id,
    parse_prompt_id,
    resolve_wasabi_key_for_model,
    validate_model_id,
    validate_prompt_id,
)


class TestModelIDParsing:
    """Test model ID parsing."""

    def test_parse_finetuned_model_id(self) -> None:
        """Test parsing fine-tuned model ID."""
        model_id = "ft:Qwen/Qwen3-0.6B:job_12345"
        parsed = parse_model_id(model_id)
        assert parsed.prefix == "ft"
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job_12345"
        assert parsed.full_id == model_id

    def test_parse_rl_model_id(self) -> None:
        """Test parsing RL model ID."""
        model_id = "rl:Qwen/Qwen3-0.6B:job_67890"
        parsed = parse_model_id(model_id)
        assert parsed.prefix == "rl"
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job_67890"

    def test_parse_peft_model_id(self) -> None:
        """Test parsing PEFT model ID."""
        model_id = "peft:Qwen/Qwen3-0.6B:job_abc123"
        parsed = parse_model_id(model_id)
        assert parsed.prefix == "peft"
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job_abc123"

    def test_parse_model_id_invalid_format(self) -> None:
        """Test parsing invalid model ID format."""
        with pytest.raises(ValueError, match="Unsupported model ID prefix"):
            parse_model_id("invalid_format")

    def test_parse_model_id_missing_parts(self) -> None:
        """Test parsing model ID with missing parts."""
        with pytest.raises(ValueError, match="Invalid ft: model ID format"):
            parse_model_id("ft:Qwen/Qwen3-0.6B")

    def test_is_model_id(self) -> None:
        """Test model ID detection."""
        assert is_model_id("ft:Qwen/Qwen3-0.6B:job_12345") is True
        assert is_model_id("rl:Qwen/Qwen3-0.6B:job_12345") is True
        assert is_model_id("peft:Qwen/Qwen3-0.6B:job_12345") is True
        assert is_model_id("pl_12345") is False
        assert is_model_id("invalid") is False

    def test_validate_model_id(self) -> None:
        """Test model ID validation."""
        assert validate_model_id("ft:Qwen/Qwen3-0.6B:job_12345") is True
        assert validate_model_id("invalid") is False


class TestPromptIDParsing:
    """Test prompt ID parsing."""

    def test_parse_prompt_id(self) -> None:
        """Test parsing prompt ID."""
        prompt_id = "pl_71c12c4c7c474c34"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_71c12c4c7c474c34"
        assert parsed.full_id == prompt_id

    def test_parse_prompt_id_with_prefix(self) -> None:
        """Test parsing prompt ID with pl_ prefix."""
        prompt_id = "pl_abc123def456"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_abc123def456"

    def test_parse_prompt_id_accepts_any_string(self) -> None:
        """Test that parse_prompt_id accepts any string (validation happens at backend)."""
        # parse_prompt_id doesn't raise for invalid formats - it accepts any string
        # and lets the backend validate
        parsed = parse_prompt_id("invalid_format")
        assert parsed.job_id == "invalid_format"
        assert parsed.full_id == "invalid_format"

    def test_is_prompt_id(self) -> None:
        """Test prompt ID detection."""
        assert is_prompt_id("pl_71c12c4c7c474c34") is True
        assert is_prompt_id("pl_abc123") is True
        assert is_prompt_id("ft:Qwen/Qwen3-0.6B:job_12345") is False
        assert is_prompt_id("invalid") is False

    def test_validate_prompt_id(self) -> None:
        """Test prompt ID validation."""
        # validate_prompt_id returns True for any string since parse_prompt_id doesn't raise
        assert validate_prompt_id("pl_71c12c4c7c474c34") is True
        assert validate_prompt_id("invalid") is True  # parse_prompt_id accepts any string


class TestArtifactTypeDetection:
    """Test artifact type detection."""

    def test_detect_model_type(self) -> None:
        """Test detecting model type."""
        assert detect_artifact_type("ft:Qwen/Qwen3-0.6B:job_12345") == "model"
        assert detect_artifact_type("rl:Qwen/Qwen3-0.6B:job_12345") == "model"
        assert detect_artifact_type("peft:Qwen/Qwen3-0.6B:job_12345") == "model"

    def test_detect_prompt_type(self) -> None:
        """Test detecting prompt type."""
        assert detect_artifact_type("pl_71c12c4c7c474c34") == "prompt"
        assert detect_artifact_type("job_pl_71c12c4c7c474c34") == "prompt"

    def test_detect_unknown_type(self) -> None:
        """Test detecting unknown type."""
        assert detect_artifact_type("unknown_format") == "unknown"
        assert detect_artifact_type("") == "unknown"
        assert detect_artifact_type("   ") == "unknown"

    def test_detect_with_whitespace(self) -> None:
        """Test detection handles whitespace."""
        assert detect_artifact_type("  ft:Qwen/Qwen3-0.6B:job_12345  ") == "model"
        assert detect_artifact_type("  pl_71c12c4c7c474c34  ") == "prompt"

    def test_detect_case_sensitivity(self) -> None:
        """Test detection is case-sensitive."""
        assert detect_artifact_type("FT:Qwen/Qwen3-0.6B:job_12345") == "unknown"  # Case sensitive
        assert detect_artifact_type("PL_71c12c4c7c474c34") == "unknown"  # Case sensitive


class TestModelIDEdgeCases:
    """Test model ID parsing edge cases and failure modes."""

    def test_model_id_with_special_chars_in_base_model(self) -> None:
        """Test model ID with special characters in base model name."""
        # Colons, slashes, dots are valid in base model names
        model_id = "ft:Qwen/Qwen3-0.6B-Instruct:job_12345"
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "Qwen/Qwen3-0.6B-Instruct"
        assert parsed.job_id == "job_12345"

    def test_model_id_with_colons_in_job_id(self) -> None:
        """Test model ID with colons in job ID."""
        model_id = "ft:Qwen/Qwen3-0.6B:job:with:colons"
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job:with:colons"

    def test_model_id_with_empty_base_model(self) -> None:
        """Test model ID with empty base model."""
        # Empty base model is actually parsed (parts[1] is empty string)
        # The validation happens at backend level
        parsed = parse_model_id("ft::job_12345")
        assert parsed.base_model == ""
        assert parsed.job_id == "job_12345"

    def test_model_id_with_empty_job_id(self) -> None:
        """Test model ID with empty job ID."""
        # Empty job ID is actually parsed (parts[2] is empty string)
        # The validation happens at backend level
        parsed = parse_model_id("ft:Qwen/Qwen3-0.6B:")
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == ""

    def test_model_id_with_only_prefix(self) -> None:
        """Test model ID with only prefix."""
        with pytest.raises(ValueError):
            parse_model_id("ft:")

    def test_model_id_with_whitespace(self) -> None:
        """Test model ID with whitespace."""
        model_id = "  ft:Qwen/Qwen3-0.6B:job_12345  "
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job_12345"

    def test_model_id_with_newlines(self) -> None:
        """Test model ID with newlines."""
        model_id = "ft:Qwen/Qwen3-0.6B:job_12345\n"
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job_12345"

    def test_model_id_very_long(self) -> None:
        """Test very long model ID."""
        long_base = "A" * 200
        long_job = "B" * 200
        model_id = f"ft:{long_base}:{long_job}"
        parsed = parse_model_id(model_id)
        assert len(parsed.base_model) == 200
        assert len(parsed.job_id) == 200

    def test_model_id_unicode_characters(self) -> None:
        """Test model ID with unicode characters."""
        model_id = "ft:模型/模型-0.6B:工作_12345"
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "模型/模型-0.6B"
        assert parsed.job_id == "工作_12345"

    def test_model_id_none_input(self) -> None:
        """Test model ID parsing with None input."""
        with pytest.raises(ValueError, match="Invalid model_id"):
            parse_model_id(None)  # type: ignore

    def test_model_id_empty_string(self) -> None:
        """Test model ID parsing with empty string."""
        with pytest.raises(ValueError, match="Invalid model_id"):
            parse_model_id("")

    def test_model_id_non_string_input(self) -> None:
        """Test model ID parsing with non-string input."""
        with pytest.raises(ValueError, match="Invalid model_id"):
            parse_model_id(12345)  # type: ignore
        with pytest.raises(ValueError, match="Invalid model_id"):
            parse_model_id([])  # type: ignore
        with pytest.raises(ValueError, match="Invalid model_id"):
            parse_model_id({})  # type: ignore

    def test_model_id_partial_match(self) -> None:
        """Test model ID that partially matches prefix."""
        # Should not match "ft" if it's part of a larger word
        with pytest.raises(ValueError, match="Unsupported model ID prefix"):
            parse_model_id("shift:Qwen/Qwen3-0.6B:job_12345")

    def test_model_id_multiple_colons(self) -> None:
        """Test model ID with multiple colons."""
        model_id = "ft:Qwen/Qwen3-0.6B:job:with:many:colons"
        parsed = parse_model_id(model_id)
        assert parsed.base_model == "Qwen/Qwen3-0.6B"
        assert parsed.job_id == "job:with:many:colons"

    def test_model_id_single_colon(self) -> None:
        """Test model ID with only one colon."""
        with pytest.raises(ValueError, match="Invalid ft: model ID format"):
            parse_model_id("ft:Qwen/Qwen3-0.6B")

    def test_model_id_no_colons(self) -> None:
        """Test model ID with no colons."""
        with pytest.raises(ValueError, match="Unsupported model ID prefix"):
            parse_model_id("ftQwen/Qwen3-0.6Bjob_12345")


class TestPromptIDEdgeCases:
    """Test prompt ID parsing edge cases and failure modes."""

    def test_prompt_id_with_underscores(self) -> None:
        """Test prompt ID with multiple underscores."""
        prompt_id = "pl_abc_123_def_456"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_abc_123_def_456"

    def test_prompt_id_very_long(self) -> None:
        """Test very long prompt ID."""
        long_id = "pl_" + "a" * 500
        parsed = parse_prompt_id(long_id)
        assert len(parsed.job_id) == 503

    def test_prompt_id_with_whitespace(self) -> None:
        """Test prompt ID with whitespace."""
        prompt_id = "  pl_71c12c4c7c474c34  "
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_71c12c4c7c474c34"

    def test_prompt_id_with_newlines(self) -> None:
        """Test prompt ID with newlines."""
        prompt_id = "pl_71c12c4c7c474c34\n"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_71c12c4c7c474c34"

    def test_prompt_id_unicode_characters(self) -> None:
        """Test prompt ID with unicode characters."""
        prompt_id = "pl_工作_12345"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_工作_12345"

    def test_prompt_id_none_input(self) -> None:
        """Test prompt ID parsing with None input."""
        with pytest.raises(ValueError, match="Invalid prompt_id"):
            parse_prompt_id(None)  # type: ignore

    def test_prompt_id_empty_string(self) -> None:
        """Test prompt ID parsing with empty string."""
        with pytest.raises(ValueError, match="Invalid prompt_id"):
            parse_prompt_id("")

    def test_prompt_id_non_string_input(self) -> None:
        """Test prompt ID parsing with non-string input."""
        with pytest.raises(ValueError, match="Invalid prompt_id"):
            parse_prompt_id(12345)  # type: ignore
        with pytest.raises(ValueError, match="Invalid prompt_id"):
            parse_prompt_id([])  # type: ignore
        with pytest.raises(ValueError, match="Invalid prompt_id"):
            parse_prompt_id({})  # type: ignore

    def test_prompt_id_bare_job_id(self) -> None:
        """Test prompt ID as bare job ID (no prefix)."""
        # parse_prompt_id accepts bare job IDs
        prompt_id = "job_12345"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "job_12345"

    def test_prompt_id_job_pl_prefix(self) -> None:
        """Test prompt ID with job_pl_ prefix."""
        prompt_id = "job_pl_71c12c4c7c474c34"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "job_pl_71c12c4c7c474c34"

    def test_prompt_id_special_characters(self) -> None:
        """Test prompt ID with special characters."""
        prompt_id = "pl_abc-123_def.456"
        parsed = parse_prompt_id(prompt_id)
        assert parsed.job_id == "pl_abc-123_def.456"


class TestWasabiKeyResolution:
    """Test Wasabi key resolution for models."""

    def test_resolve_wasabi_key_ft_merged(self) -> None:
        """Test resolving Wasabi key for fine-tuned model (merged)."""
        parsed = parse_model_id("ft:Qwen/Qwen3-0.6B:job_12345")
        key = resolve_wasabi_key_for_model(parsed, prefer_merged=True)
        assert key == "models/Qwen-Qwen3-0-6B-job_12345-fp16.tar.gz"

    def test_resolve_wasabi_key_ft_adapter(self) -> None:
        """Test resolving Wasabi key for fine-tuned model (adapter)."""
        parsed = parse_model_id("ft:Qwen/Qwen3-0.6B:job_12345")
        key = resolve_wasabi_key_for_model(parsed, prefer_merged=False)
        assert key == "models/Qwen/Qwen3-0.6B/ft-job_12345/adapter.tar.gz"

    def test_resolve_wasabi_key_rl(self) -> None:
        """Test resolving Wasabi key for RL model."""
        parsed = parse_model_id("rl:Qwen/Qwen3-0.6B:job_12345")
        key = resolve_wasabi_key_for_model(parsed)
        assert key == "models/Qwen-Qwen3-0-6B-job_12345-rl.tar.gz"

    def test_resolve_wasabi_key_peft(self) -> None:
        """Test resolving Wasabi key for PEFT model."""
        parsed = parse_model_id("peft:Qwen/Qwen3-0.6B:job_12345")
        key = resolve_wasabi_key_for_model(parsed, prefer_merged=True)
        assert key == "models/Qwen-Qwen3-0-6B-job_12345-fp16.tar.gz"

    def test_resolve_wasabi_key_with_special_chars(self) -> None:
        """Test resolving Wasabi key with special characters in base model."""
        parsed = parse_model_id("ft:model/name.with:colons:job_12345")
        key = resolve_wasabi_key_for_model(parsed, prefer_merged=True)
        # Special chars in base model should be replaced with dashes for merged format
        # The base_model part gets sanitized: / -> -, . -> -, : -> -
        # Format: models/BASE-JOB_ID-fp16.tar.gz
        # Note: job_id may contain colons, so the full key may have colons
        assert key.startswith("models/model-name-with-colons")
        assert key.endswith("-fp16.tar.gz")
        # Check that base model special chars are sanitized
        # Extract the base model part (everything before the job_id)
        # The base model part should not have slashes or dots
        parts = key.replace("models/", "").replace("-fp16.tar.gz", "").split("-")
        # The base model part is everything before the job_id
        # Since job_id starts with "job_", we can identify it
        base_parts = []
        for part in parts:
            if part.startswith("job_"):
                break
            base_parts.append(part)
        base_part = "-".join(base_parts)
        assert "/" not in base_part
        assert "." not in base_part

    def test_resolve_wasabi_key_invalid_type(self) -> None:
        """Test resolving Wasabi key for invalid model type."""
        # Create a parsed model with invalid prefix (shouldn't happen in practice)
        parsed = ParsedModelId(prefix="invalid", base_model="test", job_id="job", full_id="invalid:test:job")
        with pytest.raises(ValueError, match="Unsupported model type"):
            resolve_wasabi_key_for_model(parsed)


class TestValidationFunctions:
    """Test validation functions with edge cases."""

    def test_validate_model_id_valid_cases(self) -> None:
        """Test validate_model_id with valid cases."""
        assert validate_model_id("ft:Qwen/Qwen3-0.6B:job_12345") is True
        assert validate_model_id("rl:Qwen/Qwen3-0.6B:job_12345") is True
        assert validate_model_id("peft:Qwen/Qwen3-0.6B:job_12345") is True

    def test_validate_model_id_invalid_cases(self) -> None:
        """Test validate_model_id with invalid cases."""
        assert validate_model_id("invalid") is False
        assert validate_model_id("ft:incomplete") is False
        assert validate_model_id("") is False
        assert validate_model_id("   ") is False

    def test_validate_model_id_edge_cases(self) -> None:
        """Test validate_model_id with edge cases."""
        # None and non-string should return False (caught by parse_model_id)
        assert validate_model_id(None) is False  # type: ignore
        assert validate_model_id(12345) is False  # type: ignore

    def test_validate_prompt_id_all_cases(self) -> None:
        """Test validate_prompt_id accepts all strings."""
        # parse_prompt_id doesn't raise, so validate always returns True
        assert validate_prompt_id("pl_71c12c4c7c474c34") is True
        assert validate_prompt_id("invalid") is True
        assert validate_prompt_id("") is False  # Empty string raises ValueError
        assert validate_prompt_id("   ") is True  # Whitespace-only is accepted


class TestDetectionFunctions:
    """Test detection functions with edge cases."""

    def test_is_model_id_valid_cases(self) -> None:
        """Test is_model_id with valid cases."""
        assert is_model_id("ft:Qwen/Qwen3-0.6B:job_12345") is True
        assert is_model_id("rl:Qwen/Qwen3-0.6B:job_12345") is True
        assert is_model_id("peft:Qwen/Qwen3-0.6B:job_12345") is True

    def test_is_model_id_invalid_cases(self) -> None:
        """Test is_model_id with invalid cases."""
        assert is_model_id("pl_71c12c4c7c474c34") is False
        assert is_model_id("invalid") is False
        assert is_model_id("") is False

    def test_is_prompt_id_valid_cases(self) -> None:
        """Test is_prompt_id with valid cases."""
        assert is_prompt_id("pl_71c12c4c7c474c34") is True
        assert is_prompt_id("job_pl_71c12c4c7c474c34") is True

    def test_is_prompt_id_invalid_cases(self) -> None:
        """Test is_prompt_id with invalid cases."""
        assert is_prompt_id("ft:Qwen/Qwen3-0.6B:job_12345") is False
        assert is_prompt_id("invalid") is False
        assert is_prompt_id("") is False

    def test_detection_with_whitespace(self) -> None:
        """Test detection functions handle whitespace."""
        assert is_model_id("  ft:Qwen/Qwen3-0.6B:job_12345  ") is True
        assert is_prompt_id("  pl_71c12c4c7c474c34  ") is True

