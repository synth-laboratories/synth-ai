"""
Test Synth model registry and routing logic.

This test verifies that the model registry correctly identifies Qwen3 models
without requiring API keys or client instantiation.
"""

from synth_ai.lm.core.synth_models import SYNTH_SUPPORTED_MODELS, QWEN3_MODELS, FINE_TUNED_MODELS
import re


class TestSynthModelRegistry:
    """Test Synth model detection and registry."""

    def test_synth_qwen3_models_in_registry(self):
        """Test that Qwen3 models are in the registry."""
        expected_qwen3_models = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-4B-Thinking-2507",
        ]

        for model in expected_qwen3_models:
            assert model in SYNTH_SUPPORTED_MODELS, f"Model {model} should be in SYNTH_SUPPORTED_MODELS"
            assert model in QWEN3_MODELS, f"Model {model} should be in QWEN3_MODELS"

    def test_fine_tuned_models_pattern(self):
        """Test that fine-tuned model pattern is detected correctly."""
        ft_models = [
            "ft:Qwen/Qwen3-4B-Instruct-2507:ftjob-22",
            "ft:Qwen/Qwen3-0.6B:custom-job-123",
            "ft:gpt-4:some-job",
        ]

        pattern = re.compile(r"^ft:.*$")
        for model in ft_models:
            assert pattern.match(model), f"Fine-tuned model {model} should match ft: pattern"

    def test_qwen3_regex_pattern(self):
        """Test that Qwen3 regex pattern works correctly."""
        qwen3_pattern = re.compile(r"^Qwen/Qwen3.*$")

        # Should match
        matching_models = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-32B-large",
        ]

        # Should NOT match
        non_matching_models = [
            "Qwen/Qwen2-7B",  # Qwen2, not Qwen3
            "Qwen/Qwen2.5-14B",  # Qwen2.5, not Qwen3
            "Meta/Llama-3.1-8B",  # Different provider
            "gpt-4",  # No slash
        ]

        for model in matching_models:
            assert qwen3_pattern.match(model), f"Model {model} should match Qwen3 pattern"

        for model in non_matching_models:
            assert not qwen3_pattern.match(model), f"Model {model} should NOT match Qwen3 pattern"

    def test_registry_consistency(self):
        """Test that registry is properly constructed."""
        # Registry should contain all Qwen3 models
        for model in QWEN3_MODELS:
            assert model in SYNTH_SUPPORTED_MODELS, f"Qwen3 model {model} should be in main registry"

        # Registry should contain all fine-tuned models
        for model in FINE_TUNED_MODELS:
            assert model in SYNTH_SUPPORTED_MODELS, f"Fine-tuned model {model} should be in main registry"

        # Registry size should be reasonable
        expected_min_size = len(QWEN3_MODELS) + len(FINE_TUNED_MODELS)
        assert len(SYNTH_SUPPORTED_MODELS) >= expected_min_size, "Registry should contain at least expected models"
