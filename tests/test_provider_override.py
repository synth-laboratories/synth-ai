import os
import pytest
from synth_ai.lm.core.main import LM
from synth_ai.lm.core.vendor_clients import get_client
from synth_ai.lm.core.all import (
    OpenAIStructuredOutputClient,
    AnthropicClient,
)


@pytest.mark.slow
def test_provider_override_basic():
    """Test basic provider override functionality."""
    # Test that a model normally detected as one provider can be forced to use another
    lm = LM(
        model_name="my-custom-model",
        formatting_model_name="gpt-4o-mini",
        temperature=0.5,
        provider="openai",
    )

    # The client should be OpenAI even though the model name doesn't match OpenAI patterns
    assert isinstance(lm.client, OpenAIStructuredOutputClient)


@pytest.mark.slow
def test_invalid_provider_error():
    """Test that invalid provider raises appropriate error."""
    with pytest.raises(ValueError) as exc_info:
        LM(
            model_name="some-model",
            formatting_model_name="gpt-4o-mini",
            temperature=0.5,
            provider="unsupported-provider",
        )

    assert "Unsupported provider: 'unsupported-provider'" in str(exc_info.value)
    assert "Supported providers are:" in str(exc_info.value)


@pytest.mark.slow
def test_environment_variable_override():
    """Test provider override via environment variable."""
    # Save original env var if exists
    original_provider = os.environ.get("SYNTH_AI_DEFAULT_PROVIDER")

    try:
        # Set environment variable
        os.environ["SYNTH_AI_DEFAULT_PROVIDER"] = "anthropic"

        # Create LM without explicit provider
        lm = LM(
            model_name="my-model",
            formatting_model_name="gpt-4o-mini",
            temperature=0.5,
        )

        # Should use Anthropic client due to env var
        assert isinstance(lm.client, AnthropicClient)

    finally:
        # Restore original env var
        if original_provider is not None:
            os.environ["SYNTH_AI_DEFAULT_PROVIDER"] = original_provider
        else:
            os.environ.pop("SYNTH_AI_DEFAULT_PROVIDER", None)


@pytest.mark.slow
def test_explicit_provider_overrides_env_var():
    """Test that explicit provider parameter takes precedence over env var."""
    # Save original env var if exists
    original_provider = os.environ.get("SYNTH_AI_DEFAULT_PROVIDER")

    try:
        # Set environment variable to anthropic
        os.environ["SYNTH_AI_DEFAULT_PROVIDER"] = "anthropic"

        # Create LM with explicit provider
        lm = LM(
            model_name="my-model",
            formatting_model_name="gpt-4o-mini",
            temperature=0.5,
            provider="openai",  # This should override the env var
        )

        # Should use OpenAI client, not Anthropic
        assert isinstance(lm.client, OpenAIStructuredOutputClient)

    finally:
        # Restore original env var
        if original_provider is not None:
            os.environ["SYNTH_AI_DEFAULT_PROVIDER"] = original_provider
        else:
            os.environ.pop("SYNTH_AI_DEFAULT_PROVIDER", None)


@pytest.mark.slow
def test_get_client_direct():
    """Test get_client function directly with provider override."""
    # Test without provider - should use regex matching
    client1 = get_client("gpt-4o", synth_logging=False)
    assert isinstance(client1, OpenAIStructuredOutputClient)

    # Test with provider override
    client2 = get_client("gpt-4o", provider="anthropic", synth_logging=False)
    assert isinstance(client2, AnthropicClient)

    # Test invalid provider
    with pytest.raises(ValueError) as exc_info:
        get_client("some-model", provider="invalid", synth_logging=False)

    assert "Unsupported provider: 'invalid'" in str(exc_info.value)


if __name__ == "__main__":
    # Run a simple test
    print("Testing provider override functionality...")

    # Test 1: Basic override
    lm = LM(
        model_name="test-model",
        formatting_model_name="gpt-4o-mini",
        temperature=0.5,
        provider="openai",
    )
    print(f"✓ Basic override test passed: {type(lm.client).__name__}")

    # Test 2: Invalid provider
    try:
        lm = LM(
            model_name="test-model",
            formatting_model_name="gpt-4o-mini",
            temperature=0.5,
            provider="unsupported",
        )
        print("✗ Invalid provider test failed: should have raised error")
    except ValueError as e:
        print(f"✓ Invalid provider test passed: {e}")

    print("\nAll manual tests passed!")
