"""
Basic functionality tests for Synth AI - mix of fast and slow tests
"""

import pytest
import time
from synth_ai.zyk import LM


@pytest.mark.fast
def test_basic_imports():
    """Test that basic imports work correctly - should be very fast."""
    from synth_ai import LM
    from synth_ai.lm.config import SynthConfig

    # These should import without error
    assert LM is not None
    assert SynthConfig is not None


@pytest.mark.fast
def test_config_creation():
    """Test basic configuration creation - should be fast."""
    from synth_ai.lm.config import SynthConfig

    # Test creating config with required parameters
    config = SynthConfig(base_url="test", api_key="test")
    assert config is not None
    assert hasattr(config, "timeout")


@pytest.mark.fast
def test_basic_data_structures():
    """Test basic data structure functionality - should be fast."""
    # Test basic operations that don't require external calls
    test_dict = {"role": "user", "content": "test"}
    assert test_dict["role"] == "user"

    test_list = [1, 2, 3, 4]
    assert sum(test_list) == 10


@pytest.mark.slow
def test_lm_initialization():
    """Test LM initialization - slower due to model setup."""
    lm = LM(model_name="gpt-4o-mini", formatting_model_name="gpt-4o-mini", temperature=0.7)

    assert lm is not None
    assert lm.model_name == "gpt-4o-mini"


@pytest.mark.slow
def test_simple_lm_call():
    """Test a simple LM call - slow due to API call."""
    lm = LM(model_name="gpt-4o-mini", formatting_model_name="gpt-4o-mini", temperature=0)

    response = lm.respond_sync(
        system_message="You are a helpful assistant.", user_message="Say 'hello world' exactly."
    )

    assert "hello world" in response.raw_response.lower()


@pytest.mark.slow
def test_timeout_simulation():
    """Simulate a test that takes time - marked as slow."""
    # Simulate some work that takes time
    time.sleep(2)
    assert True


@pytest.mark.fast
def test_string_operations():
    """Test string operations - should be very fast."""
    test_string = "Hello, World!"
    assert test_string.lower() == "hello, world!"
    assert len(test_string) == 13
    assert test_string.startswith("Hello")


@pytest.mark.fast
def test_list_operations():
    """Test list operations - should be very fast."""
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert max(test_list) == 5
    assert min(test_list) == 1

    # Test list comprehension
    squared = [x**2 for x in test_list]
    assert squared == [1, 4, 9, 16, 25]


@pytest.mark.fast
def test_dict_operations():
    """Test dictionary operations - should be very fast."""
    test_dict = {"a": 1, "b": 2, "c": 3}
    assert len(test_dict) == 3
    assert test_dict["a"] == 1
    assert "b" in test_dict

    # Test dictionary comprehension
    doubled = {k: v * 2 for k, v in test_dict.items()}
    assert doubled == {"a": 2, "b": 4, "c": 6}
