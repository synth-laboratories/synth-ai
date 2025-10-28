"""Unit tests for crafter policy vision support."""



import pytest

# Import the policy class
from synth_ai.demos.demo_task_apps.crafter.grpo_crafter_task_app import _load_build_config

# We need to dynamically import the policy from the task app
import importlib.util
import sys
from pathlib import Path


@pytest.fixture
def crafter_policy_class():
    """Dynamically import CrafterPolicy from the task app."""
    try:
        # Try importing from installed package
        module = importlib.import_module("examples.task_apps.crafter.task_app.synth_envs_hosted.envs.crafter.policy")
        return module.CrafterPolicy
    except ImportError:
        # Fallback to direct file import
        import synth_ai
        synth_ai_path = Path(synth_ai.__file__).resolve().parent.parent
        policy_path = (
            synth_ai_path / "examples" / "task_apps" / "crafter" / "task_app" / 
            "synth_envs_hosted" / "envs" / "crafter" / "policy.py"
        )
        
        spec = importlib.util.spec_from_file_location("crafter_policy", policy_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["crafter_policy"] = module
            spec.loader.exec_module(module)
            return module.CrafterPolicy
        
        pytest.skip("Could not import CrafterPolicy")


@pytest.mark.fast
def test_is_vision_model_detects_gpt4o(crafter_policy_class):
    """Test that GPT-4o models are detected as vision models."""
    assert crafter_policy_class._is_vision_model("gpt-4o")
    assert crafter_policy_class._is_vision_model("gpt-4o-2024-05-13")
    assert crafter_policy_class._is_vision_model("gpt-4o-mini")


@pytest.mark.fast
def test_is_vision_model_detects_gpt5(crafter_policy_class):
    """Test that GPT-5 models are detected as vision models."""
    assert crafter_policy_class._is_vision_model("gpt-5")
    assert crafter_policy_class._is_vision_model("gpt-5-nano")
    assert crafter_policy_class._is_vision_model("gpt-5-mini")
    assert crafter_policy_class._is_vision_model("GPT-5-NANO")  # Case insensitive


@pytest.mark.fast
def test_is_vision_model_detects_claude3(crafter_policy_class):
    """Test that Claude 3 models are detected as vision models."""
    assert crafter_policy_class._is_vision_model("claude-3-opus-20240229")
    assert crafter_policy_class._is_vision_model("claude-3-5-sonnet-20241022")
    assert crafter_policy_class._is_vision_model("claude-3-haiku")


@pytest.mark.fast
def test_is_vision_model_detects_gemini(crafter_policy_class):
    """Test that Gemini models are detected as vision models."""
    assert crafter_policy_class._is_vision_model("gemini-pro-vision")
    assert crafter_policy_class._is_vision_model("gemini-1.5-pro")


@pytest.mark.fast
def test_is_vision_model_rejects_text_only(crafter_policy_class):
    """Test that text-only models are not detected as vision models."""
    assert not crafter_policy_class._is_vision_model("gpt-3.5-turbo")
    assert not crafter_policy_class._is_vision_model("gpt-4")
    assert not crafter_policy_class._is_vision_model("claude-2")
    assert not crafter_policy_class._is_vision_model("llama-3-70b")


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_auto_detects_vision_from_model(crafter_policy_class):
    """Test that policy automatically enables vision for VLMs."""
    policy = crafter_policy_class(inference_url="http://test", model="gpt-4o")
    await policy.initialize({"model": "gpt-4o"})
    
    assert policy.use_vision is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_respects_explicit_use_vision(crafter_policy_class):
    """Test that explicit use_vision config overrides auto-detection."""
    # Explicitly enable vision for a non-VLM
    policy = crafter_policy_class(inference_url="http://test", model="gpt-3.5-turbo")
    await policy.initialize({"model": "gpt-3.5-turbo", "use_vision": True})
    
    assert policy.use_vision is True
    
    # Explicitly disable vision for a VLM
    policy2 = crafter_policy_class(inference_url="http://test", model="gpt-4o")
    await policy2.initialize({"model": "gpt-4o", "use_vision": False})
    
    assert policy2.use_vision is False


@pytest.mark.fast
def test_extract_image_parts_with_vision_disabled(crafter_policy_class):
    """Test that no images are extracted when vision is disabled."""
    policy = crafter_policy_class(inference_url="http://test")
    policy.use_vision = False
    
    observation = {
        "observation": {
            "observation_image_data_url": "data:image/png;base64,iVBORw0KG...",
            "observation_image_width": 64,
            "observation_image_height": 64,
        }
    }
    
    image_parts = policy._extract_image_parts(observation)
    assert image_parts == []


@pytest.mark.fast
def test_extract_image_parts_with_vision_enabled(crafter_policy_class):
    """Test that images are extracted when vision is enabled."""
    policy = crafter_policy_class(inference_url="http://test")
    policy.use_vision = True
    
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    observation = {
        "observation": {
            "observation_image_data_url": data_url,
            "observation_image_width": 1,
            "observation_image_height": 1,
        }
    }
    
    image_parts = policy._extract_image_parts(observation)
    
    assert len(image_parts) == 1
    assert image_parts[0]["type"] == "image_url"
    assert image_parts[0]["image_url"]["url"] == data_url


@pytest.mark.fast
def test_extract_image_parts_handles_missing_data(crafter_policy_class):
    """Test that missing image data is handled gracefully."""
    policy = crafter_policy_class(inference_url="http://test")
    policy.use_vision = True
    
    # No observation_image_data_url
    observation = {
        "observation": {
            "health": 10,
        }
    }
    
    image_parts = policy._extract_image_parts(observation)
    assert image_parts == []
    
    # Empty observation
    assert policy._extract_image_parts({}) == []
    
    # None observation
    assert policy._extract_image_parts(None) == []


@pytest.mark.fast
def test_extract_image_parts_handles_nested_observation(crafter_policy_class):
    """Test that nested observation structure is handled correctly."""
    policy = crafter_policy_class(inference_url="http://test")
    policy.use_vision = True
    
    data_url = "data:image/png;base64,abc123"
    
    # Nested structure
    observation = {
        "observation": {
            "observation_image_data_url": data_url,
        },
        "step_idx": 5,
    }
    
    image_parts = policy._extract_image_parts(observation)
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == data_url


@pytest.mark.fast
@pytest.mark.asyncio
async def test_serialize_includes_use_vision(crafter_policy_class):
    """Test that serialization includes use_vision flag."""
    policy = crafter_policy_class(inference_url="http://test", model="gpt-4o")
    await policy.initialize({"model": "gpt-4o"})
    
    serialized = await policy.serialize()
    
    assert "use_vision" in serialized["config"]
    assert serialized["config"]["use_vision"] is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_deserialize_restores_use_vision(crafter_policy_class):
    """Test that deserialization restores use_vision flag."""
    payload = {
        "name": "crafter-react",
        "config": {
            "inference_url": "http://test",
            "model": "gpt-4o",
            "use_tools": True,
            "use_vision": True,
        },
        "state": {
            "turn_index": 0,
            "history_messages": [],
            "trajectory_history": [],
        }
    }
    
    policy = await crafter_policy_class.deserialize(payload)
    
    assert policy.use_vision is True
    assert policy.model == "gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

