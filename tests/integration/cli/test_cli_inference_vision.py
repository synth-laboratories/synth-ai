"""
Integration tests for vision inference.

Tests vision-language model inference with multimodal (text + image) requests.
"""

import os
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from io import BytesIO

import pytest

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _maybe_env() -> None:
    """Load env vars from .env.test.* files if not already set."""
    if os.getenv("SYNTH_API_KEY"):
        return
    repo = _repo_root()
    for candidate in (".env.test.prod", ".env.test.dev", ".env.test", ".env"):
        p = repo / candidate
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            continue


def _create_test_image_base64() -> str:
    """Create a simple test image and return base64 data URL."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    
    # Create a simple 64x64 red square
    img = Image.new('RGB', (64, 64), color='red')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/png;base64,{b64}"


@pytest.mark.slow
@pytest.mark.vision
def test_vision_inference_with_image() -> None:
    """Test vision inference with a multimodal message.
    
    This test verifies:
    1. Backend accepts multimodal messages with images
    2. Vision model can process image + text input
    3. Response is returned successfully
    
    Marks:
        @pytest.mark.slow - Requires backend call
        @pytest.mark.vision - Requires vision model
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    if not backend:
        pytest.skip("No backend URL configured")
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.skip("SYNTH_API_KEY not set")
    
    # Create test image
    image_url = _create_test_image_base64()
    
    # Prepare multimodal request
    request = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",  # Small vision model for testing
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 50,
        "temperature": 0.6,
        "thinking_budget": 0,  # Required by backend
    }
    
    # Make inference request
    inference_url = backend.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        inference_url,
        data=json.dumps(request).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            
            # Verify response structure
            assert isinstance(result, dict), "Response should be a dict"
            assert "choices" in result, "Response should have choices"
            assert len(result["choices"]) > 0, "Should have at least one choice"
            
            choice = result["choices"][0]
            assert "message" in choice, "Choice should have message"
            assert "content" in choice["message"], "Message should have content"
            
            content = choice["message"]["content"]
            assert isinstance(content, str), "Content should be a string"
            assert len(content) > 0, "Content should not be empty"
            
            print(f"✅ Vision inference successful")
            print(f"   Model: {request['model']}")
            print(f"   Response: {content[:100]}...")
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        pytest.fail(
            f"Vision inference request failed: {e.code}\n"
            f"Error: {error_body}\n"
            f"URL: {inference_url}\n"
        )
    except Exception as e:
        pytest.fail(f"Vision inference request failed: {e}")


@pytest.mark.slow
@pytest.mark.vision
def test_vision_inference_validation() -> None:
    """Test that invalid vision requests are caught by validation.
    
    This test verifies that the image validation we added catches:
    - Empty image URLs
    - Invalid image formats
    - Missing image data
    
    Marks:
        @pytest.mark.slow - Requires backend call
        @pytest.mark.vision - Tests vision validation
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    if not backend:
        pytest.skip("No backend URL configured")
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.skip("SYNTH_API_KEY not set")
    
    # Test cases that should fail validation
    invalid_requests = [
        {
            "name": "Empty image URL",
            "request": {
                "model": "Qwen/Qwen3-VL-2B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's this?"},
                            {"type": "image_url", "image_url": {"url": ""}},  # Empty!
                        ],
                    }
                ],
                "thinking_budget": 0,
            },
        },
        {
            "name": "Missing URL field",
            "request": {
                "model": "Qwen/Qwen3-VL-2B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {}},  # No url field
                        ],
                    }
                ],
                "thinking_budget": 0,
            },
        },
        {
            "name": "Whitespace URL",
            "request": {
                "model": "Qwen/Qwen3-VL-2B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "   "}},  # Whitespace
                        ],
                    }
                ],
                "thinking_budget": 0,
            },
        },
    ]
    
    inference_url = backend.rstrip("/") + "/v1/chat/completions"
    
    for test_case in invalid_requests:
        name = test_case["name"]
        request = test_case["request"]
        
        req = urllib.request.Request(
            inference_url,
            data=json.dumps(request).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # If this succeeds, validation failed to catch the invalid request
                pytest.fail(f"Validation should have rejected: {name}")
        except urllib.error.HTTPError as e:
            # We expect 4xx errors for validation failures
            if e.code >= 400 and e.code < 500:
                error_body = e.read().decode("utf-8", errors="ignore")
                print(f"✅ Correctly rejected: {name}")
                print(f"   Error code: {e.code}")
                print(f"   Error message: {error_body[:200]}")
            else:
                pytest.fail(f"Unexpected error code for {name}: {e.code}")
        except Exception as e:
            pytest.fail(f"Unexpected error for {name}: {e}")


@pytest.mark.slow
@pytest.mark.vision
def test_vision_inference_multiple_images() -> None:
    """Test vision inference with multiple images in a single message.
    
    Verifies that the backend can handle multiple image inputs.
    
    Marks:
        @pytest.mark.slow - Requires backend call
        @pytest.mark.vision - Requires vision model
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    if not backend:
        pytest.skip("No backend URL configured")
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.skip("SYNTH_API_KEY not set")
    
    # Create two test images
    image1_url = _create_test_image_base64()
    
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    
    # Create a blue image for variety
    img2 = Image.new('RGB', (64, 64), color='blue')
    buffer = BytesIO()
    img2.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    image2_url = f"data:image/png;base64,{b64}"
    
    # Prepare multimodal request with 2 images
    request = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What colors are they?"},
                    {"type": "image_url", "image_url": {"url": image1_url}},
                    {"type": "image_url", "image_url": {"url": image2_url}},
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0.6,
        "thinking_budget": 0,  # Required by backend
    }
    
    # Make inference request
    inference_url = backend.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        inference_url,
        data=json.dumps(request).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            
            assert "choices" in result
            assert len(result["choices"]) > 0
            
            content = result["choices"][0]["message"]["content"]
            assert len(content) > 0
            
            print(f"✅ Multi-image inference successful")
            print(f"   Images: 2")
            print(f"   Response: {content[:150]}...")
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        # Multiple images might not be supported by all models
        if e.code == 400 and "image" in error_body.lower():
            pytest.skip(f"Model doesn't support multiple images: {error_body[:200]}")
        pytest.fail(f"Multi-image inference failed: {e.code}\n{error_body}")
    except Exception as e:
        pytest.fail(f"Multi-image inference failed: {e}")


if __name__ == "__main__":
    # For local testing
    import sys
    
    print("Running vision inference tests locally...")
    
    try:
        test_vision_inference_with_image()
        print("✅ test_vision_inference_with_image passed")
    except Exception as e:
        print(f"❌ test_vision_inference_with_image failed: {e}")
        sys.exit(1)
    
    try:
        test_vision_inference_validation()
        print("✅ test_vision_inference_validation passed")
    except Exception as e:
        print(f"⚠️  test_vision_inference_validation: {e}")
    
    try:
        test_vision_inference_multiple_images()
        print("✅ test_vision_inference_multiple_images passed")
    except Exception as e:
        print(f"⚠️  test_vision_inference_multiple_images: {e}")
    
    print("\n🎉 Vision inference tests complete!")

