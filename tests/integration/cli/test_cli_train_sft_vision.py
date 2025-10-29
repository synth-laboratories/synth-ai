"""
Integration tests for vision SFT training.

Tests supervised fine-tuning with vision-language models using multimodal data.
"""

import os
import re
import json
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_JOB_ID_PATTERN = re.compile(r"job[_-]id\s*[:=]\s*([a-zA-Z0-9_-]+)")


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


def _create_vision_sft_dataset(output_path: Path) -> None:
    """Create a minimal vision SFT dataset for testing."""
    try:
        from PIL import Image
        import base64
        from io import BytesIO
    except ImportError:
        pytest.skip("PIL not available for dataset creation")
    
    # Create test images
    img1 = Image.new('RGB', (64, 64), color='red')
    buffer1 = BytesIO()
    img1.save(buffer1, format='PNG')
    b64_1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    image1_url = f"data:image/png;base64,{b64_1}"
    
    img2 = Image.new('RGB', (64, 64), color='blue')
    buffer2 = BytesIO()
    img2.save(buffer2, format='PNG')
    b64_2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    image2_url = f"data:image/png;base64,{b64_2}"
    
    # Create minimal SFT examples
    examples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this?"},
                        {"type": "image_url", "image_url": {"url": image1_url}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "This image is red.",
                },
            ],
            "metadata": {"example_id": 1},
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": image2_url}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "This is a blue colored image.",
                },
            ],
            "metadata": {"example_id": 2},
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": image1_url}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I see a red square.",
                },
            ],
            "metadata": {"example_id": 3},
        },
    ]
    
    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✅ Created vision SFT dataset: {output_path}")
    print(f"   Examples: {len(examples)}")
    print(f"   Images per example: 1")


@pytest.mark.slow
@pytest.mark.vision
def test_cli_train_sft_vision_qwen3vl(tmp_path: Path) -> None:
    """Test SFT training with vision model using CLI.
    
    This test verifies:
    1. Vision SFT dataset creation with multimodal messages
    2. Job submission for vision SFT training
    3. Backend accepts vision training config
    
    Uses Qwen3-VL-2B for faster testing.
    
    Marks:
        @pytest.mark.slow - Training job submission takes time
        @pytest.mark.vision - Requires vision model support
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.fail("SYNTH_API_KEY is required for SFT vision test")
    
    # Create minimal vision dataset
    dataset_path = tmp_path / "vision_sft_test.jsonl"
    _create_vision_sft_dataset(dataset_path)
    
    # Create minimal vision SFT config
    config_path = tmp_path / "vision_sft_config.toml"
    config_content = f"""
[algorithm]
type = "offline"
method = "sft"
variety = "lora"

[job]
model = "Qwen/Qwen3-VL-2B-Instruct"
data = "{dataset_path}"

[compute]
gpu_type = "H200"
gpu_count = 1
nodes = 1

[training]
mode = "lora"
use_qlora = true

[training.validation]
enabled = false

[hyperparameters]
n_epochs = 1
train_kind = "peft"
per_device_batch = 1
gradient_accumulation_steps = 2
sequence_length = 2048
learning_rate = 5e-5
warmup_ratio = 0.03
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ["q_proj", "v_proj"]

[hyperparameters.parallelism]
use_deepspeed = false
fsdp = false
bf16 = true
fp16 = false
activation_checkpointing = false

[model_config]
supports_vision = true
max_images_per_message = 1

[tags]
experiment = "ci_test_sft_vision"
purpose = "integration_test"
"""
    config_path.write_text(config_content, encoding="utf-8")
    
    # Prepare environment
    poll_timeout = "120"
    poll_interval = "5"
    
    envfile = tmp_path / "sft_vision.env"
    envfile.write_text(f"SYNTH_API_KEY={api_key}\n", encoding="utf-8")
    
    # Run SFT training command
    repo = _repo_root()
    cmd = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "sft",
        "--config",
        str(config_path),
        "--backend",
        backend,
        "--env-file",
        str(envfile),
        "--no-poll",
        "--poll-timeout",
        poll_timeout,
        "--poll-interval",
        poll_interval,
    ]
    
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        text=True,
        capture_output=True,
        env=env,
        timeout=int(float(poll_timeout)) + 60,
    )
    
    if proc.returncode != 0:
        pytest.fail(
            "CLI SFT vision test failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
    
    # Verify job creation
    assert "✓ Job created" in proc.stdout or "job" in proc.stdout.lower(), \
        f"Job creation not confirmed in stdout:\n{proc.stdout}"
    
    # Verify job ID is present
    match = _JOB_ID_PATTERN.search(proc.stdout)
    assert match, f"job id not found in output:\n{proc.stdout}"
    
    job_id = match.group(1)
    print(f"✅ Vision SFT job created: {job_id}")
    print(f"   Model: Qwen3-VL-2B-Instruct")
    print(f"   Dataset: {dataset_path}")
    print(f"   Examples: 3 (with images)")


@pytest.mark.slow
@pytest.mark.vision
def test_vision_sft_dataset_validation(tmp_path: Path) -> None:
    """Test that vision SFT dataset validation works correctly.
    
    This test verifies that our image validation catches invalid examples
    during dataset preparation (not at training time).
    
    Marks:
        @pytest.mark.slow - Dataset processing can take time
        @pytest.mark.vision - Tests vision data validation
    """
    from synth_ai.learning.sft.data import load_jsonl, validate_vision_example
    
    # Create dataset with mixed valid/invalid examples
    dataset_path = tmp_path / "mixed_vision_sft.jsonl"
    
    try:
        from PIL import Image
        import base64
        from io import BytesIO
    except ImportError:
        pytest.skip("PIL not available")
    
    # Create valid image
    img = Image.new('RGB', (64, 64), color='green')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    valid_image_url = f"data:image/png;base64,{b64}"
    
    examples = [
        # Valid example
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": valid_image_url}},
                    ],
                },
                {"role": "assistant", "content": "A green image."},
            ],
        },
        # Invalid: empty URL
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": ""}},  # Empty!
                    ],
                },
                {"role": "assistant", "content": "Error."},
            ],
        },
        # Invalid: missing URL field
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {}},  # No url
                    ],
                },
                {"role": "assistant", "content": "Error."},
            ],
        },
        # Valid example 2
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": valid_image_url}},
                    ],
                },
                {"role": "assistant", "content": "Green square."},
            ],
        },
    ]
    
    with dataset_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    # Load and validate
    loaded = load_jsonl(dataset_path, min_messages=1)
    assert len(loaded) == 4, "Should load all 4 examples"
    
    valid_count = 0
    invalid_count = 0
    
    for i, example in enumerate(loaded):
        is_valid, error = validate_vision_example(example, require_images=True)
        if is_valid:
            valid_count += 1
            print(f"✅ Example {i}: Valid")
        else:
            invalid_count += 1
            print(f"❌ Example {i}: Invalid - {error}")
    
    # Should have 2 valid, 2 invalid
    assert valid_count == 2, f"Expected 2 valid examples, got {valid_count}"
    assert invalid_count == 2, f"Expected 2 invalid examples, got {invalid_count}"
    
    print(f"✅ Dataset validation working correctly")
    print(f"   Total examples: {len(loaded)}")
    print(f"   Valid: {valid_count}")
    print(f"   Invalid: {invalid_count}")


@pytest.mark.slow
@pytest.mark.vision
def test_cli_train_sft_vision_small_config(tmp_path: Path) -> None:
    """Fast SFT vision test using artifact config.
    
    This test uses a pre-defined minimal config for fast CI validation.
    
    Marks:
        @pytest.mark.slow - Still requires job submission
        @pytest.mark.vision - Tests vision SFT
    """
    _maybe_env()
    
    backend = (
        os.getenv("BACKEND_OVERRIDE")
        or os.getenv("BACKEND_BASE_URL")
        or os.getenv("SYNTH_BASE_URL")
    )
    if not backend:
        backend = "https://agent-learning.onrender.com/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        pytest.fail("SYNTH_API_KEY is required")
    
    # Create dataset
    dataset_path = tmp_path / "vision_sft_small.jsonl"
    _create_vision_sft_dataset(dataset_path)
    
    # Use artifact config (create if doesn't exist)
    repo = _repo_root()
    artifact_config = repo / "tests" / "artifacts" / "configs" / "sft.vision.small.toml"
    
    if not artifact_config.exists():
        # Create it
        artifact_config.parent.mkdir(parents=True, exist_ok=True)
        config_content = f"""
[algorithm]
type = "offline"
method = "sft"
variety = "lora"

[job]
model = "Qwen/Qwen3-VL-2B-Instruct"
data = "{dataset_path}"

[compute]
gpu_type = "H200"
gpu_count = 1

[training]
mode = "lora"
use_qlora = true

[training.validation]
enabled = false

[hyperparameters]
n_epochs = 1
train_kind = "peft"
per_device_batch = 1
gradient_accumulation_steps = 1
sequence_length = 1024
learning_rate = 5e-5
lora_rank = 8
lora_alpha = 16
lora_target_modules = ["q_proj", "v_proj"]

[hyperparameters.parallelism]
bf16 = true
fp16 = false

[model_config]
supports_vision = true
max_images_per_message = 1

[tags]
purpose = "ci_test"
"""
        artifact_config.write_text(config_content, encoding="utf-8")
    else:
        # Update data path in existing config
        config_text = artifact_config.read_text()
        # Simple replacement - in real scenario might use TOML parser
        config_text = re.sub(
            r'data = ".*?"',
            f'data = "{dataset_path}"',
            config_text
        )
        tmp_config = tmp_path / "sft_vision_config.toml"
        tmp_config.write_text(config_text)
        artifact_config = tmp_config
    
    envfile = tmp_path / "sft_small.env"
    envfile.write_text(f"SYNTH_API_KEY={api_key}\n")
    
    cmd = [
        "uvx",
        "synth-ai",
        "train",
        "--type",
        "sft",
        "--config",
        str(artifact_config),
        "--backend",
        backend,
        "--env-file",
        str(envfile),
        "--no-poll",
    ]
    
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    
    if proc.returncode != 0:
        pytest.fail(
            f"SFT vision small config test failed\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
    
    assert "✓ Job created" in proc.stdout or "job" in proc.stdout.lower()
    match = _JOB_ID_PATTERN.search(proc.stdout)
    assert match
    
    print(f"✅ Fast vision SFT job created: {match.group(1)}")


if __name__ == "__main__":
    # For local testing
    import sys
    tmp = Path("/tmp/test_sft_vision")
    tmp.mkdir(exist_ok=True)
    
    print("Running vision SFT tests locally...")
    
    try:
        test_vision_sft_dataset_validation(tmp)
        print("✅ test_vision_sft_dataset_validation passed")
    except Exception as e:
        print(f"❌ test_vision_sft_dataset_validation failed: {e}")
        sys.exit(1)
    
    try:
        test_cli_train_sft_vision_small_config(tmp)
        print("✅ test_cli_train_sft_vision_small_config passed")
    except Exception as e:
        print(f"⚠️  test_cli_train_sft_vision_small_config: {e}")
    
    print("\n🎉 Vision SFT tests complete!")
