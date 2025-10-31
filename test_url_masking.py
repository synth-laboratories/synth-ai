#!/usr/bin/env python3
"""Test URL masking in streaming handlers."""

from synth_ai.streaming.handlers import _mask_sensitive_urls


def test_url_masking():
    """Verify sensitive URLs are properly masked."""
    
    test_cases = [
        # S3 URLs with different file types
        (
            "Uploaded final checkpoint to Wasabi s3://synth-artifacts/models/Qwen-Qwen3-4B/rl/job_19a38041c38f96e638c/checkpoint-epoch-1.tar.gz",
            "Uploaded final checkpoint to Wasabi s3://***/***/[checkpoint-epoch-1.tar.gz]"
        ),
        (
            "Attached final model artifact s3://synth-artifacts/models/Qwen-Qwen3-4B/rl/job_19a38041c38f96e638c/checkpoint-epoch-1.tar.gz",
            "Attached final model artifact s3://***/***/[checkpoint-epoch-1.tar.gz]"
        ),
        (
            "Downloaded weights from s3://my-bucket/path/to/model.safetensors",
            "Downloaded weights from s3://***/***/[model.safetensors]"
        ),
        (
            "https://s3.wasabisys.com/bucket/models/checkpoint.pt uploaded successfully",
            "s3://***/***/[checkpoint.pt] uploaded successfully"
        ),
        # Non-sensitive text should pass through
        (
            "Training started with 100 examples",
            "Training started with 100 examples"
        ),
        (
            "Model loaded: Qwen/Qwen3-4B",
            "Model loaded: Qwen/Qwen3-4B"
        ),
    ]
    
    print("Testing URL masking...")
    print("=" * 80)
    
    all_passed = True
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = _mask_sensitive_urls(input_text)
        passed = result == expected
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"\nTest {i}: {status}")
        print(f"  Input:    {input_text[:80]}...")
        print(f"  Expected: {expected[:80]}...")
        print(f"  Got:      {result[:80]}...")
        if not passed:
            print(f"  MISMATCH!")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    success = test_url_masking()
    exit(0 if success else 1)

