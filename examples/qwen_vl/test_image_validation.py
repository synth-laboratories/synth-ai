#!/usr/bin/env python3
"""
Quick test script to demonstrate image validation.

Run from synth-ai root:
    uv run python examples/qwen_vl/test_image_validation.py
"""

from synth_ai.learning.sft.data import coerce_example, validate_vision_example

# Test cases
test_cases = [
    {
        "name": "Valid - HTTP URL",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                    ],
                },
                {"role": "assistant", "content": "A beautiful image"},
            ]
        },
        "should_pass": True,
    },
    {
        "name": "Valid - Base64",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
                    ],
                },
                {"role": "assistant", "content": "An image"},
            ]
        },
        "should_pass": True,
    },
    {
        "name": "Invalid - Empty URL",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": ""}},  # Empty!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
    {
        "name": "Invalid - Missing URL field",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {}},  # No url field!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
    {
        "name": "Invalid - Null URL",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": None}},  # Null!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
    {
        "name": "Invalid - Whitespace URL",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "   "}},  # Whitespace!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
    {
        "name": "Invalid - Mixed valid and invalid",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/valid.jpg"}},
                        {"type": "image_url", "image_url": {"url": ""}},  # One invalid!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
    {
        "name": "Invalid - Non-string URL",
        "data": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": 12345}},  # Integer!
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        },
        "should_pass": False,
    },
]


def main():
    print("=" * 80)
    print("IMAGE VALIDATION TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        name = test["name"]
        data = test["data"]
        should_pass = test["should_pass"]
        
        try:
            example = coerce_example(data)
            is_valid, error = validate_vision_example(example, require_images=True)
            
            if should_pass:
                if is_valid:
                    print(f"✅ PASS: {name}")
                    print(f"   → Correctly accepted valid example")
                    passed += 1
                else:
                    print(f"❌ FAIL: {name}")
                    print(f"   → Should pass but got error: {error}")
                    failed += 1
            else:
                if not is_valid:
                    print(f"✅ PASS: {name}")
                    print(f"   → Correctly rejected: {error}")
                    passed += 1
                else:
                    print(f"❌ FAIL: {name}")
                    print(f"   → Should fail but passed validation")
                    failed += 1
        except Exception as exc:
            if should_pass:
                print(f"❌ FAIL: {name}")
                print(f"   → Unexpected exception: {exc}")
                failed += 1
            else:
                print(f"✅ PASS: {name}")
                print(f"   → Correctly raised exception: {exc}")
                passed += 1
        
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} passed, {failed}/{len(test_cases)} failed")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

