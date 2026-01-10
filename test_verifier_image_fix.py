#!/usr/bin/env python3
"""Test that verifier can now see images in traces."""

import base64
import json
from io import BytesIO

import httpx
from PIL import Image


# Create a small test image (red square)
def create_test_image() -> str:
    """Create a small red square image as base64 data URL."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/png;base64,{img_b64}"


# Create a mock trace with an image in the response
def create_mock_trace_with_image(image_url: str) -> dict:
    """Create a V3 format trace with an image in the LLM response."""
    return {
        "event_history": [
            {
                "type": "lm_call",
                "event_type": "lm_call",
                "sequence_index": 0,
                "timestamp": 1767908737.912854,
                "trace_id": "test_trace_001",
                "policy_iter": 0,
                "llm_request": {
                    "messages": [{"role": "user", "content": "Generate a red square image"}]
                },
                "llm_response": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    }
                },
            }
        ],
        "session_time_steps": [
            {
                "events": [
                    {
                        "type": "lm_call",
                        "llm_request": {
                            "messages": [{"role": "user", "content": "Generate a red square image"}]
                        },
                        "llm_response": {
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                            }
                        },
                    }
                ]
            }
        ],
    }


def test_verifier():
    """Test the verifier with an image."""
    print("=" * 80)
    print("TESTING VERIFIER WITH IMAGE")
    print("=" * 80)

    # Create test image
    print("\n1. Creating test image (red 100x100 square)...")
    test_image_url = create_test_image()
    print(f"   Image size: {len(test_image_url)} bytes")

    # Create mock trace
    print("\n2. Creating mock trace with image in response...")
    trace = create_mock_trace_with_image(test_image_url)
    print(f"   Trace has {len(trace['event_history'])} events")

    # Create rubric
    print("\n3. Creating rubric...")
    rubric = {
        "outcome": {
            "name": "Image Quality",
            "criteria": [
                {
                    "id": "color_accuracy",
                    "description": "Does the generated image show a red square?",
                    "weight": 1.0,
                }
            ],
        }
    }

    # Call verifier
    print("\n4. Calling verifier API...")
    url = "http://localhost:8000/api/graphs/verifiers/completions"

    payload = {
        "job_id": "zero_shot_verifier_rubric_single",
        "input": {"trace": trace, "rubric": rubric, "options": {"model": "gemini-2.5-flash"}},
        "model": "gemini-2.5-flash",
    }

    try:
        response = httpx.post(
            url, json=payload, timeout=60.0, headers={"Content-Type": "application/json"}
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 80)
            print("✅ VERIFIER RESPONSE")
            print("=" * 80)

            # Check if verifier saw the image
            outcome_review = result.get("output", {}).get("outcome_review", {})
            criteria = outcome_review.get("criteria", {})

            if criteria:
                print("\n✅ Verifier returned criteria scores!")
                for criterion_id, criterion_data in criteria.items():
                    score = criterion_data.get("score", 0.0)
                    reason = criterion_data.get("reason", "")
                    print(f"\n{criterion_id}:")
                    print(f"  Score: {score}")
                    print(f"  Reason: {reason}")

                # Check if the reason mentions seeing the image
                if (
                    "red" in reason.lower()
                    or "square" in reason.lower()
                    or "image" in reason.lower()
                ):
                    print("\n" + "=" * 80)
                    print("✅✅✅ SUCCESS! Verifier can SEE the image!")
                    print("=" * 80)
                else:
                    print("\n" + "=" * 80)
                    print("⚠️  WARNING: Verifier responded but didn't clearly describe the image")
                    print("=" * 80)
            else:
                print("\n❌ No criteria in outcome_review")
                print(f"Full response: {json.dumps(result, indent=2)}")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_verifier()
