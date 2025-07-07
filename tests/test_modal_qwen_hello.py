"""Simple test for Modal Qwen integration.
Run with: uv run python test_modal_qwen_hello.py
"""

import os

def test_modal_qwen():
    """Test basic Modal Qwen functionality."""
    from synth_ai import LM
    
    model_url = os.environ.get("MODAL_TEST_URL")
    if not model_url:
        raise SystemExit("Set MODAL_TEST_URL to your deployed app (e.g. org--qwen-test.modal.run)")
    
    print(f"Testing Modal URL: {model_url}")
    
    lm = LM(
        model_name=model_url,
        formatting_model_name=model_url,  # Use same endpoint for formatting
        temperature=0.7
    )
    
    response = lm.respond_sync(
        system_message="You are a helpful assistant.",
        user_message="Say hello and tell me what model you are."
    )
    
    print(f"Response: {response}")
    print("✓ Modal Qwen test passed!")

if __name__ == "__main__":
    test_modal_qwen() 