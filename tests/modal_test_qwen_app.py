"""Minimal Modal server for Qwen2.5-0.5B-Instruct.
Deploy with:  modal deploy modal_test_qwen_app.py
The resulting URL will look like  <org>--qwen-test.modal.run
"""

import modal

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
app = modal.App("qwen-test-v2")

# Shared image with all dependencies
image = modal.Image.debian_slim().pip_install(
    ["torch", "transformers", "accelerate", "peft", "fastapi"]
)


# Combined FastAPI app that includes the model directly
@app.function(image=image, gpu="a10g", min_containers=1, max_containers=2)
@modal.asgi_app()
def fastapi_app():

    import torch
    from fastapi import FastAPI
    from pydantic import BaseModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model on startup
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded!")

    web_app = FastAPI()

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = ""
        messages: list[ChatMessage]
        temperature: float = 0.7

    def messages_to_prompt(messages):
        """Very naive prompt conversion (user-assistant pairs)."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    @web_app.post("/chat/completions")
    def chat_completions(request: ChatRequest):
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt = messages_to_prompt(messages)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Handle temperature=0.0 by using greedy decoding
            if request.temperature == 0.0:
                generation_kwargs = {
                    "max_new_tokens": 128,
                    "do_sample": False,  # Greedy decoding for temperature=0
                }
            else:
                generation_kwargs = {
                    "max_new_tokens": 128,
                    "temperature": request.temperature,
                    "do_sample": True,
                }

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

            completion = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
            ).strip()

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": completion,
                        }
                    }
                ]
            }
        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}

    return web_app


# Convenience local entrypoint for quick manual tests
@app.local_entrypoint()
def main():
    import json
    import os

    import requests

    url = os.environ.get("MODAL_TEST_URL")
    if not url:
        raise SystemExit(
            "Set MODAL_TEST_URL to the deployed app name (e.g. org--qwen-test.modal.run)"
        )
    payload = {"messages": [{"role": "user", "content": "Hello!"}]}
    r = requests.post(f"https://{url}/chat/completions", json=payload)
    print(json.dumps(r.json(), indent=2))
