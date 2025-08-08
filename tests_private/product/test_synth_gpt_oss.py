#!/usr/bin/env python3
"""
Test Synth backend with OpenAI GPT-OSS models.
Tests: warmup, inference, reasoning extraction, tool calling, and fine-tuning.
"""

import os
import json
import time
import httpx
import asyncio
from typing import Dict, Any, Optional, List

# Configuration
SYNTH_API_URL = os.getenv("SYNTH_API_URL", "http://localhost:8000")
SYNTH_API_KEY = os.getenv("SYNTH_API_KEY", "")

# GPT-OSS models to test (only 20B and 120B variants exist)
GPT_OSS_MODELS = {
    "20B": "openai/gpt-oss-20b",
    "120B": "openai/gpt-oss-120b"
}

# Use the smallest model by default for faster testing
MODEL_ID = GPT_OSS_MODELS["20B"]

# GPU configurations for GPT-OSS models (MoE architecture requires powerful GPUs)
# 20B: A100, H100, L40S (inference)
# 120B: H100 only
GPU_CONFIGS = {
    "20B": ["A100", "H100"],
    "120B": ["H100"]
}

# Training data for fine-tuning
TRAINING_DATA = [
    {
        "messages": [
            {"role": "system", "content": "You are ChatGPT, a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "<|thinking|>\nThe user is asking about machine learning. I should provide a clear, concise explanation.\n<|/thinking|>\n\n<|final|>\nMachine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.\n"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain neural networks."},
            {"role": "assistant", "content": "<|thinking|>\nNeural networks are a fundamental concept in deep learning. I should explain them clearly.\n<|/thinking|>\n\n<|final|>\nNeural networks are computational models inspired by the human brain's structure. They consist of interconnected nodes (neurons) organized in layers that process information by adjusting connection weights through training.\n"}
        ]
    }
]

# Test prompts
TEST_PROMPTS = [
    "What is 2+2?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about artificial intelligence."
]

# Test tools for GPT-OSS native tool support
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


class SynthGPTOSSTester:
    """Test Synth backend with GPT-OSS models."""
    
    def __init__(self, model_size: str = "20B"):
        self.model_size = model_size
        self.model_id = GPT_OSS_MODELS[model_size]
        self.gpu_configs = GPU_CONFIGS[model_size]
        self.headers = {
            "Authorization": f"Bearer {SYNTH_API_KEY}",
            "Content-Type": "application/json"
        }
        self.results = []
        self.finetuned_model_id = None
    
    async def test_warmup(self):
        """Test 1: Warm up GPT-OSS model on different GPUs."""
        print("\n" + "="*60)
        print(f"TEST 1: GPU Warmup for GPT-OSS {self.model_size} ({self.model_id})")
        print("="*60)
        
        warmup_results = []
        
        for gpu in self.gpu_configs:
            print(f"\nWarming up on {gpu}...")
            
            headers = self.headers.copy()
            headers["X-GPU-Preference"] = gpu
            
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"{SYNTH_API_URL}/api/warmup/{self.model_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  Status: {data.get('status')}")
                        print(f"  GPU: {data.get('gpu', gpu)}")
                        
                        # GPT-OSS models are large, may take longer to warm up
                        warmed = await self._wait_for_warmup(self.model_id, gpu, timeout=180)
                        elapsed = time.time() - start_time
                        
                        warmup_results.append({
                            "gpu": gpu,
                            "success": warmed,
                            "time": elapsed
                        })
                        
                        if warmed:
                            print(f"  ✓ Warmed up successfully in {elapsed:.2f}s")
                        else:
                            print(f"  ✗ Warmup timeout")
                    else:
                        print(f"  ✗ Warmup failed: {response.status_code}")
                        print(f"    Response: {response.text[:200]}")
                        warmup_results.append({
                            "gpu": gpu,
                            "success": False,
                            "error": f"Status {response.status_code}"
                        })
                        
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    warmup_results.append({
                        "gpu": gpu,
                        "success": False,
                        "error": str(e)
                    })
        
        successful = sum(1 for r in warmup_results if r.get("success"))
        print(f"\n✓ Warmup completed: {successful}/{len(self.gpu_configs)} GPUs ready")
        self.results.append(("warmup", warmup_results))
        return successful > 0
    
    async def test_inference_with_reasoning(self):
        """Test 2: Run inference with GPT-OSS model and extract reasoning."""
        print("\n" + "="*60)
        print(f"TEST 2: Inference with Reasoning - GPT-OSS {self.model_size}")
        print("="*60)
        
        inference_results = []
        
        # Test with reasoning prompts
        reasoning_prompts = [
            "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?",
            "Think through this: What are the pros and cons of renewable energy?",
            "Analyze: Why do seasons change on Earth?"
        ]
        
        for i, prompt in enumerate(reasoning_prompts, 1):
            # Add system message to enable Harmony format with thinking
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": "reasoning language: English\nYou are ChatGPT. Use the thinking channel for step-by-step reasoning."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.6,
                "max_tokens": 500,
                "gpu_preference": self.gpu_configs[0]
            }
            
            async with httpx.AsyncClient(timeout=90) as client:
                try:
                    start_time = time.time()
                    response = await client.post(
                        f"{SYNTH_API_URL}/api/v1/chat/completions",
                        json=payload,
                        headers=self.headers
                    )
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        
                        # Check for reasoning/thinking in response
                        has_reasoning = "<|thinking|>" in content or "<|analysis|>" in content
                        
                        print(f"\nPrompt {i}: '{prompt[:50]}...'")
                        
                        if has_reasoning:
                            # Try to extract reasoning and final answer
                            if "<|thinking|>" in content and "<|/thinking|>" in content:
                                reasoning_start = content.index("<|thinking|>") + len("<|thinking|>")
                                reasoning_end = content.index("<|/thinking|>")
                                reasoning = content[reasoning_start:reasoning_end].strip()
                                print(f"  Reasoning: {reasoning[:100]}...")
                            
                            if "<|final|>" in content:
                                final_start = content.index("<|final|>") + len("<|final|>")
                                final_answer = content[final_start:].strip()
                                print(f"  Final Answer: {final_answer[:100]}...")
                            else:
                                print(f"  Response: {content[:100]}...")
                        else:
                            print(f"  Response: {content[:100]}...")
                        
                        print(f"  Time: {elapsed:.2f}s")
                        print(f"  Has Reasoning: {'Yes' if has_reasoning else 'No'}")
                        
                        inference_results.append({
                            "prompt": prompt,
                            "success": True,
                            "has_reasoning": has_reasoning,
                            "time": elapsed,
                            "response_length": len(content)
                        })
                    else:
                        print(f"  ✗ Inference failed: {response.status_code}")
                        inference_results.append({
                            "prompt": prompt,
                            "success": False,
                            "error": f"Status {response.status_code}"
                        })
                        
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    inference_results.append({
                        "prompt": prompt,
                        "success": False,
                        "error": str(e)
                    })
        
        successful = sum(1 for r in inference_results if r.get("success"))
        with_reasoning = sum(1 for r in inference_results if r.get("has_reasoning"))
        
        print(f"\n✓ Inference completed: {successful}/{len(inference_results)} successful")
        print(f"  With reasoning: {with_reasoning}/{successful}")
        
        self.results.append(("inference_reasoning", inference_results))
        return successful > 0
    
    async def test_native_tool_calling(self):
        """Test 3: Test GPT-OSS native tool calling support."""
        print("\n" + "="*60)
        print(f"TEST 3: Native Tool Calling - GPT-OSS {self.model_size}")
        print("="*60)
        
        # Test with a prompt that should trigger tool use
        test_prompt = "Calculate the result of 15 * 28 + 342"
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are ChatGPT. Use the provided tools when needed."},
                {"role": "user", "content": test_prompt}
            ],
            "tools": TEST_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.6,
            "max_tokens": 200,
            "gpu_preference": self.gpu_configs[0]
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{SYNTH_API_URL}/api/v1/chat/completions",
                    json=payload,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    message = data["choices"][0]["message"]
                    
                    # Check for tool calls in response
                    if "tool_calls" in message and message["tool_calls"]:
                        print("  ✓ Model used tools")
                        for tool_call in message["tool_calls"]:
                            func_name = tool_call["function"]["name"]
                            func_args = tool_call["function"]["arguments"]
                            print(f"    Tool: {func_name}")
                            print(f"    Arguments: {func_args}")
                        
                        self.results.append(("tool_calling", {
                            "success": True,
                            "num_tools_called": len(message["tool_calls"])
                        }))
                        return True
                    else:
                        # Check if tool call is in content (Harmony format)
                        content = message.get("content", "")
                        if "<|tool_call|>" in content:
                            print("  ✓ Model used tools (Harmony format)")
                            print(f"    Response: {content[:200]}...")
                            self.results.append(("tool_calling", {
                                "success": True,
                                "format": "harmony"
                            }))
                            return True
                        else:
                            print("  ⚠ Model did not use tools")
                            print(f"    Response: {content[:200]}...")
                            self.results.append(("tool_calling", {
                                "success": False,
                                "note": "No tool calls detected"
                            }))
                            return False
                else:
                    print(f"  ✗ Request failed: {response.status_code}")
                    self.results.append(("tool_calling", {
                        "success": False,
                        "error": f"Status {response.status_code}"
                    }))
                    return False
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results.append(("tool_calling", {
                    "success": False,
                    "error": str(e)
                }))
                return False
    
    async def test_moe_finetuning(self):
        """Test 4: Fine-tune GPT-OSS MoE model with MXFP4 quantization."""
        print("\n" + "="*60)
        print(f"TEST 4: MoE Fine-tuning - GPT-OSS {self.model_size}")
        print("="*60)
        
        # Upload training data
        print("\nUploading training data...")
        file_id = await self._upload_training_file(TRAINING_DATA)
        
        if not file_id:
            print("✗ Failed to upload training data")
            self.results.append(("moe_finetuning", {"success": False, "error": "Upload failed"}))
            return False
        
        print(f"✓ Training file uploaded: {file_id}")
        
        # Create fine-tuning job with MoE-specific settings
        print("\nCreating MoE fine-tuning job...")
        
        # GPT-OSS specific hyperparameters
        job_payload = {
            "model": self.model_id,
            "training_file": file_id,
            "hyperparameters": {
                "n_epochs": 1,
                "batch_size": 4 if self.model_size == "20B" else 1,
                "gradient_accumulation_steps": 4 if self.model_size == "20B" else 32,
                "learning_rate": 2e-4,  # Recommended for GPT-OSS
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine_with_min_lr",
                "use_qlora": False,  # GPT-OSS uses MXFP4 instead
                "quantization_type": "mxfp4",
                "gradient_checkpointing": True,
                # MoE specific
                "is_moe": True,
                "num_experts": 8 if self.model_size == "20B" else 64,
                "experts_per_token": 2,
                # LoRA for MoE
                "lora_rank": 8 if self.model_size == "20B" else 64,
                "lora_alpha": 16 if self.model_size == "20B" else 128,
                "lora_dropout": 0.1,
                "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            },
            "gpu_preference": "H100",  # H100 recommended for MoE
            "suffix": f"gpt-oss-{self.model_size.lower()}-moe"
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{SYNTH_API_URL}/api/fine_tuning/jobs",
                    json=job_payload,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    try:
                        text = response.text
                        print(f"  Response: {text[:200]}")
                        data = response.json() if text and text != "null" else {}
                    except Exception as e:
                        print(f"  Failed to parse response: {e}")
                        self.results.append(("moe_finetuning", {
                            "success": False,
                            "error": f"Invalid response: {e}"
                        }))
                        return False
                    
                    if not data or not isinstance(data, dict):
                        print(f"  Empty or invalid response")
                        self.results.append(("moe_finetuning", {
                            "success": False,
                            "error": "Empty response"
                        }))
                        return False
                    
                    job_id = data.get("id")
                    if job_id:
                        print(f"✓ MoE fine-tuning job created: {job_id}")
                        print(f"  GPU: {data.get('gpu', 'H100')}")
                        print(f"  Quantization: MXFP4")
                        print(f"  Experts: {job_payload['hyperparameters']['num_experts']}")
                        
                        self.results.append(("moe_finetuning", {
                            "success": True,
                            "job_id": job_id,
                            "note": "MoE job created with MXFP4 quantization"
                        }))
                        return True
                    else:
                        print("  No job ID in response")
                        self.results.append(("moe_finetuning", {
                            "success": False,
                            "error": "No job ID"
                        }))
                        return False
                else:
                    print(f"✗ Failed to create job: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    self.results.append(("moe_finetuning", {
                        "success": False,
                        "error": f"Status {response.status_code}"
                    }))
                    return False
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                self.results.append(("moe_finetuning", {"success": False, "error": str(e)}))
                return False
    
    async def test_streaming(self):
        """Test 5: Test streaming support for GPT-OSS models."""
        print("\n" + "="*60)
        print(f"TEST 5: Streaming - GPT-OSS {self.model_size}")
        print("="*60)
        
        test_prompt = "Write a short story about AI in 3 sentences."
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "temperature": 0.6,
            "max_tokens": 150,
            "stream": True,
            "gpu_preference": self.gpu_configs[0]
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                chunks_received = 0
                full_response = ""
                
                async with client.stream(
                    "POST",
                    f"{SYNTH_API_URL}/api/v1/chat/completions",
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                chunk_data = line[6:]
                                if chunk_data != "[DONE]":
                                    try:
                                        chunk = json.loads(chunk_data)
                                        if "choices" in chunk and chunk["choices"]:
                                            delta = chunk["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                full_response += delta["content"]
                                                chunks_received += 1
                                    except json.JSONDecodeError:
                                        pass
                        
                        if chunks_received > 0:
                            print(f"  ✓ Streaming successful")
                            print(f"    Chunks received: {chunks_received}")
                            print(f"    Response length: {len(full_response)} chars")
                            print(f"    Preview: {full_response[:100]}...")
                            
                            self.results.append(("streaming", {
                                "success": True,
                                "chunks": chunks_received
                            }))
                            return True
                        else:
                            print(f"  ✗ No streaming chunks received")
                            self.results.append(("streaming", {
                                "success": False,
                                "error": "No chunks"
                            }))
                            return False
                    else:
                        print(f"  ✗ Streaming failed: {response.status_code}")
                        self.results.append(("streaming", {
                            "success": False,
                            "error": f"Status {response.status_code}"
                        }))
                        return False
                        
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results.append(("streaming", {
                    "success": False,
                    "error": str(e)
                }))
                return False
    
    # Helper methods
    
    async def _wait_for_warmup(self, model_id: str, gpu: str, timeout: int = 180) -> bool:
        """Wait for model warmup completion."""
        start_time = time.time()
        headers = self.headers.copy()
        headers["X-GPU-Preference"] = gpu
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(
                        f"{SYNTH_API_URL}/api/warmup/status/{model_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "warmed":
                            return True
                        elif data.get("status") == "failed":
                            return False
                            
                except Exception:
                    pass
                
                await asyncio.sleep(3)
        
        return False
    
    async def _upload_training_file(self, data: List[Dict]) -> Optional[str]:
        """Upload training data file."""
        content = "\n".join([json.dumps(item) for item in data])
        
        async with httpx.AsyncClient(timeout=60) as client:
            files = {
                "file": ("training.jsonl", content, "application/jsonl"),
                "purpose": (None, "fine-tune")
            }
            
            try:
                response = await client.post(
                    f"{SYNTH_API_URL}/api/files",
                    files=files,
                    headers={"Authorization": self.headers["Authorization"]}
                )
                
                if response.status_code == 200:
                    return response.json().get("id")
                    
            except Exception as e:
                print(f"Upload error: {e}")
        
        return None
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print(f"TEST SUMMARY - GPT-OSS {self.model_size} on Synth Backend")
        print("="*60)
        
        for test_name, result in self.results:
            print(f"\n{test_name.upper()}:")
            
            if isinstance(result, list):
                successful = sum(1 for r in result if r.get("success"))
                total = len(result)
                print(f"  Success rate: {successful}/{total}")
                
                # Special handling for reasoning test
                if test_name == "inference_reasoning":
                    with_reasoning = sum(1 for r in result if r.get("has_reasoning"))
                    print(f"  With reasoning: {with_reasoning}/{successful}")
            elif isinstance(result, dict):
                if result.get("success"):
                    print(f"  ✓ Successful")
                    if "job_id" in result:
                        print(f"    Job ID: {result['job_id']}")
                    if "note" in result:
                        print(f"    Note: {result['note']}")
                    if "chunks" in result:
                        print(f"    Chunks: {result['chunks']}")
                else:
                    print(f"  ✗ Failed")
                    if "error" in result:
                        print(f"    Error: {result['error']}")
                    if "note" in result:
                        print(f"    Note: {result['note']}")
        
        # Overall summary
        total_tests = len(self.results)
        successful_tests = sum(1 for _, r in self.results 
                             if (isinstance(r, dict) and r.get("success")) or
                                (isinstance(r, list) and any(x.get("success") for x in r)))
        
        print(f"\n" + "="*60)
        print(f"Overall: {successful_tests}/{total_tests} test categories passed")
        
        if successful_tests == total_tests:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ {total_tests - successful_tests} test(s) need attention")
        print("="*60)


async def main():
    """Main test runner."""
    import sys
    
    # Allow model size selection via command line
    model_size = "20B"  # Default
    if len(sys.argv) > 1 and sys.argv[1] in GPT_OSS_MODELS:
        model_size = sys.argv[1]
    
    print("Synth Backend Test Suite - OpenAI GPT-OSS")
    print(f"Model Size: {model_size}")
    print(f"Model ID: {GPT_OSS_MODELS[model_size]}")
    print(f"GPU Requirements: {', '.join(GPU_CONFIGS[model_size])}")
    print(f"API URL: {SYNTH_API_URL}")
    print(f"API Key: {'Set' if SYNTH_API_KEY else 'Not set'}")
    
    if not SYNTH_API_KEY:
        print("\n⚠️  Warning: SYNTH_API_KEY not set")
        print("Set it with: export SYNTH_API_KEY=your_key")
    
    tester = SynthGPTOSSTester(model_size)
    
    # Run all tests in sequence
    await tester.test_warmup()
    await tester.test_inference_with_reasoning()
    await tester.test_native_tool_calling()
    await tester.test_moe_finetuning()
    await tester.test_streaming()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())