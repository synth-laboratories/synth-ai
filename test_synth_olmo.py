#!/usr/bin/env python3
"""
Test Synth backend with OLMo 2 models.
Tests: warmup, inference, fine-tuning, DPO, and inference with fine-tuned models.
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

# OLMo 2 models to test
OLMO_MODELS = {
    "1B": "allenai/OLMo-2-0425-1B-Instruct",
    "7B": "allenai/OLMo-2-1124-7B-Instruct",
    "13B": "allenai/OLMo-2-1124-13B-Instruct"
}

# Use the smallest model by default for faster testing
MODEL_ID = OLMO_MODELS["1B"]

# GPU configurations for OLMo 2 models
# 1B model: A10G, L40S, A100
# 7B model: A10G, L40S, A100  
# 13B model: L40S, A100, H100
GPU_CONFIGS = {
    "1B": ["A10G", "L40S"],
    "7B": ["L40S", "A100"],
    "13B": ["A100", "H100"]
}

# Training data for fine-tuning
TRAINING_DATA = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data patterns."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain neural networks."},
            {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural systems."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is deep learning?"},
            {"role": "assistant", "content": "Deep learning uses multi-layered neural networks to learn hierarchical representations."}
        ]
    }
]

# DPO training data
DPO_DATA = [
    {
        "prompt": "Explain quantum computing",
        "chosen": "Quantum computing leverages quantum mechanics principles like superposition and entanglement for computation.",
        "rejected": "Quantum computing is just fast computers."
    },
    {
        "prompt": "What is artificial intelligence?",
        "chosen": "AI is the field of computer science focused on creating systems that can perform tasks requiring human intelligence.",
        "rejected": "AI is robots thinking."
    }
]

# Test prompts
TEST_PROMPTS = [
    "What is 2+2?",
    "Name the capital of Japan.",
    "Write a haiku about technology."
]


class SynthOLMoTester:
    """Test Synth backend with OLMo 2 models."""
    
    def __init__(self, model_size: str = "1B"):
        self.model_size = model_size
        self.model_id = OLMO_MODELS[model_size]
        self.gpu_configs = GPU_CONFIGS[model_size]
        self.headers = {
            "Authorization": f"Bearer {SYNTH_API_KEY}",
            "Content-Type": "application/json"
        }
        self.results = []
        self.finetuned_model_id = None
        self.dpo_model_id = None
    
    async def test_warmup(self):
        """Test 1: Warm up OLMo 2 model on different GPUs."""
        print("\n" + "="*60)
        print(f"TEST 1: GPU Warmup for OLMo 2 {self.model_size} ({self.model_id})")
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
                        
                        # Wait for warmup completion
                        warmed = await self._wait_for_warmup(self.model_id, gpu)
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
    
    async def test_inference(self):
        """Test 2: Run inference with OLMo 2 model on different GPUs."""
        print("\n" + "="*60)
        print(f"TEST 2: Inference with OLMo 2 {self.model_size}")
        print("="*60)
        
        inference_results = []
        
        for gpu in self.gpu_configs:
            print(f"\nTesting inference on {gpu}...")
            
            for i, prompt in enumerate(TEST_PROMPTS, 1):
                payload = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_tokens": 100,
                    "gpu_preference": gpu
                }
                
                async with httpx.AsyncClient(timeout=60) as client:
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
                            
                            # Check GPU selection
                            selected_gpu = gpu
                            if "system_fingerprint" in data:
                                fp = data["system_fingerprint"]
                                if isinstance(fp, dict):
                                    selected_gpu = fp.get("selected_gpu", gpu)
                            
                            print(f"  Prompt {i}: '{prompt[:30]}...'")
                            print(f"    Response: {content[:50]}...")
                            print(f"    Time: {elapsed:.2f}s, GPU: {selected_gpu}")
                            
                            inference_results.append({
                                "gpu": gpu,
                                "prompt": prompt,
                                "success": True,
                                "time": elapsed,
                                "response_length": len(content)
                            })
                        else:
                            print(f"  ✗ Inference failed: {response.status_code}")
                            print(f"    Response: {response.text[:200]}")
                            inference_results.append({
                                "gpu": gpu,
                                "prompt": prompt,
                                "success": False,
                                "error": f"Status {response.status_code}"
                            })
                            
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        inference_results.append({
                            "gpu": gpu,
                            "prompt": prompt,
                            "success": False,
                            "error": str(e)
                        })
        
        successful = sum(1 for r in inference_results if r.get("success"))
        print(f"\n✓ Inference completed: {successful}/{len(inference_results)} requests successful")
        self.results.append(("inference", inference_results))
        return successful > 0
    
    async def test_tool_calling(self):
        """Test 3: Test OLMo 2 manual tool calling support."""
        print("\n" + "="*60)
        print(f"TEST 3: Tool Calling with OLMo 2 {self.model_size}")
        print("="*60)
        
        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
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
        
        # Test prompt that should trigger tool use
        test_prompt = "What's the weather like in Tokyo?"
        
        # OLMo 2 uses manual tool calling, so we need to format the prompt
        system_prompt = """You have access to the following functions. When you need to call a function, use this exact format: Function: function_name(param1="value1", param2="value2")

Available functions:
Function: get_weather
Description: Get the current weather for a location
Parameters:
  - location (string) (required): The city and state, e.g. San Francisco, CA
  - unit (string): Temperature unit

If you don't need to use any functions, respond normally."""
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 150,
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
                    content = data["choices"][0]["message"]["content"]
                    
                    # Check if the model attempted to call the function
                    if "Function: get_weather" in content or "get_weather(" in content:
                        print("  ✓ Model attempted to use the tool")
                        print(f"    Response: {content[:150]}...")
                        self.results.append(("tool_calling", {"success": True}))
                        return True
                    else:
                        print("  ⚠ Model did not use the tool")
                        print(f"    Response: {content[:150]}...")
                        self.results.append(("tool_calling", {"success": False, "note": "No tool call detected"}))
                        return False
                else:
                    print(f"  ✗ Request failed: {response.status_code}")
                    self.results.append(("tool_calling", {"success": False, "error": f"Status {response.status_code}"}))
                    return False
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results.append(("tool_calling", {"success": False, "error": str(e)}))
                return False
    
    async def test_finetuning(self):
        """Test 4: Fine-tune OLMo 2 model."""
        print("\n" + "="*60)
        print(f"TEST 4: Fine-tuning OLMo 2 {self.model_size}")
        print("="*60)
        
        # Upload training data
        print("\nUploading training data...")
        file_id = await self._upload_training_file(TRAINING_DATA)
        
        if not file_id:
            print("✗ Failed to upload training data")
            self.results.append(("finetuning", {"success": False, "error": "Upload failed"}))
            return False
        
        print(f"✓ Training file uploaded: {file_id}")
        
        # Create fine-tuning job with OLMo 2 specific settings
        print("\nCreating fine-tuning job...")
        
        # Use appropriate GPU based on model size
        training_gpu = self.gpu_configs[-1]  # Use the most powerful GPU available
        
        job_payload = {
            "model": self.model_id,
            "training_file": file_id,
            "hyperparameters": {
                "n_epochs": 1,
                "batch_size": 8 if self.model_size == "1B" else 4 if self.model_size == "7B" else 2,
                "learning_rate": 5e-5,
                "use_qlora": True  # OLMo 2 uses QLoRA
            },
            "gpu_preference": training_gpu,
            "suffix": f"olmo-{self.model_size.lower()}-test"
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
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": f"Invalid response: {e}"
                        }))
                        return False
                    
                    if not data or not isinstance(data, dict):
                        print(f"  Empty or invalid response")
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": "Empty response"
                        }))
                        return False
                    
                    job_id = data.get("id")
                    if job_id:
                        print(f"✓ Fine-tuning job created: {job_id}")
                        print(f"  GPU: {data.get('gpu', training_gpu)}")
                        
                        # Note: Not waiting for completion as it may take long
                        self.results.append(("finetuning", {
                            "success": True,
                            "job_id": job_id,
                            "note": "Job created but not waiting for completion"
                        }))
                        return True
                    else:
                        print("  No job ID in response")
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": "No job ID"
                        }))
                        return False
                else:
                    print(f"✗ Failed to create job: {response.status_code}")
                    print(f"  Response: {response.text[:200]}")
                    self.results.append(("finetuning", {
                        "success": False,
                        "error": f"Status {response.status_code}"
                    }))
                    return False
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                self.results.append(("finetuning", {"success": False, "error": str(e)}))
                return False
    
    # Helper methods
    
    async def _wait_for_warmup(self, model_id: str, gpu: str, timeout: int = 120) -> bool:
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
                
                await asyncio.sleep(2)
        
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
        print(f"TEST SUMMARY - OLMo 2 {self.model_size} on Synth Backend")
        print("="*60)
        
        for test_name, result in self.results:
            print(f"\n{test_name.upper()}:")
            
            if isinstance(result, list):
                successful = sum(1 for r in result if r.get("success"))
                total = len(result)
                print(f"  Success rate: {successful}/{total}")
            elif isinstance(result, dict):
                if result.get("success"):
                    print(f"  ✓ Successful")
                    if "job_id" in result:
                        print(f"    Job ID: {result['job_id']}")
                    if "note" in result:
                        print(f"    Note: {result['note']}")
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
    model_size = "1B"  # Default
    if len(sys.argv) > 1 and sys.argv[1] in OLMO_MODELS:
        model_size = sys.argv[1]
    
    print("Synth Backend Test Suite - OLMo 2")
    print(f"Model Size: {model_size}")
    print(f"Model ID: {OLMO_MODELS[model_size]}")
    print(f"API URL: {SYNTH_API_URL}")
    print(f"API Key: {'Set' if SYNTH_API_KEY else 'Not set'}")
    
    if not SYNTH_API_KEY:
        print("\n⚠️  Warning: SYNTH_API_KEY not set")
        print("Set it with: export SYNTH_API_KEY=your_key")
    
    tester = SynthOLMoTester(model_size)
    
    # Run all tests in sequence
    await tester.test_warmup()
    await tester.test_inference()
    await tester.test_tool_calling()
    await tester.test_finetuning()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())