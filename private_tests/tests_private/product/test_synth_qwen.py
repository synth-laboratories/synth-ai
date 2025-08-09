#!/usr/bin/env python3
"""
Test Synth backend with Qwen3 0.6B model.
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
MODEL_ID = "Qwen/Qwen3-0.6B"  # Target model for all tests

# GPU configurations to test (Qwen3-0.6B only supports A10G and L40S)
GPU_CONFIGS = ["L40S", "A10G"]

# Training data for fine-tuning
TRAINING_DATA = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain neural networks."},
            {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is deep learning?"},
            {"role": "assistant", "content": "Deep learning uses multi-layered neural networks."}
        ]
    }
]

# DPO training data
DPO_DATA = [
    {
        "prompt": "Explain quantum computing",
        "chosen": "Quantum computing uses qubits in superposition for parallel processing.",
        "rejected": "Quantum computing is just fast computers."
    },
    {
        "prompt": "What is AI?",
        "chosen": "AI simulates human intelligence in machines for learning and problem-solving.",
        "rejected": "AI is robots."
    }
]


class SynthQwenTester:
    """Test Synth backend with Qwen3 0.6B."""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {SYNTH_API_KEY}",
            "Content-Type": "application/json"
        }
        self.results = []
        self.finetuned_model_id = None
        self.dpo_model_id = None
    
    async def test_warmup(self):
        """Test 1: Warm up Qwen3 0.6B on different GPUs."""
        print("\n" + "="*60)
        print("TEST 1: GPU Warmup for Qwen3 0.6B")
        print("="*60)
        
        warmup_results = []
        
        for gpu in GPU_CONFIGS:
            print(f"\nWarming up on {gpu}...")
            
            headers = self.headers.copy()
            headers["X-GPU-Preference"] = gpu
            
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    # Initiate warmup
                    start_time = time.time()
                    response = await client.post(
                        f"{SYNTH_API_URL}/api/warmup/{MODEL_ID}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"  Status: {data.get('status')}")
                        print(f"  GPU: {data.get('gpu', gpu)}")
                        
                        # Wait for warmup completion
                        warmed = await self._wait_for_warmup(MODEL_ID, gpu)
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
        print(f"\n✓ Warmup completed: {successful}/{len(GPU_CONFIGS)} GPUs ready")
        self.results.append(("warmup", warmup_results))
        return successful > 0
    
    async def test_inference(self):
        """Test 2: Run inference with Qwen3 0.6B on different GPUs."""
        print("\n" + "="*60)
        print("TEST 2: Inference with Qwen3 0.6B")
        print("="*60)
        
        test_prompts = [
            "What is 2+2?",
            "Name the capital of France.",
            "Write a haiku about AI."
        ]
        
        inference_results = []
        
        for gpu in GPU_CONFIGS[:2]:  # Test on first 2 GPUs
            print(f"\nTesting inference on {gpu}...")
            
            for i, prompt in enumerate(test_prompts, 1):
                payload = {
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
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
                            selected_gpu = "unknown"
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
    
    async def test_finetuning(self):
        """Test 3: Fine-tune Qwen3 0.6B."""
        print("\n" + "="*60)
        print("TEST 3: Fine-tuning Qwen3 0.6B")
        print("="*60)
        
        # Upload training data
        print("\nUploading training data...")
        file_id = await self._upload_training_file(TRAINING_DATA)
        
        if not file_id:
            print("✗ Failed to upload training data")
            self.results.append(("finetuning", {"success": False, "error": "Upload failed"}))
            return False
        
        print(f"✓ Training file uploaded: {file_id}")
        
        # Create fine-tuning job
        print("\nCreating fine-tuning job...")
        job_payload = {
            "model": MODEL_ID,
            "training_file": file_id,
            "hyperparameters": {
                "n_epochs": 1,
                "batch_size": 2,
                "learning_rate": 5e-5
            },
            "gpu_preference": "L40S",
            "suffix": "qwen-test"
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
                        print(f"  Response text: {text[:200]}")
                        data = response.json() if text else {}
                    except Exception as e:
                        print(f"  Failed to parse response: {e}")
                        print(f"  Raw response: {response.text[:500]}")
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": f"Invalid response format: {e}"
                        }))
                        return False
                    
                    if not data or not isinstance(data, dict):
                        print(f"  Empty or invalid response data: {data}")
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": "Empty response"
                        }))
                        return False
                    
                    job_id = data.get("id")
                    if not job_id:
                        print(f"  No job ID in response: {data}")
                        self.results.append(("finetuning", {
                            "success": False,
                            "error": "No job ID in response"
                        }))
                        return False
                    
                    print(f"✓ Fine-tuning job created: {job_id}")
                    print(f"  GPU: {data.get('gpu', 'L40S')}")
                    
                    # Wait for completion
                    print("  Waiting for training to complete...")
                    result = await self._wait_for_job(job_id, "fine_tuning")
                    
                    if result["status"] in ["succeeded", "completed"]:
                        self.finetuned_model_id = result.get("fine_tuned_model")
                        print(f"✓ Fine-tuning completed!")
                        print(f"  Model ID: {self.finetuned_model_id}")
                        
                        self.results.append(("finetuning", {
                            "success": True,
                            "model_id": self.finetuned_model_id,
                            "job_id": job_id
                        }))
                        return True
                    else:
                        print(f"✗ Fine-tuning failed: {result['status']}")
                        self.results.append(("finetuning", {
                            "success": False,
                            "status": result["status"]
                        }))
                        return False
                else:
                    print(f"✗ Failed to create job: {response.status_code}")
                    print(f"  Response: {response.text[:500]}")
                    self.results.append(("finetuning", {
                        "success": False,
                        "error": f"Status {response.status_code}: {response.text[:200]}"
                    }))
                    return False
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                self.results.append(("finetuning", {"success": False, "error": str(e)}))
                return False
    
    async def test_dpo(self):
        """Test 4: DPO training with Qwen3 0.6B."""
        print("\n" + "="*60)
        print("TEST 4: DPO Training with Qwen3 0.6B")
        print("="*60)
        
        # Upload DPO data
        print("\nUploading DPO training data...")
        file_id = await self._upload_dpo_file(DPO_DATA)
        
        if not file_id:
            print("✗ Failed to upload DPO data")
            self.results.append(("dpo", {"success": False, "error": "Upload failed"}))
            return False
        
        print(f"✓ DPO file uploaded: {file_id}")
        
        # Create DPO job
        print("\nCreating DPO training job...")
        job_payload = {
            "model": MODEL_ID,
            "training_file": file_id,
            "hyperparameters": {
                "beta": 0.1,
                "n_epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-5,
                "training_type": "dpo"
            },
            "gpu_preference": "L40S"
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
                        print(f"  Response text: {text[:200]}")
                        data = response.json() if text else {}
                    except Exception as e:
                        print(f"  Failed to parse response: {e}")
                        print(f"  Raw response: {response.text[:500]}")
                        self.results.append(("dpo", {
                            "success": False,
                            "error": f"Invalid response format: {e}"
                        }))
                        return False
                    
                    if not data or not isinstance(data, dict):
                        print(f"  Empty or invalid response data: {data}")
                        self.results.append(("dpo", {
                            "success": False,
                            "error": "Empty response"
                        }))
                        return False
                    
                    job_id = data.get("id")
                    if not job_id:
                        print(f"  No job ID in response: {data}")
                        self.results.append(("dpo", {
                            "success": False,
                            "error": "No job ID in response"
                        }))
                        return False
                    
                    print(f"✓ DPO job created: {job_id}")
                    print(f"  GPU: {data.get('gpu', 'L40S')}")
                    if 'hyperparameters' in data:
                        print(f"  Beta: {data['hyperparameters'].get('beta', 0.1)}")
                    
                    # Wait for completion
                    print("  Waiting for DPO training to complete...")
                    result = await self._wait_for_job(job_id, "dpo")
                    
                    if result["status"] in ["succeeded", "completed"]:
                        self.dpo_model_id = result.get("dpo_model", result.get("fine_tuned_model"))
                        print(f"✓ DPO training completed!")
                        print(f"  Model ID: {self.dpo_model_id}")
                        
                        self.results.append(("dpo", {
                            "success": True,
                            "model_id": self.dpo_model_id,
                            "job_id": job_id
                        }))
                        return True
                    else:
                        print(f"✗ DPO training failed: {result['status']}")
                        self.results.append(("dpo", {
                            "success": False,
                            "status": result["status"]
                        }))
                        return False
                else:
                    print(f"✗ Failed to create DPO job: {response.status_code}")
                    print(f"  Response: {response.text[:500]}")
                    self.results.append(("dpo", {
                        "success": False,
                        "error": f"Status {response.status_code}: {response.text[:200]}"
                    }))
                    return False
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                self.results.append(("dpo", {"success": False, "error": str(e)}))
                return False
    
    async def test_finetuned_inference(self):
        """Test 5: Run inference with fine-tuned models."""
        print("\n" + "="*60)
        print("TEST 5: Inference with Fine-tuned Models")
        print("="*60)
        
        # Allow Modal volume replication time after training
        await asyncio.sleep(10)
        models_to_test = []
        if self.finetuned_model_id:
            models_to_test.append(("Fine-tuned", self.finetuned_model_id))
        if self.dpo_model_id:
            models_to_test.append(("DPO", self.dpo_model_id))
        
        if not models_to_test:
            print("✗ No fine-tuned models available for testing")
            self.results.append(("finetuned_inference", {"success": False, "error": "No models"}))
            return False
        
        test_prompt = "Explain the benefits of machine learning."
        inference_results = []
        
        for model_type, model_id in models_to_test:
            print(f"\nTesting {model_type} model: {model_id[:40]}...")
            
            # Skipping explicit warm-up; inference will lazy-load the model
            # (warm-up removed to avoid volume replication race conditions)
            
            # Run inference
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 150,
                "gpu_preference": "L40S"
            }
            
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    retries = 0
                    max_retries = 5
                    while retries < max_retries:
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
                            print(f"  ✓ Inference successful (try {retries+1})")
                            print(f"    Response: {content[:100]}...")
                            print(f"    Time: {elapsed:.2f}s")
                            inference_results.append({
                                "model_type": model_type,
                                "success": True,
                                "time": elapsed,
                                "response_length": len(content)
                            })
                            break
                        else:
                            # If adapter not yet replicated, wait then retry
                            if response.status_code == 500 and ("adapter_config.json" in response.text or "Fine-tuned model adapter not found" in response.text):
                                wait_sec = 5 * (retries + 1)
                                print(f"  🕒 Adapter not ready, waiting {wait_sec}s then retrying...")
                                await asyncio.sleep(wait_sec)
                                retries += 1
                                continue
                            print(f"  ✗ Inference failed: {response.status_code}")
                            inference_results.append({
                                "model_type": model_type,
                                "success": False,
                                "error": f"Status {response.status_code}"
                            })
                            break
                        
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    inference_results.append({
                        "model_type": model_type,
                        "success": False,
                        "error": str(e)
                    })
        
        successful = sum(1 for r in inference_results if r.get("success"))
        print(f"\n✓ Fine-tuned inference completed: {successful}/{len(inference_results)} successful")
        self.results.append(("finetuned_inference", inference_results))
        return successful > 0
    
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
                
                await asyncio.sleep(2)
        
        return False
    
    async def _warmup_model(self, model_id: str, gpu: str, timeout: int = 180) -> bool:
        """Warm up a specific model."""
        headers = self.headers.copy()
        headers["X-GPU-Preference"] = gpu
        
        async with httpx.AsyncClient(timeout=30) as client:
            start = time.time()
            while time.time() - start < timeout:
                try:
                    response = await client.post(
                        f"{SYNTH_API_URL}/api/warmup/{model_id}",
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        return await self._wait_for_warmup(model_id, gpu, timeout=timeout)
                    elif response.status_code == 404:
                        # Model not yet visible – adapter directory likely not replicated
                        print(f"    🕒 Warm-up 404 – retrying in 4s...")
                        await asyncio.sleep(4)
                        continue
                    else:
                        print(f"    Warm-up error: {response.status_code} – {response.text[:200]}")
                        return False
                        
                except Exception as e:
                    print(f"    Warm-up request error: {e} – retrying in 4s")
                    await asyncio.sleep(4)
            
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
    
    async def _upload_dpo_file(self, data: List[Dict]) -> Optional[str]:
        """Upload DPO training data file."""
        content = "\n".join([json.dumps(item) for item in data])
        
        async with httpx.AsyncClient(timeout=60) as client:
            files = {
                "file": ("dpo_data.jsonl", content, "application/jsonl"),
                "purpose": (None, "dpo-training")
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
    
    async def _wait_for_job(self, job_id: str, job_type: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for training job completion."""
        start_time = time.time()
        
        # Both SFT and DPO jobs use the same fine_tuning/jobs endpoint in learning_v2
        endpoint = f"/api/fine_tuning/jobs/{job_id}"
        
        last_status = None
        check_count = 0
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                check_count += 1
                try:
                    response = await client.get(
                        f"{SYNTH_API_URL}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status", "unknown")
                        
                        # Print status updates
                        if status != last_status:
                            elapsed = int(time.time() - start_time)
                            print(f"    [{elapsed}s] Status: {status}")
                            
                            # Print additional info if available
                            if "trained_tokens" in data:
                                print(f"        Trained tokens: {data['trained_tokens']}")
                            if "error" in data and data["error"]:
                                print(f"        Error: {data['error']}")
                            
                            last_status = status
                        
                        # Check for terminal states
                        if status in ["succeeded", "completed"]:
                            print(f"    ✓ Job completed successfully after {int(time.time() - start_time)}s")
                            return data
                        elif status in ["failed", "cancelled"]:
                            print(f"    ✗ Job {status} after {int(time.time() - start_time)}s")
                            if "error" in data:
                                print(f"        Reason: {data.get('error', 'Unknown')}")
                            return data
                    else:
                        print(f"    Status check failed: {response.status_code}")
                        if check_count % 3 == 0:  # Print error every 3rd attempt
                            print(f"        Response: {response.text[:200]}")
                            
                except Exception as e:
                    if check_count % 3 == 0:  # Print error every 3rd attempt
                        print(f"    Check error: {e}")
                
                # Adaptive sleep - shorter at beginning, longer as time goes on
                elapsed = int(time.time() - start_time)
                sleep_time = 5 if elapsed < 60 else 10 if elapsed < 300 else 15
                await asyncio.sleep(sleep_time)
        
        print(f"    ⏰ Timeout after {timeout}s - job still running")
        return {"status": "timeout"}
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY - Qwen3 0.6B on Synth Backend")
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
                    if "model_id" in result:
                        print(f"    Model: {result['model_id']}")
                else:
                    print(f"  ✗ Failed")
                    if "error" in result:
                        print(f"    Error: {result['error']}")
        
        # Overall summary
        total_tests = 5
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
    print("Synth Backend Test Suite")
    print(f"Model: {MODEL_ID}")
    print(f"API URL: {SYNTH_API_URL}")
    print(f"API Key: {'Set' if SYNTH_API_KEY else 'Not set'}")
    
    if not SYNTH_API_KEY:
        print("\n⚠️  Warning: SYNTH_API_KEY not set")
        print("Set it with: export SYNTH_API_KEY=your_key")
    
    tester = SynthQwenTester()
    
    # Run all tests in sequence
    await tester.test_warmup()
    await tester.test_inference()
    await tester.test_finetuning()
    await tester.test_dpo()
    await tester.test_finetuned_inference()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())