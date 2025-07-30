#!/usr/bin/env python3
"""
Script to kick off fine-tuning jobs on Modal using generated Crafter rollout data
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# Modal API configuration
MODAL_BASE_URL = "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run"
MODAL_API_KEY = os.environ.get("MODAL_API_KEY", "sk-test-11111111111111111111111111111111")

# Default hyperparameters for Crafter fine-tuning
DEFAULT_HYPERPARAMS = {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 2.0,
    "use_qlora": False,  # Can enable for larger models
}

# Supported base models for fine-tuning
SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]


async def upload_training_file(file_path: Path, api_key: str) -> str:
    """Upload a training file to Modal and return the file ID."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        print(f"📤 Uploading {file_path.name}...")
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/jsonl")}
            response = await client.post(
                f"{MODAL_BASE_URL}/v1/files?purpose=fine-tune",
                files=files,
                headers=headers
            )
        
        if response.status_code != 200:
            raise Exception(f"Failed to upload file: {response.status_code} - {response.text}")
        
        file_data = response.json()
        file_id = file_data["id"]
        print(f"✅ Uploaded successfully: {file_id}")
        return file_id


async def create_fine_tuning_job(
    base_model: str,
    training_file_id: str,
    suffix: str,
    hyperparameters: dict,
    api_key: str,
    validation_file_id: Optional[str] = None
) -> str:
    """Create a fine-tuning job and return the job ID."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": base_model,
            "training_file": training_file_id,
            "hyperparameters": hyperparameters,
            "suffix": suffix
        }
        
        if validation_file_id:
            payload["validation_file"] = validation_file_id
        
        print(f"🚀 Creating fine-tuning job...")
        print(f"   Base model: {base_model}")
        print(f"   Suffix: {suffix}")
        print(f"   Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
        
        response = await client.post(
            f"{MODAL_BASE_URL}/v1/fine_tuning/jobs",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to create job: {response.status_code} - {response.text}")
        
        job_data = response.json()
        job_id = job_data["id"]
        print(f"✅ Job created: {job_id}")
        return job_id


async def monitor_job(job_id: str, api_key: str) -> dict:
    """Monitor a fine-tuning job until completion."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        print(f"\n📊 Monitoring job {job_id}...")
        
        while True:
            # Get job status
            response = await client.get(
                f"{MODAL_BASE_URL}/v1/fine_tuning/jobs/{job_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"⚠️  Failed to get job status: {response.text}")
                await asyncio.sleep(10)
                continue
            
            job_data = response.json()
            status = job_data["status"]
            
            # Print status update
            print(f"   Status: {status}")
            
            # Get recent events
            events_response = await client.get(
                f"{MODAL_BASE_URL}/v1/fine_tuning/jobs/{job_id}/events?limit=5",
                headers=headers
            )
            
            if events_response.status_code == 200:
                events = events_response.json()["data"]
                for event in events[:2]:  # Show last 2 events
                    print(f"   - {event.get('message', 'No message')}")
            
            # Check if job is complete
            if status in ["succeeded", "failed", "cancelled"]:
                if status == "succeeded":
                    print(f"\n✅ Fine-tuning completed successfully!")
                    print(f"   Model ID: {job_data['fine_tuned_model']}")
                else:
                    print(f"\n❌ Fine-tuning {status}")
                    if job_data.get("error"):
                        print(f"   Error: {job_data['error']}")
                
                return job_data
            
            # Wait before next check
            await asyncio.sleep(30)


async def validate_training_data(file_path: Path) -> tuple[int, int]:
    """Validate training data and return (num_examples, num_tokens_estimate)."""
    num_examples = 0
    total_chars = 0
    
    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "messages" in data:
                    num_examples += 1
                    # Rough token estimate (chars/4)
                    total_chars += len(json.dumps(data["messages"]))
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON line: {line[:50]}...")
    
    estimated_tokens = total_chars // 4
    return num_examples, estimated_tokens


async def main():
    parser = argparse.ArgumentParser(description="Kick off fine-tuning jobs on Modal")
    parser.add_argument("training_file", type=str, help="Path to training data (JSONL)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       choices=SUPPORTED_MODELS, help="Base model to fine-tune")
    parser.add_argument("--suffix", type=str, default=None,
                       help="Model suffix (default: crafter-TIMESTAMP)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2.0,
                       help="Learning rate multiplier")
    parser.add_argument("--use-qlora", action="store_true",
                       help="Enable QLoRA for efficient training")
    parser.add_argument("--validation-file", type=str, default=None,
                       help="Optional validation data file")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Modal API key (or set MODAL_API_KEY)")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't monitor job after creation")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("MODAL_API_KEY", MODAL_API_KEY)
    
    # Generate suffix if not provided
    if args.suffix is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.suffix = f"crafter-{timestamp}"
    
    # Validate training file
    training_path = Path(args.training_file)
    if not training_path.exists():
        print(f"❌ Training file not found: {training_path}")
        return
    
    print(f"🔍 Validating training data...")
    num_examples, est_tokens = await validate_training_data(training_path)
    print(f"   Examples: {num_examples}")
    print(f"   Estimated tokens: {est_tokens:,}")
    
    if num_examples < 10:
        print("⚠️  Warning: Very few training examples. Consider generating more data.")
    
    # Prepare hyperparameters
    hyperparams = {
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate_multiplier": args.learning_rate,
        "use_qlora": args.use_qlora
    }
    
    try:
        # Upload training file
        training_file_id = await upload_training_file(training_path, api_key)
        
        # Upload validation file if provided
        validation_file_id = None
        if args.validation_file:
            val_path = Path(args.validation_file)
            if val_path.exists():
                validation_file_id = await upload_training_file(val_path, api_key)
        
        # Create fine-tuning job
        job_id = await create_fine_tuning_job(
            base_model=args.base_model,
            training_file_id=training_file_id,
            suffix=args.suffix,
            hyperparameters=hyperparams,
            api_key=api_key,
            validation_file_id=validation_file_id
        )
        
        # Monitor job unless disabled
        if not args.no_monitor:
            job_data = await monitor_job(job_id, api_key)
            
            if job_data["status"] == "succeeded":
                print("\n🎉 Fine-tuning complete!")
                print(f"Your model is ready: {job_data['fine_tuned_model']}")
                print("\nTo use your model:")
                print(f"  curl -X POST {MODAL_BASE_URL}/v1/chat/completions \\")
                print(f"    -H 'Authorization: Bearer YOUR_API_KEY' \\")
                print(f"    -H 'Content-Type: application/json' \\")
                print(f"    -d '{{")
                print(f'      "model": "{job_data["fine_tuned_model"]}",')
                print(f'      "messages": [{{"role": "user", "content": "Hello!"}}]')
                print(f"    }}'")
        else:
            print(f"\nJob created: {job_id}")
            print(f"Check status at: {MODAL_BASE_URL}/v1/fine_tuning/jobs/{job_id}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())