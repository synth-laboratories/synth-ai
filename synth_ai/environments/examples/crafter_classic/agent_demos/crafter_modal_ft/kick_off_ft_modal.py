#!/usr/bin/env python3
"""
Modal/Synth Fine-Tuning Script
==============================
Uploads a JSONL file to Modal, kicks off a fine-tuning job, and polls until completion.
Updated for OpenAI v1 compatible unified fine-tuning service.
"""

import os
import sys
import time
import argparse
import json
import random
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import httpx
from datetime import datetime

# Add synth_ai to path (optional - only if needed)
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Import Synth LM utilities (optional - only if needed)
# from synth_ai.lm import SynthConfig

# Modal fine-tuning endpoints - Updated for OpenAI v1 compatible service
MODAL_BASE_URL = os.getenv('MODAL_BASE_URL', os.getenv('SYNTH_BASE_URL', 'https://synth-laboratories--unified-ft-service-fastapi-app.modal.run'))
MODAL_API_KEY = os.getenv('MODAL_API_KEY', os.getenv('SYNTH_API_KEY', 'sk-test-11111111111111111111111111111111'))


def analyze_jsonl_tokens(file_path: Path, model: str) -> tuple[int, int, float]:
    """Analyze JSONL file to estimate token usage."""
    print(f"🔍 Analyzing {file_path.name} for token usage...")
    
    # For Modal/Synth, we'll do a rough estimate based on character count
    # Approximate: 1 token ≈ 4 characters (rough estimate)
    CHARS_PER_TOKEN = 4
    
    total_input_tokens = 0
    total_output_tokens = 0
    line_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                
                # Count input tokens (all messages except the last assistant message)
                input_chars = 0
                output_chars = 0
                
                for i, msg in enumerate(messages):
                    content = msg.get('content', '')
                    char_count = len(content)
                    
                    if msg.get('role') == 'assistant' and i == len(messages) - 1:
                        # This is the target output
                        output_chars += char_count
                    else:
                        # This is input context
                        input_chars += char_count
                    
                    # Include tool calls
                    tool_calls = msg.get('tool_calls', [])
                    for tc in tool_calls:
                        if tc.get('function', {}).get('arguments'):
                            if i == len(messages) - 1 and msg.get('role') == 'assistant':
                                output_chars += len(tc['function']['arguments'])
                            else:
                                input_chars += len(tc['function']['arguments'])
                
                input_tokens = input_chars // CHARS_PER_TOKEN
                output_tokens = output_chars // CHARS_PER_TOKEN
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                line_count += 1
                
            except json.JSONDecodeError:
                print(f"   ⚠️  Skipping invalid JSON line {line_count + 1}")
                continue
            except Exception as e:
                print(f"   ⚠️  Error processing line {line_count + 1}: {e}")
                continue
    
    total_tokens = total_input_tokens + total_output_tokens
    avg_tokens_per_line = total_tokens / line_count if line_count > 0 else 0
    
    print(f"   📊 Analysis complete:")
    print(f"      Lines: {line_count:,}")
    print(f"      Input tokens (est.): {total_input_tokens:,}")
    print(f"      Output tokens (est.): {total_output_tokens:,}")
    print(f"      Total tokens (est.): {total_tokens:,}")
    print(f"      Avg tokens/line: {avg_tokens_per_line:.1f}")
    
    return line_count, total_tokens, avg_tokens_per_line


def create_subset_file(original_path: Path, num_lines: int) -> Path:
    """Create a subset of the original JSONL file with specified number of lines."""
    subset_path = original_path.parent / f"{original_path.stem}_subset_{num_lines}.jsonl"
    
    print(f"📝 Creating subset with {num_lines} lines...")
    
    # Read all lines
    with open(original_path, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    # Randomly sample lines
    if num_lines >= len(all_lines):
        selected_lines = all_lines
        print(f"   ⚠️  Requested {num_lines} lines, but file only has {len(all_lines)}. Using all lines.")
    else:
        selected_lines = random.sample(all_lines, num_lines)
    
    # Write subset
    with open(subset_path, 'w') as f:
        for line in selected_lines:
            f.write(line + '\n')
    
    print(f"   ✅ Subset saved to: {subset_path.name}")
    return subset_path


async def upload_file(file_path: Path) -> str:
    """Upload training file to Modal using OpenAI v1 compatible endpoint."""
    print(f"📤 Uploading {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Create multipart form data for OpenAI v1 compatible endpoint
        files = {
            'file': (file_path.name, file_content, 'application/json')
        }
        
        # Use OpenAI v1 compatible endpoint
        response = await client.post(
            f"{MODAL_BASE_URL}/v1/files?purpose=fine-tune",
            files=files,
            headers={"Authorization": f"Bearer {MODAL_API_KEY}"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
        
        result = response.json()
        file_id = result.get('id')
        
        print(f"✅ File uploaded: {file_id}")
        return file_id


async def create_fine_tune_job(file_id: str, model: str = "Qwen/Qwen2.5-7B-Instruct", 
                              config: Optional[Dict[str, Any]] = None) -> str:
    """Create a fine-tuning job on Modal using OpenAI v1 compatible endpoint."""
    print(f"🚀 Starting fine-tune job for {model}...")
    
    # Default fine-tuning configuration for OpenAI v1 compatible service
    ft_config = {
        "model": model,
        "training_file": file_id,
        "training_type": "sft",  # sft or dpo
        "hyperparameters": {
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "use_qlora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
        },
        "suffix": f"modal-{int(time.time())}"
    }
    
    # Update with user config if provided
    if config:
        ft_config["hyperparameters"].update(config)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{MODAL_BASE_URL}/v1/fine_tuning/jobs",
            json=ft_config,
            headers={
                "Authorization": f"Bearer {MODAL_API_KEY}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Job creation failed: {response.status_code} - {response.text}")
        
        result = response.json()
        job_id = result.get('id')
        
        print(f"✅ Fine-tune job created: {job_id}")
        print(f"   Model: {model}")
        print(f"   Status: {result.get('status', 'created')}")
        
        return job_id


async def poll_job_status(job_id: str, poll_interval: int = 30) -> Optional[str]:
    """Poll job status until completion using OpenAI v1 compatible endpoint."""
    print(f"⏳ Polling job {job_id} every {poll_interval}s...")
    
    start_time = time.time()
    last_status = None
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                response = await client.get(
                    f"{MODAL_BASE_URL}/v1/fine_tuning/jobs/{job_id}",
                    headers={"Authorization": f"Bearer {MODAL_API_KEY}"}
                )
                
                if response.status_code != 200:
                    print(f"   ⚠️  Failed to get status: {response.status_code}")
                    await asyncio.sleep(poll_interval)
                    continue
                
                job = response.json()
                status = job.get('status', 'unknown')
                
                if status != last_status:
                    elapsed = time.time() - start_time
                    print(f"   Status: {status} (elapsed: {elapsed/60:.1f}m)")
                    last_status = status
                    
                    # Show training progress if available
                    if 'hyperparameters' in job:
                        hp = job['hyperparameters']
                        print(f"   Model: {job.get('model', 'unknown')}")
                        print(f"   Training file: {job.get('training_file', 'unknown')}")
                
                # Terminal states
                if status == "succeeded":
                    print(f"🎉 Fine-tuning completed successfully!")
                    final_model = job.get('fine_tuned_model')
                    if final_model:
                        print(f"   Final model: {final_model}")
                    return final_model
                
                elif status == "failed":
                    print(f"❌ Fine-tuning failed!")
                    if 'error' in job:
                        print(f"   Error: {job['error']}")
                    return None
                
                elif status == "cancelled":
                    print(f"⚠️  Fine-tuning was cancelled")
                    return None
                
                # Continue polling for running states
                elif status in ["queued", "running", "validating_files"]:
                    await asyncio.sleep(poll_interval)
                    continue
                
                else:
                    print(f"⚠️  Unknown status: {status}")
                    await asyncio.sleep(poll_interval)
                    continue
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user. Job {job_id} is still running on Modal.")
                print(f"   Check status later with the job ID: {job_id}")
                return None
            
            except Exception as e:
                print(f"❌ Error polling job: {e}")
                await asyncio.sleep(poll_interval)
                continue


async def main():
    parser = argparse.ArgumentParser(description="Modal/Synth Fine-Tuning Script")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL training file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--poll-interval", type=int, default=30, 
                       help="Polling interval in seconds (default: 30)")
    parser.add_argument("--subset", type=int, help="Use a random subset of N lines")
    parser.add_argument("--training-type", default="sft", choices=["sft", "dpo"],
                       help="Training type: sft (supervised) or dpo (preference)")
    
    args = parser.parse_args()
    
    # Validate file
    if not args.jsonl_file.exists():
        print(f"❌ File not found: {args.jsonl_file}")
        sys.exit(1)
    
    if not args.jsonl_file.suffix == '.jsonl':
        print(f"⚠️  Warning: File doesn't have .jsonl extension: {args.jsonl_file}")
    
    # Check API key
    if not MODAL_API_KEY:
        print("❌ Modal API key required. Set MODAL_API_KEY or SYNTH_API_KEY env var")
        sys.exit(1)
    
    # Analyze tokens first
    line_count, total_tokens, avg_tokens = analyze_jsonl_tokens(args.jsonl_file, args.model)
    
    # Estimate cost (Modal/Synth pricing varies by model)
    # This is a rough estimate - actual costs depend on Modal pricing
    estimated_hours = total_tokens / 1_000_000 * 0.5  # Rough estimate
    print(f"\n⏱️  Estimated training time: ~{estimated_hours:.1f} hours")
    print(f"   (Based on ~2M tokens/hour - actual time may vary)")
    
    # Use subset if requested
    training_file = args.jsonl_file
    if args.subset:
        if args.subset > line_count:
            print(f"⚠️  Subset size ({args.subset}) exceeds file size ({line_count}). Using all lines.")
        else:
            training_file = create_subset_file(args.jsonl_file, args.subset)
    
    print("\n🤖 Modal/Synth Fine-Tuning Pipeline")
    print("=" * 50)
    print(f"Training file: {training_file}")
    print(f"Base model: {args.model}")
    print(f"Training type: {args.training_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)
    
    try:
        # Step 1: Upload file
        file_id = await upload_file(training_file)
        
        # Step 2: Create fine-tune job
        ft_config = {
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
        job_id = await create_fine_tune_job(file_id, args.model, ft_config)
        
        # Step 3: Poll until completion
        final_model = await poll_job_status(job_id, args.poll_interval)
        
        if final_model:
            print("\n" + "=" * 50)
            print(f"🎯 SUCCESS! Fine-tuned model ready: {final_model}")
            print("=" * 50)
            
            # Show usage example
            print("\n📝 Usage example:")
            print(f'import httpx')
            print(f'')
            print(f'async def test_model():')
            print(f'    async with httpx.AsyncClient() as client:')
            print(f'        response = await client.post(')
            print(f'            "{MODAL_BASE_URL}/v1/chat/completions",')
            print(f'            headers={{"Authorization": f"Bearer {MODAL_API_KEY}"}},')
            print(f'            json={{')
            print(f'                "model": "{final_model}",')
            print(f'                "messages": [{{"role": "user", "content": "Hello!"}}]')
            print(f'            }}')
            print(f'        )')
            print(f'        return response.json()')
        else:
            print("\n❌ Fine-tuning did not complete successfully")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())