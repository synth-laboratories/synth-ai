#!/usr/bin/env python3
"""
OpenAI Fine-Tuning Script
========================
Uploads a JSONL file, kicks off a fine-tuning job, and polls until completion.
"""

import os
import sys
import time
import argparse
import json
import random
from pathlib import Path
from typing import Optional

try:
    import openai
except ImportError:
    print("❌ OpenAI package not found. Installing...")
    os.system("pip install openai")
    import openai

try:
    import tiktoken
except ImportError:
    print("❌ tiktoken package not found. Installing...")
    os.system("pip install tiktoken")
    import tiktoken


def encoding_for(model: str = "gpt-4.1-mini"):
    """Return a tiktoken encoding for any GPT‑4.1 family model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:                 # 4.1 isn't mapped yet
        return tiktoken.get_encoding("o200k_base")  # same BPE as 4o/modern models


def analyze_jsonl_tokens(file_path: Path, model: str) -> tuple[int, int, float]:
    """Analyze JSONL file to estimate token usage."""
    print(f"🔍 Analyzing {file_path.name} for token usage...")
    
    # Get the appropriate encoding for the model (handles GPT-4.1 properly)
    encoding = encoding_for(model)
    print(f"   🔤 Using encoding: {encoding.name}")
    
    total_input_tokens = 0
    total_output_tokens = 0
    line_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                
                # Count input tokens (all messages except the last assistant message)
                input_messages = []
                output_message = None
                
                for msg in messages:
                    if msg.get('role') == 'assistant' and msg == messages[-1]:
                        # This is the target output
                        output_message = msg
                    else:
                        # This is input context
                        input_messages.append(msg)
                
                # Estimate input tokens
                input_text = ""
                for msg in input_messages:
                    content = msg.get('content', '')
                    if content:
                        input_text += content + " "
                    
                    # Include tool calls in input if present
                    tool_calls = msg.get('tool_calls', [])
                    for tc in tool_calls:
                        if tc.get('function', {}).get('arguments'):
                            input_text += tc['function']['arguments'] + " "
                
                input_tokens = len(encoding.encode(input_text))
                total_input_tokens += input_tokens
                
                # Estimate output tokens
                output_tokens = 0
                if output_message:
                    content = output_message.get('content', '') or ''
                    output_tokens += len(encoding.encode(content))
                    
                    # Include tool calls in output
                    tool_calls = output_message.get('tool_calls', [])
                    for tc in tool_calls:
                        if tc.get('function', {}).get('arguments'):
                            output_tokens += len(encoding.encode(tc['function']['arguments']))
                
                total_output_tokens += output_tokens
                line_count += 1
                
            except json.JSONDecodeError:
                print(f"   ⚠️  Skipping invalid JSON line {line_count + 1}")
                continue
            except Exception as e:
                print(f"   ⚠️  Error processing line {line_count + 1}: {e}")
                continue
    
    avg_tokens_per_line = (total_input_tokens + total_output_tokens) / line_count if line_count > 0 else 0
    
    print(f"   📊 Analysis complete:")
    print(f"      Lines: {line_count:,}")
    print(f"      Input tokens: {total_input_tokens:,}")
    print(f"      Output tokens: {total_output_tokens:,}")
    print(f"      Total tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"      Avg tokens/line: {avg_tokens_per_line:.1f}")
    
    return line_count, total_input_tokens + total_output_tokens, avg_tokens_per_line


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


def upload_file(client: openai.OpenAI, file_path: Path) -> str:
    """Upload training file to OpenAI."""
    print(f"📤 Uploading {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    
    with open(file_path, 'rb') as f:
        file_obj = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"✅ File uploaded: {file_obj.id}")
    return file_obj.id


def create_fine_tune_job(client: openai.OpenAI, file_id: str, model: str = "gpt-4.1-nano-2025-04-14", 
                        suffix: Optional[str] = None) -> str:
    """Create a fine-tuning job."""
    print(f"🚀 Starting fine-tune job for {model}...")
    
    kwargs = {
        "training_file": file_id,
        "model": model,
    }
    
    if suffix:
        kwargs["suffix"] = suffix
    
    job = client.fine_tuning.jobs.create(**kwargs)
    
    print(f"✅ Fine-tune job created: {job.id}")
    print(f"   Model: {job.model}")
    print(f"   Status: {job.status}")
    
    return job.id


def poll_job_status(client: openai.OpenAI, job_id: str, poll_interval: int = 30) -> str:
    """Poll job status until completion."""
    print(f"⏳ Polling job {job_id} every {poll_interval}s...")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            
            if job.status != last_status:
                elapsed = time.time() - start_time
                print(f"   Status: {job.status} (elapsed: {elapsed/60:.1f}m)")
                last_status = job.status
                
                # Show training progress if available
                if hasattr(job, 'trained_tokens') and job.trained_tokens:
                    print(f"   Trained tokens: {job.trained_tokens:,}")
            
            # Terminal states
            if job.status == "succeeded":
                print(f"🎉 Fine-tuning completed successfully!")
                print(f"   Final model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            
            elif job.status == "failed":
                print(f"❌ Fine-tuning failed!")
                if hasattr(job, 'error') and job.error:
                    print(f"   Error: {job.error}")
                return None
            
            elif job.status == "cancelled":
                print(f"⚠️  Fine-tuning was cancelled")
                return None
            
            # Continue polling for running states
            elif job.status in ["validating_files", "queued", "running"]:
                time.sleep(poll_interval)
                continue
            
            else:
                print(f"⚠️  Unknown status: {job.status}")
                time.sleep(poll_interval)
                continue
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user. Job {job_id} is still running on OpenAI.")
            print(f"   Check status with: openai api fine_tunes.get -i {job_id}")
            return None
        
        except Exception as e:
            print(f"❌ Error polling job: {e}")
            time.sleep(poll_interval)
            continue


def main():
    parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning Script")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL training file")
    parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14", 
                       help="Base model to fine-tune (default: gpt-4.1-nano-2025-04-14)")
    parser.add_argument("--suffix", type=str, help="Suffix for the fine-tuned model name")
    parser.add_argument("--poll-interval", type=int, default=30, 
                       help="Polling interval in seconds (default: 30)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Validate file
    if not args.jsonl_file.exists():
        print(f"❌ File not found: {args.jsonl_file}")
        sys.exit(1)
    
    if not args.jsonl_file.suffix == '.jsonl':
        print(f"⚠️  Warning: File doesn't have .jsonl extension: {args.jsonl_file}")
    
    # Setup API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Initialize client
    client = openai.OpenAI(api_key=api_key)
    
    # Analyze tokens first
    line_count, total_tokens, avg_tokens = analyze_jsonl_tokens(args.jsonl_file, args.model)
    
    # Calculate estimated cost (rough estimate for fine-tuning)
    # OpenAI pricing is approximately $8 per 1M tokens for gpt-3.5-turbo fine-tuning
    # For gpt-4 models, it's higher (varies by model)
    estimated_cost = total_tokens / 1_000_000 * 8  # Rough estimate
    
    print(f"\n💰 Estimated fine-tuning cost: ~${estimated_cost:.2f}")
    print(f"   (Based on $8/1M tokens - actual cost may vary by model)")
    
    # Ask if user wants to use a subset
    print(f"\n🤔 Do you want to fine-tune on all {line_count:,} lines?")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Average tokens per line: {avg_tokens:.1f}")
    
    use_subset = input("\nUse a subset instead? (y/n): ").lower().strip()
    
    training_file = args.jsonl_file
    if use_subset == 'y':
        while True:
            try:
                subset_size = input(f"How many lines to use? (1-{line_count}): ").strip()
                subset_size = int(subset_size)
                
                if subset_size < 1:
                    print("   ❌ Number must be at least 1")
                    continue
                elif subset_size > line_count:
                    print(f"   ❌ Number cannot exceed {line_count}")
                    continue
                
                # Show updated estimates
                subset_tokens = int(avg_tokens * subset_size)
                subset_cost = subset_tokens / 1_000_000 * 8
                
                print(f"\n📊 Subset estimates:")
                print(f"   Lines: {subset_size:,}")
                print(f"   Estimated tokens: {subset_tokens:,}")
                print(f"   Estimated cost: ~${subset_cost:.2f}")
                
                confirm = input(f"\nProceed with {subset_size} lines? (y/n): ").lower().strip()
                if confirm == 'y':
                    training_file = create_subset_file(args.jsonl_file, subset_size)
                    break
                else:
                    continue
                    
            except ValueError:
                print("   ❌ Please enter a valid number")
                continue
    
    print("🤖 OpenAI Fine-Tuning Pipeline")
    print("=" * 50)
    print(f"Training file: {training_file}")
    print(f"Base model: {args.model}")
    if args.suffix:
        print(f"Model suffix: {args.suffix}")
    print("=" * 50)
    
    try:
        # Step 1: Upload file
        file_id = upload_file(client, training_file)
        
        # Step 2: Create fine-tune job
        job_id = create_fine_tune_job(client, file_id, args.model, args.suffix)
        
        # Step 3: Poll until completion
        final_model = poll_job_status(client, job_id, args.poll_interval)
        
        if final_model:
            print("\n" + "=" * 50)
            print(f"🎯 SUCCESS! Fine-tuned model ready: {final_model}")
            print("=" * 50)
            
            # Show usage example
            print("\n📝 Usage example:")
            print(f'client = openai.OpenAI()')
            print(f'response = client.chat.completions.create(')
            print(f'    model="{final_model}",')
            print(f'    messages=[{{"role": "user", "content": "Hello!"}}]')
            print(f')')
        else:
            print("\n❌ Fine-tuning did not complete successfully")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
