#!/usr/bin/env python3
"""Run GEPA locally on Heart Disease via backend endpoint (localhost:8000).

This script uses the backend API endpoint with proper authentication, ensuring
balance checking works correctly. It emulates the real flow but bypasses Modal.

Usage:
    python run_gepa_local.py --task-app-url http://127.0.0.1:8114 --rollout-budget 20
"""

import asyncio
import sys
import os
import json
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Load environment variables
from dotenv import load_dotenv
# Load from monorepo backend .env.dev file
monorepo_env = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "monorepo" / "backend" / ".env.dev"
if monorepo_env.exists():
    load_dotenv(dotenv_path=monorepo_env)
else:
    load_dotenv()  # Fallback to default .env lookup

# Add synth-ai source to path to use local changes (not installed package)
synth_ai_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(synth_ai_root) not in sys.path:
    sys.path.insert(0, str(synth_ai_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose HTTP logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Terminal statuses for manual streaming loop
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "canceled", "completed"}

# Import SDK
try:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    from synth_ai.learning.prompt_learning_client import PromptLearningClient
except ImportError:
    print("ERROR: synth-ai SDK not found. Install with: pip install synth-ai")
    sys.exit(1)


def create_heartdisease_gepa_toml(
    task_app_url: str,
    rollout_budget: int = 20,
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
) -> str:
    """Create TOML config for Heart Disease GEPA."""
    
    # Auto-scale GEPA parameters
    initial_population_size = max(2, min(5, rollout_budget // 10))
    num_generations = max(2, min(5, rollout_budget // (initial_population_size * 2)))
    
    if train_seeds is None:
        # Trial budget: 30 seeds for training/evaluation
        train_seeds = list(range(0, 30))
    
    if val_seeds is None:
        # Heldout pool: 50 seeds, non-overlapping with train seeds
        max_train = max(train_seeds) if train_seeds else -1
        val_seeds = list(range(max_train + 1, max_train + 1 + 50))
    
    # Get API keys
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
    
    policy_model = os.getenv("POLICY_MODEL", "llama-3.1-8b-instant")
    mutation_model = os.getenv("MUTATION_MODEL", "llama-3.3-70b-versatile")
    
    # Build TOML config
    toml_content = f"""[prompt_learning]
algorithm = "gepa"
task_app_url = "{task_app_url}"
task_app_api_key = "{task_app_api_key}"
# Backwards compatibility: also include train_seeds at top level
train_seeds = {train_seeds}

[prompt_learning.initial_prompt]
id = "heartdisease_pattern"
name = "Heart Disease Classification Pattern"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a medical classification assistant. Based on the patient's features, classify whether they have heart disease. Respond with '1' for heart disease or '0' for no heart disease."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Patient Features:\\n{{features}}\\n\\nClassify: Does this patient have heart disease? Respond with '1' for yes or '0' for no."
order = 1

[prompt_learning.initial_prompt.wildcards]
features = "REQUIRED"

[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "{policy_model}"
provider = "groq"
temperature = 0.0
max_completion_tokens = 512

[prompt_learning.gepa]
env_name = "heartdisease"

[prompt_learning.gepa.evaluation]
train_seeds = {train_seeds}
val_seeds = {val_seeds}
validation_pool = "train"
validation_top_k = 2

[prompt_learning.gepa.rollout]
budget = {rollout_budget}
max_concurrent = 5

[prompt_learning.gepa.mutation]
rate = 0.3
llm_model = "{mutation_model}"
llm_provider = "groq"
llm_inference_url = "https://api.groq.com"
temperature = 0.7
max_tokens = 512

[prompt_learning.gepa.population]
initial_size = {initial_population_size}
num_generations = {num_generations}
children_per_generation = {max(2, min(5, rollout_budget // (num_generations * 2)))}

[prompt_learning.gepa.archive]
max_size = 10
min_score_threshold = 0.0
feedback_fraction = 0.33

[prompt_learning.gepa.token]
max_limit = 4096
counting_model = "gpt-4"
enforce_limit = false

[prompt_learning.termination_config]
max_cost_usd = {max(0.10, rollout_budget * 0.001 * 10)}
max_trials = {rollout_budget * 2}
max_category_costs_usd = {{"rollout" = {max(0.10, rollout_budget * 0.001 * 10) * 0.8}, "mutation" = {max(0.10, rollout_budget * 0.001 * 10) * 0.2}}}
"""
    
    return toml_content


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GEPA locally on Heart Disease via backend endpoint")
    parser.add_argument(
        "--task-app-url",
        type=str,
        default="http://127.0.0.1:8114",
        help="Task app URL (default: http://127.0.0.1:8114)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=20,
        help="Rollout budget (default: 20)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Enable live TUI dashboard (requires rich and plotille)",
    )
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        help="Training seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--val-seeds",
        type=int,
        nargs="+",
        help="Validation seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to SYNTH_API_KEY env var)",
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("API key required (provide --api-key or set SYNTH_API_KEY env var)")
    
    print("=" * 80)
    print("GEPA Local Test: Heart Disease (via Backend Endpoint)")
    print("=" * 80)
    print(f"Backend URL: {args.backend_url}")
    print(f"Task app URL: {args.task_app_url}")
    print(f"Rollout budget: {args.rollout_budget}")
    print("=" * 80)
    print()
    
    # Create temporary TOML config
    toml_content = create_heartdisease_gepa_toml(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        config_path = Path(f.name)
    
    try:
        # Create job using SDK
        task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
        if not task_app_api_key:
            raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
        
        job = PromptLearningJob.from_config(
            config_path=str(config_path),
            backend_url=args.backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,  # Explicitly pass task app API key for health check
            overrides={"overrides": {"run_local": True}},  # Run locally in-process instead of Modal
        )
        
        # Validate config before submission
        print("Validating GEPA config...")
        try:
            from synth_ai.api.train.validators import validate_prompt_learning_config_from_file
            validate_prompt_learning_config_from_file(config_path, algorithm="gepa")
            print("✓ Config validated successfully")
        except ImportError:
            # Fallback to basic validation if validator not available
            print("⚠️  Config validator not found, skipping validation")
        except Exception as e:
            print(f"\n{'=' * 80}")
            print("❌ Config Validation Failed")
            print(f"{'=' * 80}")
            print(str(e))
            print(f"{'=' * 80}\n")
            raise
        print()
        
        print("Submitting job to backend...")
        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}")
        print()
        
        # Instantiate client for streaming and post-run queries
        client = PromptLearningClient(args.backend_url, api_key)

        # Stream events/status in real-time using manual polling
        print("Streaming job events...")
        print("=" * 80)
        
        # Optionally start TUI dashboard in background
        tui_process = None
        if args.tui:
            try:
                import subprocess
                tui_script = Path(__file__).parent / "gepa_tui.py"
                if tui_script.exists():
                    print("📊 Starting live TUI dashboard...")
                    print("   (Press Ctrl+C to stop)")
                    print()
                    # Start TUI as a subprocess that reads from stdin
                    # We'll pipe our output to it
                    tui_process = subprocess.Popen(
                        [sys.executable, str(tui_script)],
                        stdin=subprocess.PIPE,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        text=True,
                        bufsize=1,
                    )
                else:
                    print("⚠️  TUI script not found, continuing without dashboard")
            except Exception as e:
                print(f"⚠️  Could not start TUI dashboard: {e}")
                print("   Continuing without dashboard...")

        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        else:
            os.environ['PYTHONUNBUFFERED'] = '1'

        # Capture json module reference for nested function closure
        _json = json
        
        # Capture tui_process for nested function
        _tui_process = tui_process

        async def stream_job(job_client: PromptLearningClient, job_id: str, poll_interval: float = 1.0) -> tuple[dict[str, Any], int, list[tuple[int, float]]]:
            """Manual status/event polling loop to ensure visibility.
            
            Returns:
                Tuple of (job_detail, total_events, optimization_curve)
            """
            next_seq = 0
            total_events = 0
            last_status: str | None = None
            terminal_rounds = 0
            
            # Track optimization curve: (trial_count, best_score)
            optimization_curve: list[tuple[int, float]] = []
            trial_counter = 0
            best_score_so_far = 0.0

            while True:
                job_detail = await job_client.get_job(job_id)
                status = str(job_detail.get("status") or job_detail.get("state") or "").lower()
                timestamp = datetime.now().strftime("%H:%M:%S")

                if status and status != last_status:
                    status_line = f"[{timestamp}] status={status}"
                    print(status_line, flush=True)
                    last_status = status
                    
                    # Also send to TUI if running
                    if _tui_process and _tui_process.stdin:
                        try:
                            _tui_process.stdin.write(status_line + "\n")
                            _tui_process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass

                events = await job_client.get_events(job_id, since_seq=next_seq, limit=500)
                if events:
                    for event in events:
                        seq = event.get("seq")
                        if isinstance(seq, int):
                            next_seq = max(next_seq, seq + 1)
                        event_type = event.get("type", "event")
                        level = event.get("level")
                        msg = event.get("message") or ""
                        
                        # Track trial results for optimization curve
                        if event_type == "prompt.learning.trial.results":
                            data = event.get("data", {})
                            mean_score = data.get("mean")
                            if mean_score is not None:
                                trial_counter += 1
                                best_score_so_far = max(best_score_so_far, float(mean_score))
                                optimization_curve.append((trial_counter, best_score_so_far))
                        
                        # Skip verbose candidate dumps entirely - don't print message or JSON
                        # Note: trial.results is NOT filtered - we want to see trial scores
                        verbose_event_types = [
                            "prompt.learning.proposal.scored",
                            "prompt.learning.eval.summary",  # Old name, kept for compatibility
                            "prompt.learning.validation.scored",
                            "prompt.learning.final.results",
                        ]
                        skip_event = event_type in verbose_event_types
                        
                        # Skip entire event (message + JSON) for verbose events
                        if skip_event:
                            continue
                        
                        prefix = f"[{datetime.now().strftime('%H:%M:%S')}] {event_type}"
                        if level:
                            prefix += f" ({level})"
                        output_line = f"{prefix}: {msg}"
                        print(output_line, flush=True)
                        
                        # Also send to TUI if running
                        if _tui_process and _tui_process.stdin:
                            try:
                                _tui_process.stdin.write(output_line + "\n")
                                _tui_process.stdin.flush()
                            except (BrokenPipeError, OSError):
                                # TUI process may have exited
                                pass
                        
                        data = event.get("data")
                        if isinstance(data, dict) and data:
                            # Format USD values to 4 decimals and rename usd_tokens to tokens_usd for consistency
                            formatted_data = {}
                            for key, value in data.items():
                                # Rename usd_tokens to tokens_usd for consistency with sandbox_usd, total_usd
                                if key == "usd_tokens":
                                    formatted_data["tokens_usd"] = round(float(value), 4) if isinstance(value, (int, float)) else value
                                elif isinstance(value, (int, float)) and ('usd' in key.lower() or 'cost' in key.lower()):
                                    formatted_data[key] = round(float(value), 4)
                                else:
                                    formatted_data[key] = value
                            json_line = _json.dumps(formatted_data, separators=(',', ':'))
                            print(json_line, flush=True)
                            
                            # Also send JSON to TUI if running
                            if _tui_process and _tui_process.stdin:
                                try:
                                    _tui_process.stdin.write(json_line + "\n")
                                    _tui_process.stdin.flush()
                                except (BrokenPipeError, OSError):
                                    pass
                    total_events += len(events)

                # Exit condition: if we're in a terminal status, check if we should exit
                if status in TERMINAL_STATUSES:
                    terminal_rounds += 1
                    # Exit if:
                    # 1. We've seen terminal status for 2+ rounds AND no new events this round, OR
                    # 2. We've seen terminal status for 3+ rounds (give it time to receive final events)
                    if (terminal_rounds >= 2 and not events) or terminal_rounds >= 3:
                        return job_detail, total_events, optimization_curve
                else:
                    # Reset counter if we're not in terminal status
                    terminal_rounds = 0

                await asyncio.sleep(poll_interval)

        final_status, streamed_events, optimization_curve = await stream_job(client, job_id, poll_interval=0.5)

        if streamed_events == 0:
            print("\n⚠️  No real-time events received during streaming. Fetching events via API for diagnostics...\n")
            api_events = await client.get_events(job_id, limit=200)
            print(f"Fetched {len(api_events)} events via API. Sample types: {[event.get('type') for event in api_events[:5]]}")

        print()
        print("=" * 80)
        print("✅ GEPA Optimization Complete!")
        print("=" * 80)
        print(f"Job ID: {job_id}")
        print(f"Status: {final_status.get('status', 'unknown')}")
        
        # We already fetched final job detail in stream_job
        job_detail = final_status
        
        # Extract best_score from job detail (check multiple locations)
        best_score = None
        best_train_score = None
        best_validation_score = None
        
        if isinstance(job_detail, dict):
            # Try top-level fields first (from PromptLearningOnlineJobSummary)
            best_score = job_detail.get('best_score')
            best_train_score = job_detail.get('best_train_score')
            best_validation_score = job_detail.get('best_validation_score')
            
            # Fall back to nested metadata if not found
            if best_score is None or best_train_score is None:
                metadata = job_detail.get('metadata', {})
                if isinstance(metadata, dict):
                    job_metadata = metadata.get('job_metadata', {})
                    if isinstance(job_metadata, dict):
                        if best_train_score is None:
                            best_train_score = job_metadata.get('prompt_best_train_score')
                        if best_validation_score is None:
                            best_validation_score = job_metadata.get('prompt_best_validation_score')
                        if best_score is None:
                            best_score = best_train_score or job_metadata.get('prompt_best_score')
        
        # Fetch prompt results from event stream (more reliable for best prompt/score)
        prompts_data = await client.get_prompts(job_id)

        event_best_score = prompts_data.best_score
        if event_best_score is not None:
            best_score = event_best_score

        # Display scores
        print(f"Best Score: {best_score if best_score is not None else 'N/A'}")
        if best_train_score is not None and best_train_score != best_score:
            print(f"Best Train Score: {best_train_score}")
        if best_validation_score is not None:
            print(f"Best Validation Score: {best_validation_score}")
        if prompts_data.validation_results:
            val_summary = prompts_data.validation_results[0]
            val_acc = val_summary.get("accuracy")
            if val_acc is not None:
                print(f"Validation Accuracy (Top-K[0]): {val_acc}")
        
        # Extract and display best prompt(s) from best_snapshot
        best_snapshot = job_detail.get('best_snapshot') if isinstance(job_detail, dict) else None
        best_snapshot_id = job_detail.get('best_snapshot_id') if isinstance(job_detail, dict) else None
        
        print()
        best_prompt_from_events = prompts_data.best_prompt if isinstance(prompts_data.best_prompt, dict) else None
        
        # If best_prompt is null from events, try to extract from attempted_candidates
        if not best_prompt_from_events and prompts_data.attempted_candidates:
            top_attempt = max(
                (c for c in prompts_data.attempted_candidates if isinstance(c, dict)),
                key=lambda c: c.get("accuracy", 0.0),
                default=None,
            )
            if top_attempt and top_attempt.get("accuracy", 0.0) > 0:
                # Try to reconstruct prompt from candidate
                obj = top_attempt.get("object") or {}
                if isinstance(obj, dict):
                    repl = obj.get("text_replacements")
                    if isinstance(repl, list) and repl:
                        # Create a simple prompt structure from the best candidate
                        new_text = repl[0].get("new_text", "")
                        if new_text:
                            best_prompt_from_events = {
                                "messages": [{"role": "system", "content": new_text}]
                            }
        
        if best_prompt_from_events:
            print("Best Prompt:")
            print("-" * 80)
            sections = best_prompt_from_events.get("sections") or []
            messages = best_prompt_from_events.get("messages") or []
            if sections:
                for section in sections:
                    if isinstance(section, dict):
                        role = section.get("role", "unknown")
                        content = section.get("content", "")
                        if content:
                            print(f"[{role}]")
                            print(content)
                            print()
            elif messages:
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if content:
                            print(f"[{role}]")
                            print(content)
                            print()
            else:
                print(json.dumps(best_prompt_from_events, indent=2))
        elif best_snapshot and isinstance(best_snapshot, dict):
            print("Best Prompt:")
            print("-" * 80)
            
            # Snapshot payload structure: best_prompt is serialized PromptTemplate
            # Try best_prompt first (this is what GEPA stores)
            best_prompt_data = best_snapshot.get('best_prompt')
            
            if best_prompt_data and isinstance(best_prompt_data, dict):
                # PromptTemplate serialization has 'sections' or 'messages'
                sections = best_prompt_data.get('sections', [])
                messages = best_prompt_data.get('messages', [])
                
                if sections:
                    # Sections format: list of dicts with 'role' and 'content'
                    for section in sections:
                        role = section.get('role', 'unknown')
                        content = section.get('content', '')
                        if content:
                            print(f"[{role}]:\n{content}\n")
                elif messages:
                    # Messages format: list of message dicts
                    for msg in messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if content:
                            print(f"[{role}]:\n{content}\n")
                else:
                    # Fallback: show the structure
                    print(json.dumps(best_prompt_data, indent=2))
            else:
                # Try other extraction methods as fallback
                prompt_template = best_snapshot.get('prompt_template') or best_snapshot.get('template')
                
                if not prompt_template:
                    messages = best_snapshot.get('messages')
                    if messages:
                        prompt_template = {'messages': messages}
                
                if prompt_template:
                    if isinstance(prompt_template, dict):
                        messages = prompt_template.get('messages', [])
                        if messages:
                            for msg in messages:
                                role = msg.get('role', 'unknown')
                                content = msg.get('content', '')
                                if content:
                                    print(f"[{role}]:\n{content}\n")
                        else:
                            print(json.dumps(prompt_template, indent=2)[:800])
                    elif isinstance(prompt_template, str):
                        print(prompt_template)
                    else:
                        print(str(prompt_template)[:800])
                else:
                    # Debug: show what we have
                    print(f"Snapshot ID: {best_snapshot_id or 'N/A'}")
                    print(f"Snapshot keys: {list(best_snapshot.keys())[:20]}")
                    # Show best_prompt if it exists but wasn't a dict
                    if 'best_prompt' in best_snapshot:
                        bp = best_snapshot['best_prompt']
                        print(f"best_prompt type: {type(bp)}")
                        if isinstance(bp, str):
                            print(f"best_prompt (str): {bp[:400]}...")
                        else:
                            print(f"best_prompt: {json.dumps(bp, indent=2)[:800]}")
        elif best_snapshot_id:
            print(f"Best Prompt: Snapshot ID {best_snapshot_id} exists but payload not available (try fetching snapshot directly)")
        else:
            print("Best Prompt: Not available")
        
        # Show failure reason if failed (simplified)
        if final_status.get('status') == 'failed':
            print("\n⚠️  Job failed.")
            try:
                error_message = job_detail.get('error_message') or job_detail.get('error') or 'Unknown error'
                print(f"Error: {error_message}")
            except Exception:
                pass
        
        print()
        
        # Extract cost and balance from events or best_snapshot
        best_snapshot = job_detail.get("best_snapshot")
        total_cost = None
        final_balance = None
        balance_type = None
        
        # Try to get from billing.end events
        all_events = await client.get_events(job_id, limit=1000)
        billing_end_events = [e for e in all_events if e.get('type') == 'prompt.learning.billing.end']
        if billing_end_events:
            last_billing = billing_end_events[-1].get('data', {})
            total_cost = last_billing.get('total_usd')
            final_balance = last_billing.get('final_balance_usd')
            balance_type = last_billing.get('balance_type')
        
        # Fallback to best_snapshot if events don't have it
        if total_cost is None and best_snapshot and isinstance(best_snapshot, dict):
            total_cost = best_snapshot.get('total_usd')
            final_balance = best_snapshot.get('final_balance_usd')
            balance_type = best_snapshot.get('balance_type')
        
        # Extract metrics for consolidated summary table
        policy_cost_usd = None
        proposal_cost_usd = None
        total_cost_usd = total_cost
        n_rollouts = None
        rollout_tokens_millions = None
        time_seconds = None
        best_boost = None
        kth_boost = None
        
        # Extract from billing.end event
        if all_events:
            billing_end_events = [e for e in all_events if e.get('type') == 'prompt.learning.billing.end']
            if billing_end_events:
                billing_data = billing_end_events[-1].get('data', {})
                time_seconds = billing_data.get('seconds')
                total_cost_usd = billing_data.get('total_usd') or total_cost_usd
                tokens_usd = billing_data.get('tokens_usd', 0.0)
                sandbox_usd = billing_data.get('sandbox_usd', 0.0)
            
            # Extract from completed event for token breakdown
            completed_events = [e for e in all_events if e.get('type') == 'prompt.learning.completed']
            if completed_events:
                completed_data = completed_events[-1].get('data', {})
                # Policy = rollouts (evaluating prompts), Proposal = mutation (generating new prompts)
                policy_cost_usd = completed_data.get('usd_tokens_rollouts', 0.0) or 0.0
                proposal_cost_usd = completed_data.get('usd_tokens_mutation', 0.0) or 0.0
                
                # Fallback: try to calculate from token counts if USD costs are 0
                if policy_cost_usd == 0.0:
                    # If costs are 0, they might not be calculated - check if we have tokens
                    rollouts_prompt = completed_data.get('rollouts_prompt_tokens', 0) or 0
                    rollouts_completion = completed_data.get('rollouts_completion_tokens', 0) or 0
                    rollouts_unknown = completed_data.get('rollouts_unknown_tokens', 0) or 0
                    if (rollouts_prompt + rollouts_completion + rollouts_unknown) > 0:
                        # Costs are likely 0 because pricing estimation failed - show as N/A
                        policy_cost_usd = None
                
                if proposal_cost_usd == 0.0:
                    mutation_prompt = completed_data.get('mutation_prompt_tokens', 0) or 0
                    mutation_completion = completed_data.get('mutation_completion_tokens', 0) or 0
                    mutation_unknown = completed_data.get('mutation_unknown_tokens', 0) or 0
                    if (mutation_prompt + mutation_completion + mutation_unknown) > 0:
                        # Costs are likely 0 because pricing estimation failed - show as N/A
                        proposal_cost_usd = None
                
                # Rollout tokens in millions
                rollouts_prompt = completed_data.get('rollouts_prompt_tokens', 0) or 0
                rollouts_completion = completed_data.get('rollouts_completion_tokens', 0) or 0
                rollouts_unknown = completed_data.get('rollouts_unknown_tokens', 0) or 0
                rollout_tokens_total = rollouts_prompt + rollouts_completion + rollouts_unknown
                rollout_tokens_millions = rollout_tokens_total / 1_000_000.0
            
            # Extract rollout count from progress events (use max across all progress events)
            progress_events = [e for e in all_events if e.get('type') == 'prompt.learning.progress']
            trial_rollouts = 0
            if progress_events:
                # Get max rollouts_completed across all progress events (in case last one is 0)
                all_rollout_counts = [
                    e.get('data', {}).get('rollouts_completed', 0) or 0
                    for e in progress_events
                    if e.get('data', {}).get('rollouts_completed') is not None
                ]
                if all_rollout_counts:
                    trial_rollouts = max(all_rollout_counts)
                else:
                    last_progress = progress_events[-1].get('data', {})
                    trial_rollouts = last_progress.get('rollouts_completed', 0) or 0
            
            # Fallback: estimate from tokens if rollout count is still 0 but tokens exist
            if trial_rollouts == 0 and rollout_tokens_millions is not None and rollout_tokens_millions > 0:
                # Rough estimate: assume ~500 tokens per rollout on average
                estimated_rollouts = int((rollout_tokens_millions * 1_000_000) / 500)
                if estimated_rollouts > 0:
                    trial_rollouts = estimated_rollouts
            
            # Add heldout evaluation rollouts (baseline + top-K)
            heldout_rollouts = 0
            validation_summary_events = [e for e in all_events if e.get('type') == 'prompt.learning.validation.summary']
            if validation_summary_events:
                val_summary = validation_summary_events[-1].get('data', {})
                baseline = val_summary.get('baseline', {})
                results = val_summary.get('results', [])
                
                # Baseline rollouts
                baseline_seeds = baseline.get('seeds', [])
                if baseline_seeds:
                    heldout_rollouts += len(baseline_seeds)
                
                # Top-K rollouts (each result evaluated on heldout set)
                for result in results:
                    result_seeds = result.get('seeds', [])
                    if result_seeds:
                        heldout_rollouts += len(result_seeds)
            
            # Total rollouts = trial rollouts + heldout evaluation rollouts
            n_rollouts = trial_rollouts + heldout_rollouts
        
        # Extract boost metrics from validation summary
        baseline_acc_for_boost = None
        results_for_boost = []
        if all_events:
            validation_summary_events = [e for e in all_events if e.get('type') == 'prompt.learning.validation.summary']
            if validation_summary_events:
                val_summary = validation_summary_events[-1].get('data', {})
                baseline = val_summary.get('baseline', {})
                results = val_summary.get('results', [])
                
                baseline_acc = baseline.get('accuracy')
                baseline_acc_for_boost = baseline_acc
                results_for_boost = results
                baseline_seeds = baseline.get('seeds', [])
                heldout_n = len(baseline_seeds) if baseline_seeds else None
                
                print()
                print("=" * 80)
                print("HELDOUT SET EVALUATION")
                print("=" * 80)
                
                # Baseline evaluation - clearly marked and separate
                if baseline_acc is not None:
                    print()
                    print("📊 BASELINE PROMPT (Heldout Set):")
                    print(f"  Accuracy: {baseline_acc:.4f}")
                    print(f"  N: {heldout_n}")
                    baseline_instance_scores = baseline.get('instance_scores', [])
                    if baseline_instance_scores:
                        correct = sum(1 for s in baseline_instance_scores if s == 1.0)
                        total = len(baseline_instance_scores)
                        print(f"  Correct: {correct}/{total}")
                
                # Top K candidates - clearly marked and separate
                if results:
                    print()
                    print("📊 TOP-K PROPOSED PROMPTS (Heldout Set):")
                    if len(results) < 2:
                        print()
                        print(f"⚠️  WARNING: Only {len(results)} candidate(s) evaluated, but validation_top_k=2 was configured!")
                        print(f"⚠️  Possible reasons:")
                        print(f"⚠️    1. Archive had fewer than 2 candidates")
                        print(f"⚠️    2. Only {len(results)} candidate(s) passed filters")
                        print(f"⚠️    3. Early termination occurred")
                        print()
                    for i, result in enumerate(results[:2]):  # Show top K=2
                        result_acc = result.get('accuracy')
                        lift_pct = result.get('lift_pct_vs_baseline', 0.0)
                        rank = result.get('rank', i)
                        
                        if result_acc is not None:
                            print()
                            print(f"  Rank {rank}:")
                            print(f"    Accuracy: {result_acc:.4f}")
                            print(f"    N: {heldout_n}")
                            if baseline_acc is not None:
                                delta = result_acc - baseline_acc
                                sign = "+" if delta >= 0 else ""
                                print(f"    Delta vs Baseline: {sign}{delta:.4f} ({sign}{lift_pct:.2f}%)")
                            result_instance_scores = result.get('instance_scores', [])
                            if result_instance_scores:
                                correct = sum(1 for s in result_instance_scores if s == 1.0)
                                total = len(result_instance_scores)
                                print(f"    Correct: {correct}/{total}")
                
                # Extract boost metrics
                if baseline_acc_for_boost is not None and results_for_boost:
                    if len(results_for_boost) > 0:
                        rank0_acc = results_for_boost[0].get('accuracy')
                        if rank0_acc is not None:
                            best_boost = rank0_acc - baseline_acc_for_boost
                    if len(results_for_boost) > 1:
                        rank1_acc = results_for_boost[1].get('accuracy')
                        if rank1_acc is not None:
                            kth_boost = rank1_acc - baseline_acc_for_boost
                
                print()
                print("=" * 80)
        
        # Display consolidated final summary table
        print()
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print()
        
        # Build table rows
        rows = []
        
        # Cost row
        # Policy = rollouts (evaluating prompts), Proposal = mutation (generating new prompts)
        if policy_cost_usd is None:
            cost_policy = "N/A (tokens used but cost not calculated)"
        elif policy_cost_usd == 0.0:
            cost_policy = "$0.0000"
        else:
            cost_policy = f"${policy_cost_usd:.4f}"
        
        if proposal_cost_usd is None:
            cost_proposal = "N/A (tokens used but cost not calculated)"
        elif proposal_cost_usd == 0.0:
            cost_proposal = "$0.0000"
        else:
            cost_proposal = f"${proposal_cost_usd:.4f}"
        
        cost_total = f"${total_cost_usd:.4f}" if total_cost_usd is not None else "N/A"
        rows.append(("Cost", f"Policy (rollouts): {cost_policy} | Proposal (mutation): {cost_proposal} | Total: {cost_total}"))
        
        # Rollouts row
        rollouts_str = f"{n_rollouts}" if n_rollouts is not None else "N/A"
        tokens_str = f"{rollout_tokens_millions:.4f}M" if rollout_tokens_millions is not None else "N/A"
        rows.append(("Rollouts", f"N: {rollouts_str} | Tokens: {tokens_str}"))
        
        # Rollout Duration Statistics row
        duration_stats = completed_data.get('rollout_duration_stats')
        if duration_stats:
            min_dur = duration_stats.get('min', 0.0)
            median_dur = duration_stats.get('median', 0.0)
            p90_dur = duration_stats.get('p90', 0.0)
            p99_dur = duration_stats.get('p99', 0.0)
            max_dur = duration_stats.get('max', 0.0)
            duration_str = f"min={min_dur:.3f}s, median={median_dur:.3f}s, p90={p90_dur:.3f}s, p99={p99_dur:.3f}s, max={max_dur:.3f}s"
            rows.append(("Rollout Duration", duration_str))
        else:
            rows.append(("Rollout Duration", "N/A"))
        
        # Throughput metrics row
        rollouts_per_min = completed_data.get('rollouts_per_minute')
        requests_per_min = completed_data.get('requests_per_minute')
        tokens_per_min = completed_data.get('tokens_per_minute')
        
        throughput_parts = []
        if rollouts_per_min is not None:
            throughput_parts.append(f"rollouts={rollouts_per_min:.1f}/min")
        if requests_per_min is not None:
            throughput_parts.append(f"requests={requests_per_min:.1f}/min")
        if tokens_per_min is not None:
            tokens_per_min_millions = tokens_per_min / 1_000_000.0
            throughput_parts.append(f"tokens={tokens_per_min_millions:.4f}M/min")
        
        if throughput_parts:
            rows.append(("Throughput", " | ".join(throughput_parts)))
        else:
            rows.append(("Throughput", "N/A"))
        
        # Time row
        time_str = f"{time_seconds:.1f}s" if time_seconds is not None else "N/A"
        rows.append(("Time", time_str))
        
        # Boost row
        best_boost_str = f"+{best_boost:.4f}" if best_boost is not None and best_boost >= 0 else f"{best_boost:.4f}" if best_boost is not None else "N/A"
        kth_boost_str = f"+{kth_boost:.4f}" if kth_boost is not None and kth_boost >= 0 else f"{kth_boost:.4f}" if kth_boost is not None else "N/A"
        rows.append(("Boost", f"Best: {best_boost_str} | Kth: {kth_boost_str}"))
        
        # Print table
        max_label_len = max(len(row[0]) for row in rows)
        for label, value in rows:
            print(f"  {label:>{max_label_len}}: {value}")
        
        # Display optimization curve
        if optimization_curve:
            print()
            print("=" * 80)
            print("OPTIMIZATION CURVE")
            print("=" * 80)
            print()
            try:
                # Import from same directory (use different name to avoid shadowing module-level sys)
                from pathlib import Path as PathLibPath
                plot_module_path = PathLibPath(__file__).parent / "plot_optimization_curve.py"
                if plot_module_path.exists():
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("plot_optimization_curve", plot_module_path)
                    plot_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plot_module)
                    plot_optimization_curve = plot_module.plot_optimization_curve
                else:
                    raise ImportError(f"plot_optimization_curve.py not found at {plot_module_path}")
                
                trial_counts = [t for t, _ in optimization_curve]
                best_scores = [s for _, s in optimization_curve]
                curve_plot = plot_optimization_curve(
                    trial_counts=trial_counts,
                    best_scores=best_scores,
                    title="Optimization Curve: Best Score vs Trial Count",
                )
                print(curve_plot)
            except ImportError:
                # Fallback: simple text representation
                print("Trial Count → Best Score:")
                for trial_count, best_score in optimization_curve:
                    print(f"  Trial {trial_count:3d}: {best_score:.4f}")
            except Exception as e:
                print(f"⚠️  Could not generate optimization curve: {e}")
        else:
            print()
            print("=" * 80)
            print("OPTIMIZATION CURVE")
            print("=" * 80)
            print("(No trial results data available)")
        
        print()
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        # Clean up TUI process if running
        if 'tui_process' in locals() and tui_process:
            try:
                tui_process.stdin.close()
                tui_process.terminate()
                tui_process.wait(timeout=2)
            except Exception:
                try:
                    tui_process.kill()
                except Exception:
                    pass
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Error during optimization")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        # Clean up TUI process if running
        if 'tui_process' in locals() and tui_process:
            try:
                tui_process.stdin.close()
                tui_process.terminate()
                tui_process.wait(timeout=2)
            except Exception:
                try:
                    tui_process.kill()
                except Exception:
                    pass
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temp config file
        try:
            config_path.unlink()
        except Exception:
            pass
        
        # Clean up TUI process if running
        if 'tui_process' in locals() and tui_process:
            try:
                tui_process.stdin.close()
                tui_process.terminate()
                tui_process.wait(timeout=2)
            except Exception:
                try:
                    tui_process.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())

