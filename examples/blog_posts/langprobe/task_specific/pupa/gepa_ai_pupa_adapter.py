"""GEPA-AI adapter for PUPA privacy-aware delegation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Handle imports for both module and direct script execution
import sys
from pathlib import Path as PathLib

_script_dir = PathLib(__file__).resolve().parent
_langprobe_dir = _script_dir.parent.parent
if str(_langprobe_dir) not in sys.path:
    sys.path.insert(0, str(_langprobe_dir))

# Try to import LearningCurveTracker, but make it optional
LearningCurveTracker = None
try:
    from integrations.learning_curve_tracker import LearningCurveTracker
except (ImportError, TypeError, AttributeError, SyntaxError):
    try:
        from ...integrations.learning_curve_tracker import LearningCurveTracker
    except (ImportError, TypeError, AttributeError, SyntaxError):
        # LearningCurveTracker is optional - continue without it
        pass

# Import from dspy_pupa_adapter - try multiple strategies
try:
    # Strategy 1: Try absolute import from langprobe
    from task_specific.pupa.dspy_pupa_adapter import (
        compute_overlap,
        create_dspy_examples,
        load_pupa_dataset,
        pupa_metric,
        tokenize,
    )
except ImportError:
    try:
        # Strategy 2: Try relative import
        from .dspy_pupa_adapter import (
            compute_overlap,
            create_dspy_examples,
            load_pupa_dataset,
            pupa_metric,
            tokenize,
        )
    except ImportError:
        # Strategy 3: Import from same directory using importlib
        import importlib.util
        dspy_adapter_path = _script_dir / "dspy_pupa_adapter.py"
        if dspy_adapter_path.exists():
            spec = importlib.util.spec_from_file_location(
                "dspy_pupa_adapter",
                dspy_adapter_path
            )
            if spec and spec.loader:
                dspy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dspy_module)
                compute_overlap = dspy_module.compute_overlap
                create_dspy_examples = dspy_module.create_dspy_examples
                load_pupa_dataset = dspy_module.load_pupa_dataset
                pupa_metric = dspy_module.pupa_metric
                tokenize = dspy_module.tokenize
            else:
                raise ImportError(f"Could not load dspy_pupa_adapter from {dspy_adapter_path}")
        else:
            raise ImportError(f"dspy_pupa_adapter.py not found at {dspy_adapter_path}")

load_dotenv()


async def run_gepa_ai_pupa(
    task_app_url: str = "http://127.0.0.1:8113",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Run GEPA-AI library optimization on PUPA."""
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "gepa_ai"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "gepa_ai.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("gepa_ai_pupa")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    print(f"📝 Verbose logs redirected to: {log_file}")
    
    try:
        from gepa.api import optimize as gepa_optimize
    except ImportError:
        raise ImportError("GEPA-AI package ('gepa') is not installed. Install with: uv add gepa")
    
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
    import requests
    
    class PUPAGEPAAdapter(GEPAAdapter[dict, dict, dict]):
        """GEPAAdapter for PUPA privacy-aware delegation."""
        
        def __init__(self, model: str, api_key: str):
            self.model = model
            self.api_key = api_key
            # Determine base URL: if model starts with "groq/", use Groq API; otherwise use OpenAI API
            if model.lower().startswith("groq/"):
                self.base_url = "https://api.groq.com/openai/v1"
            elif "openai" in model.lower():
                self.base_url = "https://api.openai.com/v1"
            else:
                # Default to Groq for unknown formats
                self.base_url = "https://api.groq.com/openai/v1"
        
        def evaluate(
            self,
            batch: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch[dict, dict]:
            """Evaluate candidate on batch of PUPA examples."""
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list[dict] | None = [] if capture_traces else None
            
            instruction = next(iter(candidate.values())) if candidate else ""
            
            messages_list = []
            for data in batch:
                category = data.get("category", data.get("predicted_category", ""))
                redacted_query = data.get("redacted_query", "")
                
                system_msg = f"{instruction}\n\n**Task:**\nRespond to the delegation task while preserving privacy. Never reconstruct redacted details; use the anonymised placeholders as-is."
                user_msg = f"Category: {category}\n\nDelegation Task:\n{redacted_query}"
                
                messages_list.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
            
            try:
                responses = []
                for i, messages in enumerate(messages_list):
                    # Format model name for API: groq/openai/gpt-oss-120b -> openai/gpt-oss-120b
                    # Groq API expects "openai/gpt-oss-120b" format, not just "gpt-oss-120b"
                    model_name = self.model
                    if model_name.startswith("groq/"):
                        model_name = model_name[5:]  # Remove "groq/" prefix
                    # Keep "openai/" prefix if present (Groq supports OpenAI-compatible models)
                    
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": model_name,
                            "messages": messages,
                            "temperature": 0.0,
                            "max_tokens": 512,
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        resp_json = resp.json()
                        content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        responses.append(content)
                    else:
                        # Log error for debugging
                        error_msg = f"API call failed: status={resp.status_code}, model={model_name}, response={resp.text[:200]}"
                        print(f"⚠️  Warning: {error_msg}", flush=True)
                        responses.append("")
                
            except Exception as e:
                for data in batch:
                    outputs.append({"response": "", "error": str(e)})
                    scores.append(0.0)
                    if capture_traces:
                        trajectories.append({"data": data, "response": "", "error": str(e)})
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
            
            # Ensure we have responses for all items in batch
            if len(responses) != len(batch):
                print(f"⚠️  Warning: Mismatch between responses ({len(responses)}) and batch size ({len(batch)})", flush=True)
                # Pad with empty strings if needed
                while len(responses) < len(batch):
                    responses.append("")
            
            for data, response_text in zip(batch, responses):
                target_response = data.get("target_response", data.get("response", ""))
                
                # Debug: log first few responses
                if len(outputs) < 2:
                    print(f"🔍 Debug: response_text length={len(response_text)}, target_response length={len(target_response)}", flush=True)
                    if not response_text:
                        print(f"⚠️  Warning: Empty response for data keys: {list(data.keys())}", flush=True)
                
                placeholder_used = int("PRESIDIO_ANONYMIZED" in response_text or "[REDACTED]" in response_text)
                content_overlap = compute_overlap(target_response, response_text)
                score = 0.6 * placeholder_used + 0.4 * content_overlap
                
                output = {
                    "response": response_text,
                    "target_response": target_response,
                    "placeholder_used": bool(placeholder_used),
                    "content_overlap": content_overlap,
                }
                outputs.append(output)
                scores.append(score)
                
                if capture_traces:
                    trajectories.append({
                        "data": data,
                        "response": response_text,
                        "target_response": target_response,
                        "placeholder_used": bool(placeholder_used),
                        "content_overlap": content_overlap,
                        "score": score,
                    })
            
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
        
        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch: EvaluationBatch[dict, dict],
            components_to_update: list[str],
        ) -> dict[str, list[dict[str, Any]]]:
            """Build reflective dataset for instruction refinement."""
            ret_d: dict[str, list[dict[str, Any]]] = {}
            assert len(components_to_update) == 1
            comp = components_to_update[0]
            
            items: list[dict[str, Any]] = []
            if eval_batch.trajectories:
                for traj in eval_batch.trajectories:
                    data = traj.get("data", {})
                    response = traj.get("response", "")
                    target = traj.get("target_response", "")
                    placeholder_used = traj.get("placeholder_used", False)
                    content_overlap = traj.get("content_overlap", 0.0)
                    score = traj.get("score", 0.0)
                    
                    category = data.get("category", data.get("predicted_category", ""))
                    redacted_query = data.get("redacted_query", "")[:300]
                    
                    if placeholder_used:
                        placeholder_feedback = "Good: Response uses anonymised placeholders."
                    else:
                        placeholder_feedback = "Issue: Response should use anonymised placeholders (PRESIDIO_ANONYMIZED or [REDACTED])."
                    
                    if content_overlap > 0.5:
                        overlap_feedback = f"Good: Response content overlaps well with target (overlap: {content_overlap:.2f})."
                    else:
                        overlap_feedback = f"Issue: Response content overlap is low (overlap: {content_overlap:.2f})."
                    
                    feedback = f"{placeholder_feedback} {overlap_feedback}"
                    
                    items.append({
                        "Inputs": {"category": category, "redacted_query": redacted_query},
                        "Generated Outputs": {"response": response, "target_response": target},
                        "Feedback": feedback,
                    })
            
            ret_d[comp] = items if items else [{"Inputs": {}, "Generated Outputs": {}, "Feedback": "No trajectories"}]
            return ret_d
    
    # Determine API key and model based on provider
    if model is None:
        model = "groq/openai/gpt-oss-120b"  # Default fallback
    
    model_lower = model.lower()
    if "groq" in model_lower:
        # Get API key - check environment (orchestrator should have loaded it)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Try reloading from .env as fallback
            repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
            env_file = repo_root / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=True)  # Override to ensure we get latest
                api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
            raise ValueError(
                f"❌ GROQ_API_KEY required for Groq models (model: {model}).\n"
                f"   Please add GROQ_API_KEY to your .env file at: {repo_root / '.env'}\n"
                f"   Or set it in your shell environment before running the script."
            )
    elif "openai" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            openai_key_present = "OPENAI_API_KEY" in os.environ
            raise ValueError(
                f"OPENAI_API_KEY required for OpenAI models (model: {model}). "
                f"Environment variable present: {openai_key_present}. "
                f"Please set OPENAI_API_KEY in your .env file or environment."
            )
    else:
        # Default to Groq if provider unclear
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            groq_key_present = "GROQ_API_KEY" in os.environ
            raise ValueError(
                f"GROQ_API_KEY required (default provider, model: {model}). "
                f"Environment variable present: {groq_key_present}. "
                f"Please set GROQ_API_KEY in your .env file or environment."
            )
    
    pupa_examples = load_pupa_dataset(split="train")
    
    if train_seeds is None:
        train_seeds = list(range(50))
    if val_seeds is None:
        val_seeds = list(range(50, 80))
    
    train_examples = [pupa_examples[i] for i in train_seeds if i < len(pupa_examples)]
    val_examples = [pupa_examples[i] for i in val_seeds if i < len(pupa_examples)]
    
    trainset = [
        {
            "category": ex.get("predicted_category", ""),
            "redacted_query": ex.get("redacted_query", ""),
            "target_response": ex.get("target_response", ""),
            "response": ex.get("target_response", ""),
        }
        for ex in train_examples
    ]
    valset = [
        {
            "category": ex.get("predicted_category", ""),
            "redacted_query": ex.get("redacted_query", ""),
            "target_response": ex.get("target_response", ""),
            "response": ex.get("target_response", ""),
        }
        for ex in val_examples
    ]
    
    adapter = PUPAGEPAAdapter(model=model, api_key=api_key)
    
    seed_candidate = {
        "instruction": (
            "You must respond to the user while preserving privacy. "
            "Never reconstruct redacted details; use the anonymised placeholders as-is."
        )
    }
    
    if LearningCurveTracker is None:
        learning_curve = None
    else:
        learning_curve = LearningCurveTracker(
            framework="gepa_ai",
            benchmark="pupa",
            total_budget=rollout_budget,
        )
    
    print(f"📊 Evaluating baseline on {len(valset)} validation examples...")
    baseline_batch = adapter.evaluate(valset, seed_candidate, capture_traces=False)
    baseline_val = sum(baseline_batch.scores) / len(baseline_batch.scores) if baseline_batch.scores else 0.0
    print(f"✅ Baseline performance: {baseline_val:.4f} ({baseline_val*100:.1f}%)")
    
    if learning_curve is not None:
        learning_curve.curve.record(rollout_count=0, performance=baseline_val, checkpoint_pct=0.0)
    
    print(f"🚀 Starting GEPA-AI optimization")
    print(f"   Budget: {rollout_budget} metric calls")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")
    print(f"   Reflection minibatch size: {reflection_minibatch_size}")
    
    from gepa.api import optimize as gepa_optimize
    from gepa.logging.logger import Logger as GEPALogger
    
    gepa_log_file = output_dir / "gepa_optimization.log"
    
    with GEPALogger(str(gepa_log_file)):
        from gepa.logging.logger import StdOutLogger
        
        class QuietLogger(StdOutLogger):
            def __init__(self, log_file_path: str):
                self.file_handle = open(log_file_path, "a")
            def log(self, message: str):
                self.file_handle.write(message + "\n")
                self.file_handle.flush()
            def close(self):
                if hasattr(self, "file_handle"):
                    self.file_handle.close()
        
        quiet_logger = QuietLogger(str(log_file))
        
        prompt_log_file = output_dir / "gepa_ai_proposal_prompts.log"
        prompt_log_handle = open(prompt_log_file, "w")
        
        from gepa.strategies.instruction_proposal import InstructionProposalSignature
        original_run = InstructionProposalSignature.run
        
        def logged_run(cls, lm, input_dict):
            prompt = cls.prompt_renderer(input_dict)
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"GEPA-AI PROPOSAL PROMPT\n")
            prompt_log_handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"\nCurrent Instruction:\n{input_dict.get('current_instruction_doc', 'N/A')}\n")
            prompt_log_handle.write(f"\nDataset with Feedback ({len(input_dict.get('dataset_with_feedback', []))} examples):\n")
            prompt_log_handle.write(json.dumps(input_dict.get('dataset_with_feedback', []), indent=2, default=str))
            prompt_log_handle.write(f"\n\n--- FULL PROMPT SENT TO LLM ---\n{prompt}\n--- END PROMPT ---\n\n")
            prompt_log_handle.flush()
            
            result = original_run(lm, input_dict)
            
            prompt_log_handle.write(f"--- LLM RESPONSE ---\n{result.get('new_instruction', 'N/A')}\n--- END RESPONSE ---\n\n")
            prompt_log_handle.flush()
            
            return result
        
        InstructionProposalSignature.run = classmethod(logged_run)
        
        # Add timeout and progress monitoring
        import threading
        
        optimization_start_time = time.time()
        last_progress_time = optimization_start_time
        progress_check_interval = 30.0  # Check every 30 seconds
        timeout_seconds = rollout_budget * 10 + 600  # Conservative timeout: 10s per rollout + 10min buffer
        
        optimization_complete = threading.Event()
        
        def check_progress():
            """Periodically check if optimization is making progress."""
            while not optimization_complete.is_set():
                time.sleep(progress_check_interval)
                if optimization_complete.is_set():
                    break
                
                elapsed = time.time() - last_progress_time
                total_elapsed = time.time() - optimization_start_time
                
                if total_elapsed > timeout_seconds:
                    print(f"\n⚠️  Optimization timeout threshold reached ({timeout_seconds/60:.1f}m)")
                    print(f"   Check {log_file.name} for detailed progress")
                    print(f"   Check {gepa_log_file} for GEPA library logs")
                
                # Check log file size to see if it's growing
                try:
                    # Check both log files
                    log_files_to_check = []
                    if gepa_log_file.exists():
                        log_files_to_check.append(gepa_log_file)
                    if log_file.exists():
                        log_files_to_check.append(log_file)
                    
                    if log_files_to_check:
                        total_size = sum(f.stat().st_size for f in log_files_to_check)
                        if not hasattr(check_progress, "last_log_size"):
                            check_progress.last_log_size = total_size
                        
                        if total_size > check_progress.last_log_size:
                            check_progress.last_log_size = total_size
                            print(f"⏳ [pupa_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed, logs growing)")
                        else:
                            # Log hasn't grown - might be stuck
                            if elapsed > 120:  # No progress for 2 minutes
                                print(f"⚠️  [pupa_gepa_ai] No log updates for {elapsed/60:.1f}m - may be stuck")
                    else:
                        # No log files yet - print status anyway
                        if int(total_elapsed) % 60 == 0:  # Every minute
                            print(f"⏳ [pupa_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed)")
                except Exception:
                    # Don't fail on log check errors
                    if int(total_elapsed) % 60 == 0:  # Every minute
                        print(f"⏳ [pupa_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed)")
        
        progress_thread = threading.Thread(target=check_progress, daemon=True)
        progress_thread.start()
        
        try:
            print(f"   Starting optimization (timeout: {timeout_seconds/60:.1f}m, checking progress every {progress_check_interval}s)...")
            print(f"   Monitor progress in: {log_file.name}")
            
            result = gepa_optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                max_metric_calls=rollout_budget,
                reflection_lm="groq/llama-3.3-70b-versatile",
                reflection_minibatch_size=reflection_minibatch_size,
                run_dir=str(output_dir / "gepa_optimization"),
                display_progress_bar=False,
                logger=quiet_logger,
            )
            
            optimization_complete.set()
            elapsed = time.time() - optimization_start_time
            print(f"✅ Optimization completed in {elapsed/60:.1f}m")
            
        except KeyboardInterrupt:
            optimization_complete.set()
            print(f"\n⚠️  Optimization interrupted by user")
            raise
        except Exception as e:
            optimization_complete.set()
            elapsed = time.time() - optimization_start_time
            print(f"\n❌ Optimization failed after {elapsed/60:.1f}m: {e}")
            raise
        finally:
            optimization_complete.set()
            quiet_logger.close()
            prompt_log_handle.close()
            InstructionProposalSignature.run = original_run
            print(f"📝 Saved proposal prompts to: {prompt_log_file}")
    
    print(f"✅ Optimization complete")
    
    best_candidate = result.best_candidate
    print(f"📊 Evaluating best candidate on {len(valset)} validation examples...")
    final_batch = adapter.evaluate(valset, best_candidate, capture_traces=False)
    val_score_pct = sum(final_batch.scores) / len(final_batch.scores) if final_batch.scores else 0.0
    
    print(f"✅ Final validation score: {val_score_pct:.4f} ({val_score_pct*100:.1f}%)")
    print(f"   Improvement: {val_score_pct - baseline_val:+.4f} ({((val_score_pct - baseline_val) * 100):+.1f}%)")
    
    if learning_curve is not None:
        learning_curve.curve.record(rollout_count=rollout_budget, performance=val_score_pct, checkpoint_pct=1.0)
    
    total_time = time.time() - start_time
    if learning_curve is not None:
        learning_curve.save(output_dir)
    
    detailed_results_file = output_dir / "gepa_ai_detailed_results.json"
    
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "actual_rollouts": getattr(result, "total_metric_calls", rollout_budget),
        "total_time": total_time,
        "framework": "gepa_ai",
        "benchmark": "pupa",
        "candidates": [],
        "pareto_fronts": [],
        "lineage": [],
    }
    
    if hasattr(result, "candidates"):
        for i, candidate in enumerate(result.candidates):
            candidate_dict = {
                "candidate_num": i,
                "instruction": next(iter(candidate.values())) if isinstance(candidate, dict) else str(candidate),
                "is_best": candidate == best_candidate,
            }
            detailed_results["candidates"].append(candidate_dict)
    
    if hasattr(result, "pareto_fronts"):
        detailed_results["pareto_fronts"] = [
            [next(iter(c.values())) if isinstance(c, dict) else str(c) for c in front]
            for front in result.pareto_fronts
        ]
    
    if hasattr(result, "lineage"):
        detailed_results["lineage"] = [
            {
                "candidate": next(iter(c.values())) if isinstance(c, dict) else str(c),
                "parent": next(iter(p.values())) if isinstance(p, dict) else str(p) if p else None,
            }
            for c, p in result.lineage
        ]
    
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    readout_file = output_dir / "gepa_ai_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GEPA-AI PUPA OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Score: {baseline_val:.4f} ({baseline_val*100:.1f}%)\n")
        f.write(f"Best Score: {val_score_pct:.4f} ({val_score_pct*100:.1f}%)\n")
        f.write(f"Improvement: {val_score_pct - baseline_val:+.4f} ({((val_score_pct - baseline_val) * 100):+.1f}%)\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Total Rollouts: {rollout_budget}\n")
        f.write(f"Actual Rollouts: {detailed_results['actual_rollouts']}\n\n")
        f.write("BEST PROMPT\n")
        f.write("-" * 80 + "\n")
        best_instruction = next(iter(best_candidate.values())) if isinstance(best_candidate, dict) else str(best_candidate)
        f.write(f"{best_instruction}\n\n")
        f.write("ALL CANDIDATES\n")
        f.write("-" * 80 + "\n")
        for candidate_dict in detailed_results["candidates"]:
            f.write(f"\nCandidate #{candidate_dict['candidate_num']} {'(BEST)' if candidate_dict['is_best'] else ''}\n")
            f.write(f"Instruction: {candidate_dict['instruction']}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Detailed Results (JSON): {detailed_results_file}\n")
        f.write(f"Optimization Log: {log_file}\n")
        f.write(f"Proposal Prompts: {prompt_log_file}\n")
        f.write(f"GEPA Optimization Log: {gepa_log_file}\n")
        if learning_curve is not None:
            f.write(f"Learning Curve: {output_dir / 'learning_curve.json'}\n")
        f.write(f"Optimization Run Directory: {output_dir / 'gepa_optimization'}\n")
    
    print(f"📄 Saved readout to: {readout_file}")
    
    return {
        "baseline_score": float(baseline_val),
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "actual_rollouts": detailed_results["actual_rollouts"],
        "total_time": total_time,
        "readout_file": str(readout_file),
        "log_file": str(log_file),
        "results_file": str(detailed_results_file),
        "prompt_log_file": str(prompt_log_file),
    }


