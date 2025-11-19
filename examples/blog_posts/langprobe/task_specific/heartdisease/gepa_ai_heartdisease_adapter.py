"""GEPA-AI adapter for Heart Disease classification."""

from __future__ import annotations

import json
import logging
import os
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

# Import from dspy_heartdisease_adapter - try multiple strategies
try:
    # Strategy 1: Try absolute import from langprobe
    from task_specific.heartdisease.dspy_heartdisease_adapter import (
        create_dspy_examples,
        heartdisease_metric,
        load_heartdisease_dataset,
    )
except ImportError:
    try:
        # Strategy 2: Try relative import
        from .dspy_heartdisease_adapter import (
            create_dspy_examples,
            heartdisease_metric,
            load_heartdisease_dataset,
        )
    except ImportError:
        # Strategy 3: Import from same directory using importlib
        import importlib.util
        dspy_adapter_path = _script_dir / "dspy_heartdisease_adapter.py"
        if dspy_adapter_path.exists():
            spec = importlib.util.spec_from_file_location(
                "dspy_heartdisease_adapter",
                dspy_adapter_path
            )
            if spec and spec.loader:
                dspy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dspy_module)
                create_dspy_examples = dspy_module.create_dspy_examples
                heartdisease_metric = dspy_module.heartdisease_metric
                load_heartdisease_dataset = dspy_module.load_heartdisease_dataset
            else:
                raise ImportError(f"Could not load dspy_heartdisease_adapter from {dspy_adapter_path}")
        else:
            raise ImportError(f"dspy_heartdisease_adapter.py not found at {dspy_adapter_path}")

load_dotenv()


async def run_gepa_ai_heartdisease(
    task_app_url: str = "http://127.0.0.1:8114",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 500,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Run GEPA-AI library optimization on Heart Disease.
    
    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-29, 30 examples)
        val_seeds: Validation seeds (default: 30-79, 50 examples)
        rollout_budget: Rollout budget (default: 500)
        reflection_minibatch_size: Minibatch size for reflection evaluation (default: 3)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "gepa_ai"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Redirect verbose logging to file
    log_file = output_dir / "gepa_ai.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("gepa_ai_heartdisease")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    print(f"📝 Verbose logs redirected to: {log_file}")
    
    # Check for gepa package
    try:
        from gepa.api import optimize as gepa_optimize
        gepa_available = True
    except ImportError:
        gepa_available = False
    
    if not gepa_available:
        raise ImportError(
            "GEPA-AI package ('gepa') is not installed. "
            "Please install it with: uv add gepa\n"
            "Or: pip install gepa"
        )
    
    # Implement GEPAAdapter for HeartDisease
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
    import requests
    
    class HeartDiseaseGEPAAdapter(GEPAAdapter[dict, dict, dict]):
        """GEPAAdapter for Heart Disease classification."""
        
        def __init__(self, model: str, api_key: str):
            self.model = model
            self.api_key = api_key
            self.base_url = "https://api.groq.com/openai/v1" if "groq" in model.lower() else "https://api.openai.com/v1"
        
        def evaluate(
            self,
            batch: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch[dict, dict]:
            """Evaluate candidate on batch of Heart Disease examples."""
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list[dict] | None = [] if capture_traces else None
            
            # Extract instruction from candidate
            instruction = next(iter(candidate.values())) if candidate else ""
            
            # Prepare messages for each example
            messages_list = []
            for data in batch:
                features = data.get("features", data.get("feature_text", ""))
                
                system_msg = f"{instruction}\n\n**Task:**\nCall the `heart_disease_classify` tool with the `classification` parameter set to '1' for heart disease or '0' for no heart disease."
                user_msg = f"Patient Features:\n{features}\n\nClassify: Does this patient have heart disease? Respond with '1' for yes or '0' for no."
                
                messages_list.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
            
            # Call LLM API (synchronous)
            try:
                responses = []
                for messages in messages_list:
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model.replace("groq/", ""),
                            "messages": messages,
                            "tools": [{
                                "type": "function",
                                "function": {
                                    "name": "heart_disease_classify",
                                    "description": "Submit your classification prediction for the patient. Provide '1' for heart disease or '0' for no heart disease.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "classification": {
                                                "type": "string",
                                                "description": "The predicted classification: '1' for heart disease, '0' for no heart disease",
                                                "enum": ["0", "1"],
                                            },
                                        },
                                        "required": ["classification"],
                                    },
                                },
                            }],
                            "tool_choice": {"type": "function", "function": {"name": "heart_disease_classify"}},
                            "temperature": 0.0,
                            "max_tokens": 512,
                        },
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        resp_json = resp.json()
                        content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        tool_calls = resp_json.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                        responses.append((content, tool_calls))
                    else:
                        responses.append(("", []))
                
            except Exception as e:
                # On error, return failure scores
                for data in batch:
                    outputs.append({"response": "", "error": str(e)})
                    scores.append(0.0)
                    if capture_traces:
                        trajectories.append({"data": data, "response": "", "error": str(e)})
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
            
            # Process responses and compute scores
            for data, (content, tool_calls) in zip(batch, responses):
                expected_label = data.get("label", data.get("classification", "0"))
                
                # Extract predicted classification from tool calls
                predicted_classification = ""
                if tool_calls:
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name") == "heart_disease_classify":
                            args_str = tc.get("function", {}).get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                                predicted_classification = args.get("classification", "")
                            except:
                                pass
                
                # Normalize predictions (handle "yes/no", etc.)
                predicted = predicted_classification.lower().strip()
                if "yes" in predicted or "positive" in predicted or "disease" in predicted:
                    predicted = "1"
                elif "no" in predicted or "negative" in predicted or "healthy" in predicted:
                    predicted = "0"
                
                # Extract digits
                pred_digits = "".join([c for c in predicted if c.isdigit()])
                if pred_digits:
                    predicted = pred_digits[0]
                
                exp_digits = "".join([c for c in str(expected_label) if c.isdigit()])
                if exp_digits:
                    expected = exp_digits[0]
                else:
                    expected = str(expected_label)
                
                # Score: 1.0 if correct, 0.0 otherwise
                score = 1.0 if predicted == expected else 0.0
                
                output = {
                    "response": content,
                    "predicted_classification": predicted,
                    "expected_classification": expected,
                }
                outputs.append(output)
                scores.append(score)
                
                if capture_traces:
                    trajectories.append({
                        "data": data,
                        "response": content,
                        "predicted_classification": predicted,
                        "expected_classification": expected,
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
                    predicted = traj.get("predicted_classification", "")
                    expected = traj.get("expected_classification", "")
                    score = traj.get("score", 0.0)
                    response = traj.get("response", "")
                    
                    features = data.get("features", data.get("feature_text", ""))
                    
                    if score > 0.0:
                        feedback = f"The classification is correct. The predicted classification '{predicted}' matches the expected classification '{expected}'."
                    else:
                        feedback = f"The classification is incorrect. The predicted classification was '{predicted}', but the correct classification is '{expected}'. Patient features: {features[:200]}"
                    
                    items.append({
                        "Inputs": {"features": features},
                        "Generated Outputs": {"predicted_classification": predicted, "response": response},
                        "Feedback": feedback,
                    })
            
            ret_d[comp] = items
            
            if len(items) == 0:
                # Fallback: create minimal dataset from outputs
                for output, score in zip(eval_batch.outputs, eval_batch.scores):
                    predicted = output.get("predicted_classification", "")
                    expected = output.get("expected_classification", "")
                    if score > 0.0:
                        feedback = f"Correct: predicted '{predicted}'"
                    else:
                        feedback = f"Incorrect: predicted '{predicted}', expected '{expected}'"
                    items.append({
                        "Inputs": {"features": "N/A"},
                        "Generated Outputs": {"predicted_classification": predicted},
                        "Feedback": feedback,
                    })
                ret_d[comp] = items
            
            return ret_d
    
    # Determine API key and model based on provider
    if model is None:
        model = "groq/openai/gpt-oss-20b"  # Default fallback
    
    model_lower = model.lower()
    if "groq" in model_lower:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY required for Groq models (model: {model})")
    elif "openai" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY required for OpenAI models (model: {model})")
    else:
        # Default to Groq if provider unclear
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY required (default provider, model: {model})")
    
    # Load dataset
    heartdisease_examples = load_heartdisease_dataset(split="train")
    
    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(30))  # Default: 30 training examples
    if val_seeds is None:
        val_seeds = list(range(30, 80))  # Default: 50 validation examples
    
    # Filter examples by seeds
    train_examples = [heartdisease_examples[i] for i in train_seeds if i < len(heartdisease_examples)]
    val_examples = [heartdisease_examples[i] for i in val_seeds if i < len(heartdisease_examples)]
    
    # Convert to gepa DataInst format
    trainset = [
        {"features": ex.get("features", ""), "label": ex.get("label", "0"), "classification": ex.get("label", "0")}
        for ex in train_examples
    ]
    valset = [
        {"features": ex.get("features", ""), "label": ex.get("label", "0"), "classification": ex.get("label", "0")}
        for ex in val_examples
    ]
    
    # Create adapter
    adapter = HeartDiseaseGEPAAdapter(
        model=model,
        api_key=api_key,
    )
    
    # Initial seed candidate
    seed_candidate = {
        "instruction": (
            "You are a medical classification assistant. Based on the patient's features, "
            "classify whether they have heart disease. Respond with '1' for heart disease or '0' for no heart disease."
        )
    }
    
    # Learning curve tracker
    if LearningCurveTracker is None:
        learning_curve = None
    else:
        learning_curve = LearningCurveTracker(
            framework="gepa_ai",
            benchmark="heartdisease",
            total_budget=rollout_budget,
        )
    
    # Evaluate baseline
    print(f"📊 Evaluating baseline on {len(valset)} validation examples...")
    baseline_batch = adapter.evaluate(valset, seed_candidate, capture_traces=False)
    baseline_val = sum(baseline_batch.scores) / len(baseline_batch.scores) if baseline_batch.scores else 0.0
    print(f"✅ Baseline performance: {baseline_val:.4f} ({baseline_val*100:.1f}%)")
    
    if learning_curve is not None:
        learning_curve.curve.record(
            rollout_count=0,
            performance=baseline_val,
            checkpoint_pct=0.0,
        )
    
    # Run gepa optimization
    print(f"🚀 Starting GEPA-AI optimization")
    print(f"   Budget: {rollout_budget} metric calls")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")
    print(f"   Reflection minibatch size: {reflection_minibatch_size}")
    print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")
    
    from gepa.api import optimize as gepa_optimize
    from gepa.logging.logger import Logger as GEPALogger
    
    gepa_log_file = output_dir / "gepa_optimization.log"
    print(f"   Running optimization (detailed logs in {log_file.name})...")
    
    with GEPALogger(str(gepa_log_file)):
        from gepa.logging.logger import StdOutLogger
        
        class QuietLogger(StdOutLogger):
            """Logger that writes to file instead of stdout."""
            def __init__(self, log_file_path: str):
                self.log_file_path = log_file_path
                self.file_handle = open(log_file_path, "a")
            
            def log(self, message: str):
                self.file_handle.write(message + "\n")
                self.file_handle.flush()
            
            def close(self):
                if hasattr(self, "file_handle"):
                    self.file_handle.close()
        
        quiet_logger = QuietLogger(str(log_file))
        
        # Create prompt log file and monkey-patch InstructionProposalSignature
        prompt_log_file = output_dir / "gepa_ai_proposal_prompts.log"
        prompt_log_handle = open(prompt_log_file, "w")
        
        from gepa.strategies.instruction_proposal import InstructionProposalSignature
        original_run = InstructionProposalSignature.run
        
        def logged_run(cls, lm, input_dict):
            """Wrapper that logs the prompt before calling the original run method."""
            prompt = cls.prompt_renderer(input_dict)
            
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"GEPA-AI PROPOSAL PROMPT\n")
            prompt_log_handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"\nCurrent Instruction:\n")
            prompt_log_handle.write(f"{input_dict.get('current_instruction_doc', 'N/A')}\n")
            prompt_log_handle.write(f"\nDataset with Feedback ({len(input_dict.get('dataset_with_feedback', []))} examples):\n")
            prompt_log_handle.write(json.dumps(input_dict.get('dataset_with_feedback', []), indent=2, default=str))
            prompt_log_handle.write(f"\n\n--- FULL PROMPT SENT TO LLM ---\n")
            prompt_log_handle.write(prompt)
            prompt_log_handle.write(f"\n--- END PROMPT ---\n\n")
            prompt_log_handle.flush()
            
            result = original_run(lm, input_dict)
            
            prompt_log_handle.write(f"--- LLM RESPONSE ---\n")
            prompt_log_handle.write(f"{result.get('new_instruction', 'N/A')}\n")
            prompt_log_handle.write(f"--- END RESPONSE ---\n\n")
            prompt_log_handle.flush()
            
            return result
        
        InstructionProposalSignature.run = classmethod(logged_run)
        
        # Add timeout and progress monitoring
        import threading
        import signal
        
        optimization_start_time = time.time()
        last_progress_time = optimization_start_time
        progress_check_interval = 30.0  # Check every 30 seconds
        timeout_seconds = rollout_budget * 10 + 600  # Conservative timeout: 10s per rollout + 10min buffer
        
        progress_stopped = threading.Event()
        optimization_complete = threading.Event()
        result_container = {"result": None, "error": None}
        
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
                            print(f"⏳ [heartdisease_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed, logs growing)")
                        else:
                            # Log hasn't grown - might be stuck
                            if elapsed > 120:  # No progress for 2 minutes
                                print(f"⚠️  [heartdisease_gepa_ai] No log updates for {elapsed/60:.1f}m - may be stuck")
                    else:
                        # No log files yet - print status anyway
                        if int(total_elapsed) % 60 == 0:  # Every minute
                            print(f"⏳ [heartdisease_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed)")
                except Exception as e:
                    # Don't fail on log check errors
                    if int(total_elapsed) % 60 == 0:  # Every minute
                        print(f"⏳ [heartdisease_gepa_ai] Still running... ({total_elapsed/60:.1f}m elapsed)")
        
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
    
    # Evaluate best candidate
    best_candidate = result.best_candidate
    print(f"📊 Evaluating best candidate on {len(valset)} validation examples...")
    final_batch = adapter.evaluate(valset, best_candidate, capture_traces=False)
    val_score_pct = sum(final_batch.scores) / len(final_batch.scores) if final_batch.scores else 0.0
    
    print(f"✅ Final validation score: {val_score_pct:.4f} ({val_score_pct*100:.1f}%)")
    print(f"   Improvement: {val_score_pct - baseline_val:+.4f} ({((val_score_pct - baseline_val) * 100):+.1f}%)")
    
    if learning_curve is not None:
        learning_curve.curve.record(
            rollout_count=rollout_budget,
            performance=val_score_pct,
            checkpoint_pct=1.0,
        )
    
    total_time = time.time() - start_time
    if learning_curve is not None:
        learning_curve.save(output_dir)
    
    # Save detailed results
    detailed_results_file = output_dir / "gepa_ai_detailed_results.json"
    
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "actual_rollouts": getattr(result, "total_metric_calls", rollout_budget),
        "total_time": total_time,
        "framework": "gepa_ai",
        "benchmark": "heartdisease",
        "candidates": [],
        "pareto_fronts": [],
        "lineage": [],
    }
    
    # Extract candidates, pareto fronts, and lineage from result
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
    
    # Create text readout file
    readout_file = output_dir / "gepa_ai_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GEPA-AI HEART DISEASE OPTIMIZATION RESULTS\n")
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

