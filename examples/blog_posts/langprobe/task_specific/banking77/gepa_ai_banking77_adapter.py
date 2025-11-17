"""GEPA-AI adapters for Banking77 intent classification.

This module provides adapters to run GEPA-AI library optimization on Banking77.
GEPA-AI uses example-driven feedback and instruction history for prompt optimization.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Import shared utilities from DSPy adapter
try:
    from .dspy_banking77_adapter import (
        Banking77Classifier,
        create_dspy_examples,
        get_available_intents,
        load_banking77_dataset,
        banking77_metric_gepa,
        LearningCurveTracker,
        _warn_if_dotenv_is_messy,
    )
except ImportError:
    # Fallback: try absolute import
    import sys
    from pathlib import Path as PathLib
    _script_dir = PathLib(__file__).resolve().parent
    _langprobe_dir = _script_dir.parent.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from task_specific.banking77.dspy_banking77_adapter import (
        Banking77Classifier,
        create_dspy_examples,
        get_available_intents,
        load_banking77_dataset,
        banking77_metric_gepa,
        LearningCurveTracker,
        _warn_if_dotenv_is_messy,
    )

load_dotenv()


async def run_gepa_ai_banking77(
    task_app_url: str = "http://127.0.0.1:8102",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 200,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Run GEPA-AI library optimization on Banking77.
    
    GEPA-AI uses example-driven feedback and instruction history for optimization.
    This implementation attempts to use a GEPA-AI library if available, or falls
    back to using DSPy's GEPA with GEPA-AI-like configuration.
    
    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-49, 50 examples)
        val_seeds: Validation seeds (default: 50-79, 30 examples)
        rollout_budget: Rollout budget (default: 200)
        reflection_minibatch_size: Number of examples for subsample evaluation (default: 3)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    import logging
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "gepa_ai"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    # Set up logging: redirect verbose logs to file, keep terminal clean
    log_file = output_dir / "gepa_ai.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    
    # Redirect gepa loggers to file
    gepa_logger = logging.getLogger("gepa")
    gepa_logger.setLevel(logging.DEBUG)
    gepa_logger.addHandler(file_handler)
    gepa_logger.propagate = False
    
    # Also redirect all gepa sub-loggers
    for logger_name in ["gepa.core", "gepa.proposer", "gepa.strategies", "gepa.utils"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.propagate = False
    
    print(f"📝 Verbose logs redirected to: {log_file}")
    
    # Check for gepa package (PyPI package name is "gepa")
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
    
    # ✅ IMPLEMENT: GEPAAdapter for Banking77
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
    from typing import Any
    import requests  # Use requests for synchronous HTTP calls
    
    class Banking77GEPAAdapter(GEPAAdapter[dict, dict, dict]):
        """GEPAAdapter for Banking77 intent classification."""
        
        def __init__(self, model: str, api_key: str, available_intents: list[str]):
            self.model = model
            self.api_key = api_key
            self.available_intents = available_intents
            self.base_url = "https://api.groq.com/openai/v1" if "groq" in model.lower() else "https://api.openai.com/v1"
        
        def evaluate(
            self,
            batch: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ) -> EvaluationBatch[dict, dict]:
            """Evaluate candidate on batch of Banking77 examples."""
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list[dict] | None = [] if capture_traces else None
            
            # Extract instruction from candidate (component name -> instruction text)
            instruction = next(iter(candidate.values())) if candidate else ""
            
            # Prepare messages for each example
            messages_list = []
            for data in batch:
                query = data.get("query", data.get("text", ""))
                intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(self.available_intents[:77])])
                
                system_msg = f"{instruction}\n\n**Available Banking Intents:**\n{intents_str}\n\n**Task:**\nCall the `banking77_classify` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query."
                user_msg = f"Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
                
                messages_list.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
            
            # Call LLM API (synchronous, since gepa adapter methods are sync)
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
                                    "name": "banking77_classify",
                                    "description": "Classify a banking query into one of 77 intents",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "intent": {
                                                "type": "string",
                                                "description": "The banking intent label",
                                            }
                                        },
                                        "required": ["intent"],
                                    },
                                },
                            }],
                            "tool_choice": {"type": "function", "function": {"name": "banking77_classify"}},
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
                expected_intent = data.get("intent", data.get("label_name", ""))
                
                # Extract predicted intent from tool calls
                predicted_intent = ""
                if tool_calls:
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name") == "banking77_classify":
                            import json
                            args_str = tc.get("function", {}).get("arguments", "{}")
                            try:
                                args = json.loads(args_str)
                                predicted_intent = args.get("intent", "")
                            except:
                                pass
                
                # Score: 1.0 if correct, 0.0 otherwise
                score = 1.0 if predicted_intent == expected_intent else 0.0
                
                output = {
                    "response": content,
                    "predicted_intent": predicted_intent,
                    "expected_intent": expected_intent,
                }
                outputs.append(output)
                scores.append(score)
                
                if capture_traces:
                    trajectories.append({
                        "data": data,
                        "response": content,
                        "predicted_intent": predicted_intent,
                        "expected_intent": expected_intent,
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
                    predicted = traj.get("predicted_intent", "")
                    expected = traj.get("expected_intent", "")
                    score = traj.get("score", 0.0)
                    response = traj.get("response", "")
                    
                    query = data.get("query", data.get("text", ""))
                    
                    if score > 0.0:
                        feedback = f"The classification is correct. The predicted intent '{predicted}' matches the expected intent '{expected}'."
                    else:
                        feedback = f"The classification is incorrect. The predicted intent was '{predicted}', but the correct intent is '{expected}'. The query was: '{query}'"
                    
                    items.append({
                        "Inputs": {"query": query, "available_intents": self.available_intents[:77]},
                        "Generated Outputs": {"predicted_intent": predicted, "response": response},
                        "Feedback": feedback,
                    })
            
            ret_d[comp] = items
            
            if len(items) == 0:
                # Fallback: create minimal dataset from outputs
                for output, score in zip(eval_batch.outputs, eval_batch.scores):
                    predicted = output.get("predicted_intent", "")
                    expected = output.get("expected_intent", "")
                    if score > 0.0:
                        feedback = f"Correct: predicted '{predicted}'"
                    else:
                        feedback = f"Incorrect: predicted '{predicted}', expected '{expected}'"
                    items.append({
                        "Inputs": {"query": "N/A"},
                        "Generated Outputs": {"predicted_intent": predicted},
                        "Feedback": feedback,
                    })
                ret_d[comp] = items
            
            return ret_d
    
    # Use gepa.api.optimize
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    # Load dataset
    banking77_examples = load_banking77_dataset(split="train")
    available_intents = get_available_intents()
    
    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(25))  # Default: 25 training examples
    if val_seeds is None:
        val_seeds = list(range(50, 150))  # Default: 100 validation examples
    
    # Filter examples by seeds
    train_examples = [banking77_examples[i] for i in train_seeds if i < len(banking77_examples)]
    val_examples = [banking77_examples[i] for i in val_seeds if i < len(banking77_examples)]
    
    # Convert to gepa DataInst format
    trainset = [
        {"query": ex.get("query", ex.get("text", "")), "intent": ex.get("intent", available_intents[ex.get("label", 0)]), "label": ex.get("label", 0)}
        for ex in train_examples
    ]
    valset = [
        {"query": ex.get("query", ex.get("text", "")), "intent": ex.get("intent", available_intents[ex.get("label", 0)]), "label": ex.get("label", 0)}
        for ex in val_examples
    ]
    
    # Create adapter
    if model is None:
        model = "groq/llama-3.1-8b-instant"  # Default fallback
    adapter = Banking77GEPAAdapter(
        model=model,
        api_key=groq_api_key,
        available_intents=available_intents,
    )
    
    # Initial seed candidate (component name -> instruction text)
    seed_candidate = {
        "instruction": (
            "You are an expert banking assistant that classifies customer queries into banking intents. "
            "Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
        )
    }
    
    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="gepa_ai",
        benchmark="banking77",
        total_budget=rollout_budget,
    )
    
    # Evaluate baseline
    print(f"📊 Evaluating baseline on {len(valset)} validation examples...")
    baseline_batch = adapter.evaluate(valset, seed_candidate, capture_traces=False)
    baseline_val = sum(baseline_batch.scores) / len(baseline_batch.scores) if baseline_batch.scores else 0.0
    print(f"✅ Baseline performance: {baseline_val:.4f} ({baseline_val*100:.1f}%)")
    
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
    print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")
    
    from gepa.api import optimize as gepa_optimize
    from gepa.logging.logger import Logger as GEPALogger
    
    # Use gepa's Logger context manager to redirect all stdout/stderr to log file
    # This captures both logger.log() calls and print statements from gepa
    gepa_log_file = output_dir / "gepa_optimization.log"
    
    # Print high-level message before entering context (so it shows in terminal)
    print(f"   Running optimization (detailed logs in {log_file.name})...")
    
    with GEPALogger(str(gepa_log_file)):
        # Suppress progress bar and use quiet logger
        from gepa.logging.logger import StdOutLogger
        
        class QuietLogger(StdOutLogger):
            """Logger that writes to file instead of stdout."""
            def __init__(self, log_file_path: str):
                self.log_file_path = log_file_path
                self.file_handle = open(log_file_path, "a")
            
            def log(self, message: str):
                # Write to file, not stdout
                self.file_handle.write(message + "\n")
                self.file_handle.flush()
            
            def close(self):
                if hasattr(self, "file_handle"):
                    self.file_handle.close()
        
        quiet_logger = QuietLogger(str(log_file))
        
        # ✅ ADD: Create prompt log file and monkey-patch InstructionProposalSignature to log prompts
        prompt_log_file = output_dir / "gepa_ai_proposal_prompts.log"
        prompt_log_handle = open(prompt_log_file, "w")
        
        # Monkey-patch InstructionProposalSignature.run() to log prompts
        from gepa.strategies.instruction_proposal import InstructionProposalSignature
        original_run = InstructionProposalSignature.run
        
        def logged_run(cls, lm, input_dict):
            """Wrapper that logs the prompt before calling the original run method."""
            # Render the prompt
            prompt = cls.prompt_renderer(input_dict)
            
            # Log the prompt with metadata
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"GEPA-AI PROPOSAL PROMPT\n")
            prompt_log_handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            prompt_log_handle.write("=" * 80 + "\n")
            prompt_log_handle.write(f"\nCurrent Instruction:\n")
            prompt_log_handle.write(f"{input_dict.get('current_instruction_doc', 'N/A')}\n")
            prompt_log_handle.write(f"\nDataset with Feedback ({len(input_dict.get('dataset_with_feedback', []))} examples):\n")
            import json
            prompt_log_handle.write(json.dumps(input_dict.get('dataset_with_feedback', []), indent=2, default=str))
            prompt_log_handle.write(f"\n\n--- FULL PROMPT SENT TO LLM ---\n")
            prompt_log_handle.write(prompt)
            prompt_log_handle.write(f"\n--- END PROMPT ---\n\n")
            prompt_log_handle.flush()
            
            # Call original method
            result = original_run(lm, input_dict)
            
            # Log the response
            prompt_log_handle.write(f"--- LLM RESPONSE ---\n")
            prompt_log_handle.write(f"{result.get('new_instruction', 'N/A')}\n")
            prompt_log_handle.write(f"--- END RESPONSE ---\n\n")
            prompt_log_handle.flush()
            
            return result
        
        # Replace the classmethod
        InstructionProposalSignature.run = classmethod(logged_run)
        
        try:
            result = gepa_optimize(
                seed_candidate=seed_candidate,
                trainset=trainset,
                valset=valset,
                adapter=adapter,
                max_metric_calls=rollout_budget,
                reflection_lm="groq/llama-3.3-70b-versatile",
                reflection_minibatch_size=reflection_minibatch_size,
                run_dir=str(output_dir / "gepa_optimization"),
                display_progress_bar=False,  # Disable progress bar (too verbose)
                logger=quiet_logger,  # Use quiet file logger
            )
        finally:
            quiet_logger.close()
            prompt_log_handle.close()
            # Restore original method
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
    
    learning_curve.curve.record(
        rollout_count=rollout_budget,
        performance=val_score_pct,
        checkpoint_pct=1.0,
    )
    
    total_time = time.time() - start_time
    learning_curve.save(output_dir)
    
    # Save detailed results with ALL gepa information
    detailed_results_file = output_dir / "gepa_ai_detailed_results.json"
    
    # Extract comprehensive information from GEPAResult
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "actual_rollouts": getattr(result, "total_metric_calls", rollout_budget),
        "total_time": total_time,
        "framework": "gepa_ai",
        "num_candidates": len(result.candidates) if hasattr(result, "candidates") else 0,
        "num_full_val_evals": getattr(result, "num_full_val_evals", None),
        "run_dir": getattr(result, "run_dir", None),
        "seed": getattr(result, "seed", None),
        "best_idx": result.best_idx if hasattr(result, "best_idx") else None,
    }
    
    # Extract ALL candidates with full details
    candidates_list = []
    if hasattr(result, "candidates") and result.candidates:
        for i, cand in enumerate(result.candidates):
            cand_info = {
                "candidate_num": i,
                "instructions": dict(cand) if isinstance(cand, dict) else {"instruction": str(cand)},
                "is_best": i == result.best_idx if hasattr(result, "best_idx") else False,
                "val_aggregate_score": result.val_aggregate_scores[i] if hasattr(result, "val_aggregate_scores") and i < len(result.val_aggregate_scores) else None,
                "val_subscores": result.val_subscores[i] if hasattr(result, "val_subscores") and i < len(result.val_subscores) else None,
                "discovery_eval_count": result.discovery_eval_counts[i] if hasattr(result, "discovery_eval_counts") and i < len(result.discovery_eval_counts) else None,
                "parents": result.parents[i] if hasattr(result, "parents") and i < len(result.parents) else None,
            }
            candidates_list.append(cand_info)
    
    detailed_results["candidates"] = candidates_list
    
    # Extract pareto front information (pools)
    pareto_fronts = []
    if hasattr(result, "per_val_instance_best_candidates") and result.per_val_instance_best_candidates:
        for task_idx, candidate_indices in enumerate(result.per_val_instance_best_candidates):
            pareto_info = {
                "task_idx": task_idx,
                "candidate_indices": list(candidate_indices) if isinstance(candidate_indices, set) else candidate_indices,
                "candidates": [
                    {
                        "idx": idx,
                        "instructions": dict(result.candidates[idx]) if idx < len(result.candidates) else None,
                        "val_score": result.val_aggregate_scores[idx] if idx < len(result.val_aggregate_scores) else None,
                    }
                    for idx in (candidate_indices if isinstance(candidate_indices, set) else candidate_indices)
                    if idx < len(result.candidates)
                ],
            }
            pareto_fronts.append(pareto_info)
    
    detailed_results["pareto_fronts"] = pareto_fronts
    
    # Extract lineage information
    lineage_info = []
    if hasattr(result, "candidates") and hasattr(result, "parents"):
        for i in range(len(result.candidates)):
            lineage = {
                "candidate_idx": i,
                "instructions": dict(result.candidates[i]) if isinstance(result.candidates[i], dict) else {"instruction": str(result.candidates[i])},
                "parent_indices": result.parents[i] if i < len(result.parents) else None,
                "lineage_chain": [],  # Will populate below
            }
            
            # Build lineage chain (trace back to seed)
            chain = [i]
            current_parents = result.parents[i] if i < len(result.parents) else None
            visited = {i}
            while current_parents:
                # Get first non-None parent
                parent_idx = None
                for p in current_parents:
                    if p is not None and p not in visited:
                        parent_idx = p
                        break
                if parent_idx is None:
                    break
                chain.append(parent_idx)
                visited.add(parent_idx)
                current_parents = result.parents[parent_idx] if parent_idx < len(result.parents) else None
            
            lineage["lineage_chain"] = chain[::-1]  # Reverse to show seed -> candidate
            lineage_info.append(lineage)
    
    detailed_results["lineage"] = lineage_info
    
    # Extract best outputs per validation instance (if available)
    if hasattr(result, "best_outputs_valset") and result.best_outputs_valset:
        best_outputs = []
        for task_idx, outputs_list in enumerate(result.best_outputs_valset):
            task_outputs = {
                "task_idx": task_idx,
                "best_outputs": [
                    {
                        "program_idx": prog_idx,
                        "output": str(output) if not isinstance(output, (dict, list)) else output,
                    }
                    for prog_idx, output in outputs_list
                ],
            }
            best_outputs.append(task_outputs)
        detailed_results["best_outputs_per_task"] = best_outputs
    
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"📊 Saved detailed results to {detailed_results_file}")
    
    # Save optimized prompt
    prompt_file = output_dir / "gepa_ai_optimized_prompt.txt"
    with open(prompt_file, "w") as f:
        if isinstance(best_candidate, dict):
            for comp_name, instr_text in best_candidate.items():
                f.write(f"=== {comp_name} ===\n")
                f.write(str(instr_text))
                f.write("\n\n")
        else:
            f.write(str(best_candidate))
    
    # ✅ ADD: Save comprehensive readout file (like DSPy adapter)
    readout_file = output_dir / "gepa_ai_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GEPA-AI OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Benchmark: Banking77\n")
        f.write(f"Framework: GEPA-AI (gepa package)\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Rollout Budget: {rollout_budget}\n")
        f.write(f"Training Examples: {len(trainset)}\n")
        f.write(f"Validation Examples: {len(valset)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary
        f.write("📊 SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Score: {baseline_val:.4f} ({baseline_val*100:.1f}%)\n")
        f.write(f"Best Score:     {val_score_pct:.4f} ({val_score_pct*100:.1f}%)\n")
        improvement = ((val_score_pct - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0
        f.write(f"Improvement:    {improvement:+.1f}% relative ({(val_score_pct - baseline_val)*100:+.1f} pp absolute)\n")
        f.write(f"Total Time:     {total_time:.1f}s ({total_time/60:.1f}m)\n")
        f.write(f"Actual Rollouts: {getattr(result, 'total_metric_calls', rollout_budget)}\n")
        f.write(f"Total Candidates: {len(result.candidates) if hasattr(result, 'candidates') else 0}\n")
        f.write(f"Full Val Evals: {getattr(result, 'num_full_val_evals', 'N/A')}\n")
        f.write("\n")
        
        # All Candidates
        f.write("=" * 80 + "\n")
        f.write("📋 ALL CANDIDATES\n")
        f.write("-" * 80 + "\n")
        if hasattr(result, "candidates") and result.candidates:
            for i, cand in enumerate(result.candidates):
                is_best = i == result.best_idx if hasattr(result, "best_idx") else False
                val_score = result.val_aggregate_scores[i] if hasattr(result, "val_aggregate_scores") and i < len(result.val_aggregate_scores) else None
                discovery_rollouts = result.discovery_eval_counts[i] if hasattr(result, "discovery_eval_counts") and i < len(result.discovery_eval_counts) else None
                parents = result.parents[i] if hasattr(result, "parents") and i < len(result.parents) else None
                
                f.write(f"\n[Candidate {i}]")
                if is_best:
                    f.write(" ⭐ BEST")
                f.write(f"\n")
                f.write(f"  Validation Score: {val_score:.4f} ({val_score*100:.1f}%)" if val_score is not None else "  Validation Score: N/A")
                f.write(f"\n")
                f.write(f"  Discovery Rollouts: {discovery_rollouts}" if discovery_rollouts is not None else "  Discovery Rollouts: N/A")
                f.write(f"\n")
                f.write(f"  Parent Indices: {parents}" if parents else "  Parent Indices: None (seed)")
                f.write(f"\n")
                f.write(f"  Instructions:\n")
                if isinstance(cand, dict):
                    for comp_name, instr_text in cand.items():
                        f.write(f"    [{comp_name}]\n")
                        f.write(f"    {instr_text}\n")
                else:
                    f.write(f"    {str(cand)}\n")
        f.write("\n")
        
        # Pareto Fronts (Pools)
        f.write("=" * 80 + "\n")
        f.write("🏆 PARETO FRONTS (POOLS) PER VALIDATION INSTANCE\n")
        f.write("-" * 80 + "\n")
        if hasattr(result, "per_val_instance_best_candidates") and result.per_val_instance_best_candidates:
            for task_idx, candidate_indices in enumerate(result.per_val_instance_best_candidates):
                f.write(f"\n[Task {task_idx}] Pareto Front Candidates: {list(candidate_indices)}\n")
                for idx in candidate_indices:
                    if idx < len(result.candidates):
                        score = result.val_aggregate_scores[idx] if idx < len(result.val_aggregate_scores) else None
                        f.write(f"  Candidate {idx}: Score {score:.4f}" if score is not None else f"  Candidate {idx}: Score N/A")
                        f.write(f"\n")
        f.write("\n")
        
        # Lineage
        f.write("=" * 80 + "\n")
        f.write("🌳 CANDIDATE LINEAGE\n")
        f.write("-" * 80 + "\n")
        if hasattr(result, "candidates") and hasattr(result, "parents"):
            for i in range(len(result.candidates)):
                parents = result.parents[i] if i < len(result.parents) else None
                if parents:
                    # Build lineage chain
                    chain = [i]
                    current_parents = parents
                    visited = {i}
                    while current_parents:
                        parent_idx = None
                        for p in current_parents:
                            if p is not None and p not in visited:
                                parent_idx = p
                                break
                        if parent_idx is None:
                            break
                        chain.append(parent_idx)
                        visited.add(parent_idx)
                        current_parents = result.parents[parent_idx] if parent_idx < len(result.parents) else None
                    
                    chain_str = " -> ".join([f"Candidate {idx}" for idx in reversed(chain)])
                    f.write(f"Candidate {i}: {chain_str}\n")
                else:
                    f.write(f"Candidate {i}: Seed (no parents)\n")
        f.write("\n")
        
        # Best Prompt
        f.write("=" * 80 + "\n")
        f.write("🏆 BEST PROMPT\n")
        f.write("=" * 80 + "\n")
        if isinstance(best_candidate, dict):
            for comp_name, instr_text in best_candidate.items():
                f.write(f"\n{comp_name}:\n{str(instr_text)}\n\n")
        else:
            f.write(f"\n{str(best_candidate)}\n\n")
        
        # All Candidates
        if detailed_results.get("candidates"):
            f.write("=" * 80 + "\n")
            f.write(f"💡 ALL CANDIDATES ({len(detailed_results['candidates'])})\n")
            f.write("=" * 80 + "\n\n")
            for cand in detailed_results["candidates"]:
                cand_num = cand.get("candidate_num", "?")
                is_best = cand.get("is_best", False)
                best_marker = " 🏆 BEST" if is_best else ""
                f.write(f"[Candidate {cand_num}]{best_marker}\n")
                f.write("-" * 80 + "\n")
                instructions = cand.get("instructions", {})
                if instructions:
                    for comp_name, instr_text in instructions.items():
                        f.write(f"\n  {comp_name}:\n")
                        instr_str = str(instr_text)
                        if len(instr_str) > 500:
                            f.write(f"    {instr_str[:500]}...\n")
                            f.write(f"    [Truncated - full text in JSON file]\n")
                        else:
                            f.write(f"    {instr_str}\n")
                f.write("\n")
        
        # File references
        f.write("=" * 80 + "\n")
        f.write("📁 OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Detailed JSON results: {detailed_results_file}\n")
        f.write(f"  - Contains all candidates, scores, pareto fronts, lineage, and metadata\n")
        f.write(f"Optimized prompt: {prompt_file}\n")
        f.write(f"Verbose log: {log_file}\n")
        f.write(f"  - Contains gepa logger output (proposals, iterations, etc.)\n")
        gepa_opt_log = output_dir / "gepa_optimization.log"
        if gepa_opt_log.exists():
            f.write(f"Optimization stdout/stderr log: {gepa_opt_log}\n")
            f.write(f"  - Contains all iteration logs, proposals, and detailed progress\n")
        prompt_log_file = output_dir / "gepa_ai_proposal_prompts.log"
        if prompt_log_file.exists():
            f.write(f"Proposal prompts log: {prompt_log_file}\n")
            f.write(f"  - Contains ALL raw prompts sent to LLM for generating candidates\n")
            f.write(f"  - Includes current instruction, feedback dataset, and LLM responses\n")
        f.write(f"Optimization run directory: {output_dir / 'gepa_optimization'}\n")
        f.write(f"  - Contains gepa state files and artifacts\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"📄 Saved comprehensive readout to: {readout_file}")
    
    prompt_log_file = output_dir / "gepa_ai_proposal_prompts.log"
    
    return {
        "best_score": val_score_pct,
        "baseline_score": baseline_val,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "actual_rollouts": getattr(result, "total_metric_calls", rollout_budget),
        "total_time": total_time,
        "num_candidates": len(result.candidates) if hasattr(result, "candidates") else 0,
        "prompt_file": str(prompt_file),
        "results_file": str(detailed_results_file),
        "readout_file": str(readout_file),
        "log_file": str(log_file),
        "prompt_log_file": str(prompt_log_file) if prompt_log_file.exists() else None,
    }


async def _run_dspy_gepa_with_gepa_ai_config(
    task_app_url: str,
    train_seeds: Optional[list[int]],
    val_seeds: Optional[list[int]],
    rollout_budget: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Fallback: Run DSPy GEPA with GEPA-AI-like configuration.
    
    This uses DSPy's GEPA optimizer but configures it to emphasize example-driven
    feedback and instruction history, similar to GEPA-AI's approach.
    """
    import dspy
    import logging
    
    start_time = time.time()
    
    # Set up logging for fallback too
    log_file = output_dir / "gepa_ai.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.setLevel(logging.DEBUG)
    dspy_logger.addHandler(file_handler)
    dspy_logger.propagate = False
    
    for logger_name in ["dspy.teleprompt", "dspy.evaluate"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.propagate = False
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    # Load dataset
    banking77_examples = load_banking77_dataset(split="train")
    available_intents = get_available_intents()
    
    if train_seeds is None:
        train_seeds = list(range(50))
    if val_seeds is None:
        val_seeds = list(range(50, 80))
    
    train_examples = [banking77_examples[i] for i in train_seeds if i < len(banking77_examples)]
    val_examples = [banking77_examples[i] for i in val_seeds if i < len(banking77_examples)]
    
    trainset = create_dspy_examples(train_examples, available_intents)
    valset = create_dspy_examples(val_examples, available_intents)
    
    module = Banking77Classifier()
    
    learning_curve = LearningCurveTracker(
        framework="gepa_ai_fallback",
        benchmark="banking77",
        total_budget=rollout_budget,
    )
    
    # Configure DSPy LM using context() instead of configure() to avoid async task issues
    lm = dspy.LM("groq/llama-3.1-8b-instant", api_key=groq_api_key)
    # Use context() instead of configure() to work across async tasks
    with dspy.context(lm=lm):
        # Evaluate baseline
        def metric_fn(gold, pred, trace=None):
            return banking77_metric_gepa(gold, pred, trace)
        
        print(f"📊 Evaluating baseline on {len(valset)} validation examples...")
        from dspy.evaluate import Evaluate
        evaluate = Evaluate(devset=valset, metric=metric_fn, num_threads=1)
        baseline_score = evaluate(module)
        
        # Extract baseline score
        if isinstance(baseline_score, (int, float)):
            baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
        elif isinstance(baseline_score, dict):
            baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
        elif hasattr(baseline_score, "score"):
            baseline_val = float(baseline_score.score) / 100.0
        else:
            baseline_val = 0.0
        
        print(f"✅ Baseline performance: {baseline_val:.4f} ({baseline_val*100:.1f}%)")
        
        learning_curve.curve.record(
            rollout_count=0,
            performance=baseline_val,
            checkpoint_pct=0.0,
        )
        
        # Use DSPy GEPA with GEPA-AI-like emphasis on examples
        from dspy.teleprompt.gepa import GEPA
        
        max_metric_calls = int(rollout_budget)
        reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
        
        # Configure GEPA to emphasize example-driven feedback (GEPA-AI style)
        optimizer = GEPA(
            metric=metric_fn,
            max_metric_calls=max_metric_calls,
            reflection_lm=reflection_lm,
            track_stats=True,
            # GEPA-AI typically uses more example-driven feedback
            # These parameters may not exist in DSPy GEPA, but we try
        )
        
        print(f"🚀 Starting GEPA-AI-style optimization (using DSPy GEPA fallback)")
        print(f"   Budget: {rollout_budget} metric calls")
        print(f"   Training examples: {len(trainset)}")
        print(f"   Validation examples: {len(valset)}")
        print(f"   Note: GEPA uses subsample evaluation (3 examples) before full evaluation for efficiency")
        print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")
        
        optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)
        
        print(f"✅ Optimization complete")
        
        print(f"📊 Evaluating optimized module on {len(valset)} validation examples...")
        val_score = evaluate(optimized_module)
        
        if isinstance(val_score, (int, float)):
            val_score_pct = float(val_score) / 100.0 if val_score > 1 else float(val_score)
        elif isinstance(val_score, dict):
            val_score_pct = val_score.get("accuracy", val_score.get("score", 0.0))
        elif hasattr(val_score, "score"):
            val_score_pct = float(val_score.score) / 100.0
        else:
            val_score_pct = 0.0
        
        print(f"✅ Final validation score: {val_score_pct:.4f} ({val_score_pct*100:.1f}%)")
        print(f"   Improvement: {val_score_pct - baseline_val:+.4f} ({((val_score_pct - baseline_val) * 100):+.1f}%)")
        
        learning_curve.curve.record(
            rollout_count=rollout_budget,
            performance=val_score_pct,
            checkpoint_pct=1.0,
        )
        
        total_time = time.time() - start_time
        
        learning_curve.save(output_dir)
        
        detailed_results_file = output_dir / "gepa_ai_detailed_results.json"
        detailed_results = {
            "best_score": val_score_pct,
            "baseline_score": float(baseline_val),
            "total_rollouts": rollout_budget,
            "total_time": total_time,
            "candidates": [],
            "framework": "gepa_ai_fallback",
            "note": "Using DSPy GEPA as fallback (GEPA-AI library not found)",
        }
        
        if hasattr(optimized_module, "detailed_results"):
            gepa_results = optimized_module.detailed_results
            if hasattr(gepa_results, "total_metric_calls"):
                detailed_results["actual_rollouts"] = gepa_results.total_metric_calls
        
        with open(detailed_results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        prompt_file = output_dir / "gepa_ai_optimized_prompt.txt"
        with open(prompt_file, "w") as f:
            if hasattr(optimized_module, "named_predictors"):
                for pred_name, predictor in optimized_module.named_predictors():
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        f.write(f"=== {pred_name} ===\n")
                        f.write(str(predictor.signature.instructions))
                        f.write("\n\n")
        
        return {
            "best_score": val_score_pct,
            "baseline_score": baseline_val,
            "val_score": val_score_pct,
            "total_rollouts": rollout_budget,
            "actual_rollouts": detailed_results.get("actual_rollouts", rollout_budget),
            "total_time": total_time,
            "prompt_file": str(prompt_file),
            "results_file": str(detailed_results_file),
        }

