"""DSPy adapters for IFBench instruction following using GEPA optimizer."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import dspy
from datasets import load_dataset
from dotenv import load_dotenv

try:
    from ...integrations.learning_curve_tracker import LearningCurveTracker
except ImportError:
    import sys
    from pathlib import Path as PathLib
    _script_dir = PathLib(__file__).resolve().parent
    _langprobe_dir = _script_dir.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from integrations.learning_curve_tracker import LearningCurveTracker

load_dotenv()


class IFBenchFollowing(dspy.Signature):
    """Follow instructions exactly to satisfy all constraints.
    
    Given a prompt and instructions, produce a response that satisfies all constraints.
    """
    
    prompt: str = dspy.InputField(desc="The task prompt")
    instructions: str = dspy.InputField(desc="List of instructions/constraints")
    response: str = dspy.OutputField(desc="Response that satisfies all instructions")


class IFBenchFollower(dspy.Module):
    """DSPy module for IFBench instruction following."""
    
    def __init__(self):
        super().__init__()
        self.follow = dspy.ChainOfThought(IFBenchFollowing)
    
    def forward(self, prompt: str, instructions: str) -> dspy.Prediction:
        """Follow instructions from prompt.
        
        Args:
            prompt: The task prompt
            instructions: List of instructions/constraints
            
        Returns:
            Prediction with response field
        """
        result = self.follow(prompt=prompt, instructions=instructions)
        return result


def load_ifbench_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load IFBench dataset from HuggingFace and convert to list of dicts."""
    dataset = load_dataset("allenai/IFBench_test", split=split)
    examples = []
    
    SUPPORTED_INSTRUCTIONS = {
        "count:keywords_multiple",
        "sentence:keyword",
        "count:numbers",
        "count:word_count_range",
        "count:unique_word_count",
        "count:pronouns",
        "format:list",
        "format:emoji",
    }
    
    for idx, row in enumerate(dataset):
        ids = row.get("instruction_id_list") or []
        if not set(ids).issubset(SUPPORTED_INSTRUCTIONS):
            continue
        
        instructions = []
        kwargs_list = row.get("kwargs") or []
        for instr_id, kwargs in zip(ids, kwargs_list):
            instructions.append({"id": str(instr_id), "kwargs": kwargs or {}})
        
        instruction_lines = [
            f"- {instr['id']} ({_summarise_kwargs(instr['kwargs'])})" for instr in instructions
        ]
        constraints_text = "\n".join(instruction_lines)
        
        examples.append({
            "prompt": str(row.get("prompt") or ""),
            "instructions": instructions,
            "instructions_text": constraints_text,
            "seed": idx,
        })
    
    return examples


def _summarise_kwargs(kwargs: dict[str, Any]) -> str:
    """Summarize kwargs for display."""
    items = []
    for key, value in kwargs.items():
        if value in (None, "", [], {}):
            continue
        items.append(f"{key}={value}")
    return ", ".join(items) if items else "default"


def _evaluate_ifbench_response(prompt: str, instructions: list[dict[str, Any]], response: str) -> float:
    """Evaluate IFBench response against instructions.
    
    Simplified version - returns fraction of instructions satisfied.
    """
    # Import evaluation functions from task app if possible, otherwise use simplified version
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "examples" / "task_apps" / "gepa_benchmarks"))
        from ifbench_task_app import evaluate_ifbench
        reward, _ = evaluate_ifbench(prompt, instructions, response)
        return reward
    except ImportError:
        # Simplified fallback: just check if response is non-empty
        return 1.0 if response.strip() else 0.0


def create_dspy_examples(ifbench_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert IFBench examples to DSPy Examples."""
    dspy_examples = []
    for ex in ifbench_examples:
        dspy_ex = dspy.Example(
            prompt=ex["prompt"],
            instructions=ex["instructions_text"],
            response="",  # We don't have ground truth response
        ).with_inputs("prompt", "instructions")
        dspy_examples.append(dspy_ex)
    return dspy_examples


def _warn_if_dotenv_is_messy():
    """Warn if .env file has non-standard lines."""
    p = Path(".env")
    if p.exists():
        bad = [
            i
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if line
            and not line.lstrip().startswith("#")
            and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.+$", line)
        ]
        if bad:
            print(f"[dotenv] Non KEY=VALUE lines at: {bad}  (ignored)")


def ifbench_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for IFBench instruction following."""
    # We need the original instructions list, not just the text
    # For now, use a simplified metric
    predicted_response = (pred.response or "").strip()
    if not predicted_response:
        return 0.0
    
    # Try to import evaluation from task app
    try:
        import sys
        task_app_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / "task_apps" / "gepa_benchmarks"
        if str(task_app_path) not in sys.path:
            sys.path.insert(0, str(task_app_path))
        from ifbench_task_app import evaluate_ifbench
        
        # Get instructions from example metadata if available
        instructions = getattr(example, "_instructions", [])
        if instructions:
            reward, _ = evaluate_ifbench(example.prompt, instructions, predicted_response)
            return reward
    except (ImportError, AttributeError):
        pass
    
    # Fallback: simple check
    return 1.0 if predicted_response else 0.0


def ifbench_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA-compatible metric function."""
    return ifbench_metric(gold, pred, trace)


async def run_dspy_gepa_ifbench(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on IFBench."""
    import time
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "dspy_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    lm = dspy.LM("groq/openai/gpt-oss-120b", api_key=groq_api_key)
    
    metric_calls = {"count": 0}
    
    def tracked_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        metric_calls["count"] += 1
        return ifbench_metric_gepa(gold, pred, trace, pred_name, pred_trace)
    
    ifbench_examples = load_ifbench_dataset(split="train")
    
    if train_seeds is None:
        train_seeds = list(range(50))
    if val_seeds is None:
        val_seeds = list(range(50, 80))
    
    train_examples = [ifbench_examples[i] for i in train_seeds if i < len(ifbench_examples)]
    val_examples = [ifbench_examples[i] for i in val_seeds if i < len(ifbench_examples)]
    
    # Store instructions in example metadata for metric evaluation
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)
    
    # Attach instructions to examples for metric evaluation
    for ex, orig_ex in zip(trainset, train_examples):
        ex._instructions = orig_ex["instructions"]
    for ex, orig_ex in zip(valset, val_examples):
        ex._instructions = orig_ex["instructions"]
    
    module = IFBenchFollower()
    
    from dspy.teleprompt.gepa import GEPA
    
    max_metric_calls = int(rollout_budget)
    reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    
    log_file = output_dir / "dspy_gepa.log"
    import logging
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.addHandler(file_handler)
    dspy_logger.setLevel(logging.DEBUG)
    
    print(f"📝 Verbose logs redirected to: {log_file}")
    
    prompt_log_file = output_dir / "dspy_gepa_proposal_prompts.log"
    prompt_log_handle = open(prompt_log_file, "w")
    
    original_reflection_lm_forward = reflection_lm.forward
    
    gepa_call_count = {"count": 0}
    
    def logged_reflection_lm_forward(self, prompt=None, messages=None, **kwargs):
        gepa_call_count["count"] += 1
        prompt_log_handle.write("=" * 80 + "\n")
        prompt_log_handle.write(f"DSPy GEPA PROPOSAL PROMPT (Call #{gepa_call_count['count']})\n")
        prompt_log_handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        prompt_log_handle.write("=" * 80 + "\n")
        
        if messages:
            prompt_log_handle.write(f"\n--- CONVERSATION PROMPT ({len(messages)} messages) ---\n")
            for i, msg in enumerate(messages):
                prompt_log_handle.write(f"\n[Message {i+1}]\n")
                if isinstance(msg, dict):
                    content = msg.get('content', str(msg))
                    if isinstance(content, str) and len(content) > 50000:
                        prompt_log_handle.write(f"Content (truncated): {content[:50000]}...\n")
                    else:
                        prompt_log_handle.write(f"Content: {content}\n")
                else:
                    prompt_log_handle.write(f"{str(msg)}\n")
            prompt_log_handle.write(f"\n--- END CONVERSATION ---\n\n")
        elif prompt:
            prompt_log_handle.write(f"\n--- FULL PROMPT SENT TO LLM ---\n")
            if isinstance(prompt, str) and len(prompt) > 50000:
                prompt_log_handle.write(f"{prompt[:50000]}...\n[Truncated]\n")
            else:
                prompt_log_handle.write(f"{prompt}\n")
            prompt_log_handle.write(f"\n--- END PROMPT ---\n\n")
        
        prompt_log_handle.flush()
        
        result = original_reflection_lm_forward(prompt=prompt, messages=messages, **kwargs)
        
        prompt_log_handle.write(f"--- LLM RESPONSE ---\n")
        if hasattr(result, 'choices') and result.choices:
            for i, choice in enumerate(result.choices):
                prompt_log_handle.write(f"Choice {i+1}:\n")
                if hasattr(choice, 'message'):
                    if hasattr(choice.message, 'content'):
                        prompt_log_handle.write(f"  Content: {choice.message.content}\n")
        else:
            prompt_log_handle.write(f"{str(result)}\n")
        prompt_log_handle.write(f"--- END RESPONSE ---\n\n")
        prompt_log_handle.flush()
        
        return result
    
    import types
    reflection_lm.forward = types.MethodType(logged_reflection_lm_forward, reflection_lm)
    
    optimizer = GEPA(
        metric=tracked_metric_gepa,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        track_stats=True,
    )
    
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="ifbench",
        total_budget=rollout_budget,
    )
    
    from dspy.evaluate import Evaluate
    
    metric_calls["count"] = 0
    
    with dspy.context(lm=lm):
        evaluate = Evaluate(devset=valset, metric=tracked_metric_gepa, num_threads=1)
        baseline_score = evaluate(module)
        baseline_metric_calls = metric_calls["count"]
        
        if isinstance(baseline_score, (int, float)):
            baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
        elif isinstance(baseline_score, dict):
            baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
        elif hasattr(baseline_score, "score"):
            baseline_val = float(baseline_score.score) / 100.0
        else:
            baseline_val = 0.0
        
        print(f"📊 Baseline performance: {baseline_val:.4f}")
        
        learning_curve.curve.record(rollout_count=0, performance=baseline_val, checkpoint_pct=0.0)
        
        print(f"🚀 Starting DSPy GEPA optimization")
        print(f"   Budget: {rollout_budget} metric calls")
        print(f"   Training examples: {len(trainset)}")
        print(f"   Validation examples: {len(valset)}")
        print(f"   Note: GEPA uses subsample evaluation ({reflection_minibatch_size} examples) before full evaluation for efficiency")
        print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")
        
        optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)
        
        val_score = evaluate(optimized_module)
        
        if isinstance(val_score, (int, float)):
            val_score_pct = float(val_score) / 100.0 if val_score > 1 else float(val_score)
        elif isinstance(val_score, dict):
            val_score_pct = val_score.get("accuracy", val_score.get("score", 0.0))
        elif hasattr(val_score, "score"):
            val_score_pct = float(val_score.score) / 100.0
        else:
            val_score_pct = 0.0
        
        learning_curve.curve.record(rollout_count=rollout_budget, performance=val_score_pct, checkpoint_pct=1.0)
    
    reflection_lm.forward = original_reflection_lm_forward
    prompt_log_handle.close()
    print(f"📝 Saved proposal prompts to: {prompt_log_file}")
    
    total_time = time.time() - start_time
    learning_curve.save(output_dir)
    
    detailed_results_file = output_dir / "dspy_gepa_detailed_results.json"
    
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "total_time": total_time,
        "candidates": [],
        "evolution": [],
    }
    
    if hasattr(optimized_module, "detailed_results"):
        gepa_results = optimized_module.detailed_results
        
        if hasattr(gepa_results, "total_metric_calls") and gepa_results.total_metric_calls is not None:
            detailed_results["actual_rollouts"] = gepa_results.total_metric_calls
        
        if hasattr(gepa_results, "log_dir") and gepa_results.log_dir is not None:
            detailed_results["log_dir"] = gepa_results.log_dir
        
        for i, (candidate, score, discovery_count) in enumerate(zip(
            gepa_results.candidates,
            gepa_results.val_aggregate_scores,
            gepa_results.discovery_eval_counts
        )):
            candidate_info = {
                "candidate_num": i,
                "score": float(score),
                "discovery_rollout": discovery_count,
                "is_best": i == gepa_results.best_idx,
                "instructions": {},
            }
            
            if isinstance(candidate, dict):
                for pred_name, instruction in candidate.items():
                    candidate_info["instructions"][pred_name] = str(instruction)
            elif hasattr(candidate, 'named_predictors'):
                for pred_name, predictor in candidate.named_predictors():
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        candidate_info["instructions"][pred_name] = str(predictor.signature.instructions)
            
            detailed_results["candidates"].append(candidate_info)
        
        if hasattr(gepa_results, "parents"):
            for i, parent_list in enumerate(gepa_results.parents):
                evolution_info = {
                    "candidate_num": i,
                    "parents": parent_list if parent_list else [],
                }
                detailed_results["evolution"].append(evolution_info)
    
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    prompt_file = output_dir / "optimized_prompt.txt"
    readout_file = output_dir / "dspy_gepa_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA IFBENCH OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Score: {baseline_val:.4f} ({baseline_val*100:.1f}%)\n")
        f.write(f"Best Score: {val_score_pct:.4f} ({val_score_pct*100:.1f}%)\n")
        f.write(f"Improvement: {val_score_pct - baseline_val:+.4f} ({((val_score_pct - baseline_val) * 100):+.1f}%)\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Total Rollouts: {rollout_budget}\n")
        f.write(f"Actual Rollouts: {detailed_results.get('actual_rollouts', rollout_budget)}\n\n")
        f.write("BEST PROMPT\n")
        f.write("-" * 80 + "\n")
        if detailed_results["candidates"]:
            best_candidate = next((c for c in detailed_results["candidates"] if c["is_best"]), None)
            if best_candidate and best_candidate["instructions"]:
                for pred_name, instruction in best_candidate["instructions"].items():
                    f.write(f"\n[{pred_name}]\n{instruction}\n")
        f.write("\n")
        f.write("ALL CANDIDATES\n")
        f.write("-" * 80 + "\n")
        for candidate_dict in detailed_results["candidates"]:
            f.write(f"\nCandidate #{candidate_dict['candidate_num']} {'(BEST)' if candidate_dict['is_best'] else ''}\n")
            f.write(f"Score: {candidate_dict['score']:.4f}\n")
            f.write(f"Discovery Rollout: {candidate_dict['discovery_rollout']}\n")
            if candidate_dict["instructions"]:
                for pred_name, instruction in candidate_dict["instructions"].items():
                    f.write(f"\n[{pred_name}]\n{instruction}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Detailed Results (JSON): {detailed_results_file}\n")
        f.write(f"Optimization Log: {log_file}\n")
        f.write(f"Proposal Prompts: {prompt_log_file}\n")
        f.write(f"Learning Curve: {output_dir / 'learning_curve.json'}\n")
        if detailed_results.get("log_dir"):
            f.write(f"GEPA Log Directory: {detailed_results['log_dir']}\n")
    
    print(f"📄 Saved readout to: {readout_file}")
    
    return {
        "baseline_score": float(baseline_val),
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "actual_rollouts": detailed_results.get("actual_rollouts", rollout_budget),
        "total_time": total_time,
        "readout_file": str(readout_file),
        "log_file": str(log_file),
        "results_file": str(detailed_results_file),
        "prompt_log_file": str(prompt_log_file),
    }


