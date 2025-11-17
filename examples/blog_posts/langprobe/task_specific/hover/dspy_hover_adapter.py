"""DSPy adapters for HoVer claim verification using GEPA optimizer."""

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

# Handle imports for both module and direct script execution
try:
    from ...integrations.learning_curve_tracker import LearningCurveTracker
except ImportError:
    # When run as a script, add parent directories to path
    import sys
    from pathlib import Path as PathLib
    _script_dir = PathLib(__file__).resolve().parent
    _langprobe_dir = _script_dir.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from integrations.learning_curve_tracker import LearningCurveTracker

load_dotenv()


class HoVerVerification(dspy.Signature):
    """Verify Wikipedia claims using provided evidence.
    
    Given a claim and evidence passages, determine if the claim is SUPPORTED or REFUTED.
    """
    
    claim: str = dspy.InputField(desc="The claim to verify")
    evidence: str = dspy.InputField(desc="Evidence passages from Wikipedia")
    label: str = dspy.OutputField(desc="Label: SUPPORTED or REFUTED")
    rationale: str = dspy.OutputField(desc="Brief explanation of the reasoning")


class HoVerVerifier(dspy.Module):
    """DSPy module for HoVer claim verification."""
    
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought(HoVerVerification)
    
    def forward(self, claim: str, evidence: str) -> dspy.Prediction:
        """Verify claim from evidence.
        
        Args:
            claim: The claim to verify
            evidence: Evidence passages
            
        Returns:
            Prediction with label and rationale fields
        """
        result = self.verify(claim=claim, evidence=evidence)
        return result


def load_hover_dataset(split: str = "test") -> list[dict[str, Any]]:
    """Load HoVer dataset from HuggingFace and convert to list of dicts.
    
    Args:
        split: Dataset split (default: "test")
        
    Returns:
        List of examples with claim, evidence, and label
    """
    dataset = load_dataset("Dzeniks/hover", split=split)
    examples = []
    
    LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED"}
    
    for idx, row in enumerate(dataset):
        label_idx = int(row.get("label") or 0)
        label_text = LABEL_MAP.get(label_idx, "SUPPORTED")
        evidence = str(row.get("evidence") or "").strip()
        
        examples.append({
            "claim": str(row.get("claim") or ""),
            "evidence": evidence,
            "label": label_text,
            "seed": idx,
        })
    
    return examples


def create_dspy_examples(hover_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert HoVer examples to DSPy Examples.
    
    Args:
        hover_examples: List of HoVer example dicts
        
    Returns:
        List of DSPy Examples
    """
    dspy_examples = []
    for ex in hover_examples:
        dspy_ex = dspy.Example(
            claim=ex["claim"],
            evidence=ex["evidence"],
            label=ex["label"],
            rationale="",  # We don't have ground truth rationale
        ).with_inputs("claim", "evidence")
        dspy_examples.append(dspy_ex)
    return dspy_examples


def _warn_if_dotenv_is_messy():
    """Warn if .env file has non-standard lines (optional, non-fatal)."""
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


def _parse_label(response_text: str) -> tuple[str, str]:
    """Parse label and rationale from response text."""
    if not response_text:
        return "", ""
    lower = response_text.lower()
    label = ""
    rationale = ""
    if "label:" in lower:
        fragment = lower.split("label:", 1)[1]
        label_line = fragment.splitlines()[0]
        label = label_line.strip().upper()
    else:
        # fallback to first word
        label = response_text.strip().split()[0].upper() if response_text.strip() else ""
    if "rationale:" in lower:
        rationale_fragment = lower.split("rationale:", 1)[1]
        rationale = rationale_fragment.strip()
    return label, rationale


def hover_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for HoVer claim verification.
    
    Args:
        example: Ground truth example
        pred: Model prediction
        trace: Optional trace (unused)
        
    Returns:
        Score (1.0 if label correct, 0.0 if incorrect)
    """
    predicted_label = (pred.label or "").strip().upper()
    expected_label = (example.label or "").strip().upper()
    
    # Normalize labels
    predicted_normalized = predicted_label[:5] if predicted_label else ""
    expected_normalized = expected_label[:5] if expected_label else ""
    
    # Check if predicted starts with expected (handles "SUPPORTED" vs "SUPPORTED BY")
    return 1.0 if predicted_normalized.startswith(expected_normalized) or expected_normalized.startswith(predicted_normalized) else 0.0


def hover_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA-compatible metric function (5 arguments required)."""
    return hover_metric(gold, pred, trace)


async def run_dspy_gepa_hover(
    task_app_url: str = "http://127.0.0.1:8112",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on HoVer.
    
    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-49, 50 examples)
        val_seeds: Validation seeds (default: 50-79, 30 examples)
        rollout_budget: Rollout budget (default: 300)
        reflection_minibatch_size: Minibatch size for reflection evaluation (default: 3)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    import time
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "dspy_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    # Configure DSPy LM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    # Main LM: llama-3.1-8b-instant via Groq
    lm = dspy.LM("groq/llama-3.1-8b-instant", api_key=groq_api_key)
    
    # Track actual metric evaluations
    metric_calls = {"count": 0}
    
    def tracked_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Wrapped metric that counts calls (GEPA version)."""
        metric_calls["count"] += 1
        return hover_metric_gepa(gold, pred, trace, pred_name, pred_trace)
    
    # Load dataset
    hover_examples = load_hover_dataset(split="test")
    
    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(50))  # 0-49: 50 training examples
    if val_seeds is None:
        val_seeds = list(range(50, 80))  # 50-79: 30 validation examples
    
    # Filter examples by seeds
    train_examples = [hover_examples[i] for i in train_seeds if i < len(hover_examples)]
    val_examples = [hover_examples[i] for i in val_seeds if i < len(hover_examples)]
    
    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)
    
    # Create module
    module = HoVerVerifier()
    
    # Initialize GEPA optimizer
    from dspy.teleprompt.gepa import GEPA
    
    max_metric_calls = int(rollout_budget)
    # GEPA requires a reflection LM
    reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    
    # Redirect logging to file
    log_file = output_dir / "dspy_gepa.log"
    import logging
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.addHandler(file_handler)
    dspy_logger.setLevel(logging.DEBUG)
    
    print(f"📝 Verbose logs redirected to: {log_file}")
    
    # Create prompt log file for DSPy proposal prompts
    prompt_log_file = output_dir / "dspy_gepa_proposal_prompts.log"
    prompt_log_handle = open(prompt_log_file, "w")
    
    # Monkey-patch reflection_lm.forward() to capture prompts
    original_reflection_lm_forward = reflection_lm.forward
    
    gepa_call_count = {"count": 0}
    
    def logged_reflection_lm_forward(self, prompt=None, messages=None, **kwargs):
        """Wrap reflection_lm.forward() to capture GEPA proposal prompts."""
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
                    prompt_log_handle.write(f"Role: {msg.get('role', 'unknown')}\n")
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
        
        # Call original method
        result = original_reflection_lm_forward(prompt=prompt, messages=messages, **kwargs)
        
        # Log the response
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
    
    # Bind the wrapper to the reflection_lm instance
    import types
    reflection_lm.forward = types.MethodType(logged_reflection_lm_forward, reflection_lm)
    
    optimizer = GEPA(
        metric=tracked_metric_gepa,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        track_stats=True,
    )
    
    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="hover",
        total_budget=rollout_budget,
    )
    
    # Evaluate baseline (before optimization)
    from dspy.evaluate import Evaluate
    
    # Reset counter before baseline evaluation
    metric_calls["count"] = 0
    
    with dspy.context(lm=lm):
        evaluate = Evaluate(devset=valset, metric=tracked_metric_gepa, num_threads=1)
        baseline_score = evaluate(module)
        baseline_metric_calls = metric_calls["count"]
        
        # Extract the score
        if isinstance(baseline_score, (int, float)):
            baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
        elif isinstance(baseline_score, dict):
            baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
        elif hasattr(baseline_score, "score"):
            baseline_val = float(baseline_score.score) / 100.0
        else:
            baseline_val = 0.0
        
        print(f"📊 Baseline performance: {baseline_val:.4f}")
        
        # Record baseline checkpoint
        learning_curve.curve.record(
            rollout_count=0,
            performance=baseline_val,
            checkpoint_pct=0.0,
        )
        
        # Optimize with progress tracking
        print(f"🚀 Starting DSPy GEPA optimization")
        print(f"   Budget: {rollout_budget} metric calls")
        print(f"   Training examples: {len(trainset)}")
        print(f"   Validation examples: {len(valset)}")
        print(f"   Note: GEPA uses subsample evaluation ({reflection_minibatch_size} examples) before full evaluation for efficiency")
        print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")
        
        optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)
        
        # Evaluate optimized module
        val_score = evaluate(optimized_module)
        
        # Extract final score
        if isinstance(val_score, (int, float)):
            val_score_pct = float(val_score) / 100.0 if val_score > 1 else float(val_score)
        elif isinstance(val_score, dict):
            val_score_pct = val_score.get("accuracy", val_score.get("score", 0.0))
        elif hasattr(val_score, "score"):
            val_score_pct = float(val_score.score) / 100.0
        else:
            val_score_pct = 0.0
        
        # Record final checkpoint
        learning_curve.curve.record(
            rollout_count=rollout_budget,
            performance=val_score_pct,
            checkpoint_pct=1.0,
        )
    
    # Restore original reflection_lm method
    reflection_lm.forward = original_reflection_lm_forward
    prompt_log_handle.close()
    print(f"📝 Saved proposal prompts to: {prompt_log_file}")
    
    # Calculate time taken
    total_time = time.time() - start_time
    
    # Save results
    learning_curve.save(output_dir)
    
    # Save detailed optimization results
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
        
        # Extract candidate information
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
        
        # Add evolution/lineage information
        if hasattr(gepa_results, "parents"):
            for i, parent_list in enumerate(gepa_results.parents):
                evolution_info = {
                    "candidate_num": i,
                    "parents": parent_list if parent_list else [],
                }
                detailed_results["evolution"].append(evolution_info)
    
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create text readout file
    prompt_file = output_dir / "optimized_prompt.txt"
    readout_file = output_dir / "dspy_gepa_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA HOVER OPTIMIZATION RESULTS\n")
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

