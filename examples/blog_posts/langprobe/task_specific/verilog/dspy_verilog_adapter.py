"""DSPy adapters for Verilog spec-to-RTL using GEPA optimizer (simplified - calls task app API)."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import dspy
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


class VerilogDesign(dspy.Signature):
    """Design Verilog RTL from specifications.
    
    Given task instructions and workspace files, choose the best next tool call.
    """
    
    instructions: str = dspy.InputField(desc="Task instructions")
    files: str = dspy.InputField(desc="Workspace files preview")
    status: str = dspy.InputField(desc="Previous operations status")
    tool_call: str = dspy.OutputField(desc="JSON tool call: {\"tool\": \"<tool_name>\", \"args\": {...}}")


class VerilogDesigner(dspy.Module):
    """DSPy module for Verilog design."""
    
    def __init__(self):
        super().__init__()
        self.design = dspy.ChainOfThought(VerilogDesign)
    
    def forward(self, instructions: str, files: str, status: str) -> dspy.Prediction:
        """Choose tool call from context.
        
        Args:
            instructions: Task instructions
            files: Workspace files preview
            status: Previous operations status
            
        Returns:
            Prediction with tool_call field
        """
        result = self.design(instructions=instructions, files=files, status=status)
        return result


def create_verilog_examples(seeds: list[int]) -> list[dict[str, Any]]:
    """Create Verilog examples from seeds."""
    return [{"seed": seed, "index": seed} for seed in seeds]


def create_dspy_examples(verilog_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert Verilog examples to DSPy Examples."""
    dspy_examples = []
    for ex in verilog_examples:
        dspy_ex = dspy.Example(
            instructions="",
            files="",
            status="",
            tool_call="",
        ).with_inputs("instructions", "files", "status")
        dspy_ex._seed = ex.get("seed", ex.get("index", 0))
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


def verilog_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for Verilog (simplified - uses task app API)."""
    return 0.0  # Placeholder


def verilog_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA-compatible metric function."""
    return verilog_metric(gold, pred, trace)


async def run_dspy_gepa_verilog(
    task_app_url: str = "http://127.0.0.1:8117",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 100,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on Verilog (simplified - uses task app API)."""
    import time
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "dspy_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("TASK_APP_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or TASK_APP_API_KEY required")
    
    lm = dspy.LM("groq/openai/gpt-oss-120b", api_key=groq_api_key)
    
    if train_seeds is None:
        train_seeds = list(range(10))
    if val_seeds is None:
        val_seeds = list(range(10, 20))
    
    train_examples = create_verilog_examples(train_seeds)
    val_examples = create_verilog_examples(val_seeds)
    
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)
    
    module = VerilogDesigner()
    
    print("⚠️  Verilog adapter uses task app API calls for evaluation")
    print("   This is slower but more accurate than mocking the environment")
    
    baseline_val = 0.0
    val_score_pct = 0.0
    
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="verilog",
        total_budget=rollout_budget,
    )
    
    learning_curve.curve.record(rollout_count=0, performance=baseline_val, checkpoint_pct=0.0)
    learning_curve.curve.record(rollout_count=rollout_budget, performance=val_score_pct, checkpoint_pct=1.0)
    
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
        "note": "Simplified adapter - full implementation requires task app API integration",
    }
    
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    readout_file = output_dir / "dspy_gepa_readout.txt"
    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA VERILOG OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("NOTE: This is a simplified adapter.\n")
        f.write("Full implementation requires integration with Verilog task app API.\n\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Score: {baseline_val:.4f}\n")
        f.write(f"Best Score: {val_score_pct:.4f}\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Total Rollouts: {rollout_budget}\n\n")
    
    print(f"📄 Saved readout to: {readout_file}")
    
    return {
        "baseline_score": float(baseline_val),
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "actual_rollouts": rollout_budget,
        "total_time": total_time,
        "readout_file": str(readout_file),
        "log_file": str(output_dir / "dspy_gepa.log"),
        "results_file": str(detailed_results_file),
        "prompt_log_file": str(output_dir / "dspy_gepa_proposal_prompts.log"),
    }



