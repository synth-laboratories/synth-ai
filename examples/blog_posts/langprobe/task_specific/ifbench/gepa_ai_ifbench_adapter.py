"""GEPA-AI adapter for IFBench instruction following."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from ...integrations.learning_curve_tracker import LearningCurveTracker
    from .dspy_ifbench_adapter import (
        _summarise_kwargs,
        create_dspy_examples,
        load_ifbench_dataset,
    )
except ImportError:
    import sys
    from pathlib import Path as PathLib
    _script_dir = PathLib(__file__).resolve().parent
    _langprobe_dir = _script_dir.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from integrations.learning_curve_tracker import LearningCurveTracker
    from dspy_ifbench_adapter import (
        _summarise_kwargs,
        create_dspy_examples,
        load_ifbench_dataset,
    )

load_dotenv()


async def run_gepa_ai_ifbench(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run GEPA-AI library optimization on IFBench."""
    start_time = time.time()
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "gepa_ai"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "gepa_ai.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("gepa_ai_ifbench")
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
    
    # Import evaluation function from task app
    try:
        import sys
        task_app_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / "task_apps" / "gepa_benchmarks"
        if str(task_app_path) not in sys.path:
            sys.path.insert(0, str(task_app_path))
        from ifbench_task_app import evaluate_ifbench
    except ImportError:
        # Fallback: simplified evaluation
        def evaluate_ifbench(prompt: str, instructions: list[dict], response: str) -> tuple[float, dict]:
            return (1.0 if response.strip() else 0.0, {"passed": 1 if response.strip() else 0, "total": 1})
    
    class IFBenchGEPAAdapter(GEPAAdapter[dict, dict, dict]):
        """GEPAAdapter for IFBench instruction following."""
        
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
            """Evaluate candidate on batch of IFBench examples."""
            outputs: list[dict] = []
            scores: list[float] = []
            trajectories: list[dict] | None = [] if capture_traces else None
            
            instruction = next(iter(candidate.values())) if candidate else ""
            
            messages_list = []
            for data in batch:
                prompt = data.get("prompt", "")
                instructions_text = data.get("instructions_text", "")
                
                system_msg = f"{instruction}\n\n**Task:**\nYou must follow every instruction exactly. Produce a single response that satisfies all constraints."
                user_msg = f"Instructions:\n{instructions_text}\n\nTask:\n{prompt}"
                
                messages_list.append([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
            
            try:
                responses = []
                for messages in messages_list:
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model.replace("groq/", "").replace("openai/", ""),
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
                        responses.append("")
                
            except Exception as e:
                for data in batch:
                    outputs.append({"response": "", "error": str(e)})
                    scores.append(0.0)
                    if capture_traces:
                        trajectories.append({"data": data, "response": "", "error": str(e)})
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)
            
            for data, response_text in zip(batch, responses):
                prompt = data.get("prompt", "")
                instructions = data.get("instructions", [])
                
                reward, eval_details = evaluate_ifbench(prompt, instructions, response_text)
                
                output = {
                    "response": response_text,
                    "eval_details": eval_details,
                }
                outputs.append(output)
                scores.append(reward)
                
                if capture_traces:
                    trajectories.append({
                        "data": data,
                        "response": response_text,
                        "eval_details": eval_details,
                        "score": reward,
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
                    eval_details = traj.get("eval_details", {})
                    score = traj.get("score", 0.0)
                    
                    prompt = data.get("prompt", "")[:200]
                    instructions_text = data.get("instructions_text", "")[:300]
                    
                    passed = eval_details.get("passed", 0)
                    total = eval_details.get("total", 1)
                    
                    if score >= 1.0:
                        feedback = f"Perfect: All {total} instructions satisfied."
                    elif score > 0.5:
                        feedback = f"Partial: {passed}/{total} instructions satisfied."
                    else:
                        feedback = f"Poor: Only {passed}/{total} instructions satisfied. Review the constraints carefully."
                    
                    items.append({
                        "Inputs": {"prompt": prompt, "instructions": instructions_text},
                        "Generated Outputs": {"response": response},
                        "Feedback": feedback,
                    })
            
            ret_d[comp] = items if items else [{"Inputs": {}, "Generated Outputs": {}, "Feedback": "No trajectories"}]
            return ret_d
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    ifbench_examples = load_ifbench_dataset(split="train")
    
    if train_seeds is None:
        train_seeds = list(range(50))
    if val_seeds is None:
        val_seeds = list(range(50, 80))
    
    train_examples = [ifbench_examples[i] for i in train_seeds if i < len(ifbench_examples)]
    val_examples = [ifbench_examples[i] for i in val_seeds if i < len(ifbench_examples)]
    
    trainset = [
        {
            "prompt": ex.get("prompt", ""),
            "instructions": ex.get("instructions", []),
            "instructions_text": ex.get("instructions_text", ""),
        }
        for ex in train_examples
    ]
    valset = [
        {
            "prompt": ex.get("prompt", ""),
            "instructions": ex.get("instructions", []),
            "instructions_text": ex.get("instructions_text", ""),
        }
        for ex in val_examples
    ]
    
    model_name = "groq/openai/gpt-oss-120b"
    adapter = IFBenchGEPAAdapter(model=model_name, api_key=groq_api_key)
    
    seed_candidate = {
        "instruction": (
            "You must follow every instruction exactly. Produce a single response that satisfies all constraints."
        )
    }
    
    learning_curve = LearningCurveTracker(
        framework="gepa_ai",
        benchmark="ifbench",
        total_budget=rollout_budget,
    )
    
    print(f"📊 Evaluating baseline on {len(valset)} validation examples...")
    baseline_batch = adapter.evaluate(valset, seed_candidate, capture_traces=False)
    baseline_val = sum(baseline_batch.scores) / len(baseline_batch.scores) if baseline_batch.scores else 0.0
    print(f"✅ Baseline performance: {baseline_val:.4f} ({baseline_val*100:.1f}%)")
    
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
                display_progress_bar=False,
                logger=quiet_logger,
            )
        finally:
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
    
    learning_curve.curve.record(rollout_count=rollout_budget, performance=val_score_pct, checkpoint_pct=1.0)
    
    total_time = time.time() - start_time
    learning_curve.save(output_dir)
    
    detailed_results_file = output_dir / "gepa_ai_detailed_results.json"
    
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "actual_rollouts": getattr(result, "total_metric_calls", rollout_budget),
        "total_time": total_time,
        "framework": "gepa_ai",
        "benchmark": "ifbench",
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
        f.write("GEPA-AI IFBENCH OPTIMIZATION RESULTS\n")
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


