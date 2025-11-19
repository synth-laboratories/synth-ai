#!/usr/bin/env python3
"""Apply metric tracking changes to HotPotQA and Banking77 adapters."""

import re
from pathlib import Path

def update_adapter(adapter_path: Path, metric_name: str, metric_name_gepa: str):
    """Update an adapter file with metric tracking."""
    print(f"Processing {adapter_path}...")

    with open(adapter_path, "r") as f:
        content = f.read()

    # Step 1: Replace litellm tracking with metric tracking in MIPROv2 function
    old_pattern = r"""    # Track actual model calls
    import litellm
    actual_calls = \{"count": 0\}

    def track_success\(kwargs, completion_response, start_time, end_time\):
        actual_calls\["count"\] \+= 1

    litellm\.success_callback = \[track_success\]"""

    new_text = f"""    # Track actual metric evaluations
    metric_calls = {{"count": 0}}

    def tracked_metric(gold, pred, trace=None):
        \"\"\"Wrapped metric that counts calls.\"\"\"
        metric_calls["count"] += 1
        return {metric_name}(gold, pred, trace)"""

    content = re.sub(old_pattern, new_text, content)

    # Step 2: Update optimizer to use tracked_metric
    content = re.sub(
        rf'optimizer = MIPROv2\(metric={metric_name}, auto=auto_level\)',
        f'optimizer = MIPROv2(metric=tracked_metric, auto=auto_level)',
        content
    )

    # Step 3: Update baseline evaluation
    content = re.sub(
        rf'evaluate = Evaluate\(devset=valset, metric={metric_name}, num_threads=1\)',
        f'metric_calls["count"] = 0\n    evaluate = Evaluate(devset=valset, metric=tracked_metric, num_threads=1)',
        content
    )

    # Step 4: Add baseline_metric_calls tracking after baseline evaluation
    content = re.sub(
        r'(evaluate = Evaluate.*tracked_metric.*\n    baseline_score = evaluate\(module\))',
        r'\1\n    baseline_metric_calls = metric_calls["count"]',
        content
    )

    # Step 5: Add optimization metric tracking before compile
    content = re.sub(
        r'(print\(f"🚀 DSPy MIPROv2.*\n\n    optimized_module = optimizer\.compile)',
        r'\1',
        content
    )
    content = re.sub(
        r'(print\(f"🚀 DSPy MIPROv2.*\)\n\n)(    optimized_module = optimizer\.compile)',
        r'\1    # Reset counter for optimization phase (exclude baseline calls)\n    metric_calls["count"] = 0\n\2',
        content
    )

    # Step 6: Capture optimization_metric_calls after compile
    content = re.sub(
        r'(optimized_module = optimizer\.compile\(student=module, trainset=trainset, valset=valset\))\n\n(    # Evaluate)',
        r'\1\n    optimization_metric_calls = metric_calls["count"]\n\n\2',
        content
    )

    # Step 7: Update stats to use optimization_metric_calls
    content = re.sub(
        r'"actual_rollouts": actual_calls\["count"\]',
        r'"actual_rollouts": optimization_metric_calls',
        content
    )

    # Step 8: Add GEPA metric tracking
    content = re.sub(
        r'(# Main LM: gpt-oss-20b via Groq\n    lm = dspy\.LM.*\n    dspy\.configure.*\n\n)    # Track usage via litellm\n    import litellm\n    litellm\.success_callback = \[\]\n    litellm\.failure_callback = \[\]',
        rf'\1    # Track actual metric evaluations\n    metric_calls = {{"count": 0}}\n\n    def tracked_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):\n        """Wrapped metric that counts calls (GEPA version)."""\n        metric_calls["count"] += 1\n        return {metric_name_gepa}(gold, pred, trace, pred_name, pred_trace)',
        content
    )

    # Step 9: Update GEPA optimizer to use tracked metric
    content = re.sub(
        rf'optimizer = GEPA\(metric={metric_name_gepa},',
        f'optimizer = GEPA(metric=tracked_metric_gepa,',
        content
    )

    # Save the updated content
    with open(adapter_path, "w") as f:
        f.write(content)

    print(f"✓ Updated {adapter_path.name}")

# Update HotPotQA
hotpotqa_path = Path(__file__).parent.parent / "hotpotqa" / "dspy_hotpotqa_adapter.py"
if hotpotqa_path.exists():
    update_adapter(hotpotqa_path, "hotpotqa_metric", "hotpotqa_metric_gepa")

# Update Banking77
banking77_path = Path(__file__).parent.parent / "banking77" / "dspy_banking77_adapter.py"
if banking77_path.exists():
    update_adapter(banking77_path, "banking77_metric", "banking77_metric_gepa")

print("\n✅ All adapters updated with metric tracking!")
