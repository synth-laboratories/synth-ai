#!/usr/bin/env python3
"""Compute lift between pre/post baseline JSONs for banking77."""
import json
from pathlib import Path

def load_metrics(path: Path) -> float:
    data = json.loads(path.read_text())
    agg = data.get("aggregate_metrics", {})
    reward = agg.get("mean_outcome_reward") or agg.get("accuracy") or data.get("accuracy") or 0.0
    return float(reward)

def main() -> None:
    pre_path = Path('artifacts/banking77_baseline_pre.json')
    post_path = Path('artifacts/banking77_baseline_post.json')
    out_path = Path('artifacts/banking77_eval_summary.json')

    pre = load_metrics(pre_path)
    post = load_metrics(post_path)
    lift = (post - pre) / pre if pre else 0.0
    summary = {
        "pre_mean_reward": pre,
        "post_mean_reward": post,
        "lift": round(lift, 6),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
