#!/usr/bin/env bash
set -euo pipefail

# One-button runner for deterministic Banking77 prompt opt with gpt-5.1-nano

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts"
PRE_JSON="${ARTIFACT_DIR}/banking77_baseline_pre.json"
POST_JSON="${ARTIFACT_DIR}/banking77_baseline_post.json"
SUMMARY_JSON="${ARTIFACT_DIR}/banking77_eval_summary.json"
PROMPT_CFG="${ROOT_DIR}/banking77_prompt_learning_config.toml"
PROMPT_OUT="${ARTIFACT_DIR}/banking77_prompt_opt.json"

mkdir -p "$ARTIFACT_DIR"

die() { echo "Error: $*" >&2; exit 1; }

echo "== Preflight: checking Docker =="
if ! docker info >/dev/null 2>&1; then
die "Docker is not running; please start Docker Desktop first."
fi

UVX_BIN="${UVX_BIN:-uvx}"
MODEL="gpt-5.1-nano"
SEEDS=(0 1 2 3 4 5)

echo "== Running pre-change baseline =="
seed_results=()
for s in "${SEEDS[@]}"; do
  tmp="${ARTIFACT_DIR}/baseline_pre_tmp_${s}.json"
  $UVX_BIN synth-ai baseline banking77 --split train --model "$MODEL" --seed "$s" --concurrency 1 > "$tmp" || die "pre baseline failed for seed $s"
  seed_results+=("$tmp")
done

python3 - <<PY
import json
files = ${seed_results}
seed_results = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    acc = data.get("aggregate", {}).get("accuracy") or data.get("aggregate_metrics", {}).get("mean_outcome_reward") or 0.0
    seed_results.append({"seed": int(data.get("seed", 0)), "accuracy": float(acc), "n_examples": int(data.get("n_examples", 100))})
agg = sum(x["accuracy"] for x in seed_results)/len(seed_results) if seed_results else 0.0
out = {
    "baseline_id": "banking77",
    "split": "train",
    "model": "gpt-5.1-nano",
    "aggregate_metrics": {
        "mean_outcome_reward": round(agg, 6),
        "success_rate": round(agg, 6),
        "total_tasks": len(seed_results),
        "successful_tasks": len(seed_results),
        "failed_tasks": 0,
    },
    "seed_results": seed_results,
}
with open("${PRE_JSON}", "w") as fh:
    json.dump(out, fh, indent=2)
print("Wrote", "${PRE_JSON}")
PY

echo "== Prompt optimization =="
$UVX_BIN synth-ai train --type prompt_learning --config "$PROMPT_CFG" --poll --output "$PROMPT_OUT" || die "prompt_learning failed"

echo "== Running post-change baseline (with optimized prompt) =="
seed_results=()
for s in "${SEEDS[@]}"; do
  tmp="${ARTIFACT_DIR}/baseline_post_tmp_${s}.json"
  $UVX_BIN synth-ai baseline banking77 --split train --model "$MODEL" --seed "$s" --concurrency 1 --prompt-file "$PROMPT_OUT" > "$tmp" || die "post baseline failed for seed $s"
  seed_results+=("$tmp")
done

python3 - <<PY
import json
files = ${seed_results}
seed_results = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    acc = data.get("aggregate", {}).get("accuracy") or data.get("aggregate_metrics", {}).get("mean_outcome_reward") or 0.0
    seed_results.append({"seed": int(data.get("seed", 0)), "accuracy": float(acc), "n_examples": int(data.get("n_examples", 100))})
agg = sum(x["accuracy"] for x in seed_results)/len(seed_results) if seed_results else 0.0
out = {
    "baseline_id": "banking77",
    "split": "train",
    "model": "gpt-5.1-nano",
    "aggregate_metrics": {
        "mean_outcome_reward": round(agg, 6),
        "success_rate": round(agg, 6),
        "total_tasks": len(seed_results),
        "successful_tasks": len(seed_results),
        "failed_tasks": 0,
    },
    "seed_results": seed_results,
}
with open("${POST_JSON}", "w") as fh:
    json.dump(out, fh, indent=2)
print("Wrote", "${POST_JSON}")
PY

echo "== Evaluating lift =="
python3 "${ROOT_DIR}/scripts/banking77_eval.py" --pre "$PRE_JSON" --post "$POST_JSON" --out "$SUMMARY_JSON" || die "Evaluator failed"

echo "Completed run. Summary at ${SUMMARY_JSON}"
