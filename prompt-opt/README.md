# prompt-opt

Apache-2.0 licensed prompt optimization package that combines:

- Horizons `mipro_v2` code (vendored under `vendor/mipro_v2`)
- Synth-compatible local/offline learning adapters
- `gepa-ai`-compatible entrypoints for drop-in usage
- Shared Rust core for MIPRO across Rust + Python surfaces

## What is included

- `rust/`
  - Canonical Rust core crate (`prompt-opt`) for MIPRO optimization.
- `rust/rlm_v1/`
  - Copied Rust RLM v1 crate (from Horizons) used by the Rust core when `proposer_backend="rlm"`.
- `rust_py/`
  - PyO3 bridge crate (`prompt_opt_rust`) exposing Python access to the same Rust core.
- `src/prompt_opt/adapters/synth_offline.py`
  - Local adapters for offline prompt optimization with Synth-shaped data and outputs.
- `src/prompt_opt/gepa_ai_compat.py`
  - `gepa-ai`-style `optimize(...)` entrypoint and adapter protocol.
- `src/prompt_opt/mipro.py`
  - Python wrapper that calls the Rust core bindings (`prompt_opt_rust`).
- `src/prompt_opt/dspy/miprov2.py`
  - DSPy-compatible `MIPROv2` in local-only mode (`backend_mode="local"`).
- `src/prompt_opt/dspy/gepa.py`
  - GEPA slot-in shim (`from prompt_opt.dspy import gepa`).
- `src/gepa/__init__.py`
  - Import compatibility shim so `import gepa` works against this package.
- `vendor/mipro_v2`
  - Vendored copy of the Horizons standalone `mipro_v2` crate source.

## Install (editable, local)

```bash
cd prompt-opt
pip install -e .
```

## Quick usage

```python
from gepa import optimize
from prompt_opt.adapters.synth_offline import LocalEvaluator, SynthOfflineLearningAdapter

def score_fn(example, candidate):
    expected = str(example.get("answer", "")).strip().lower()
    prompt = " ".join(candidate.values()).lower()
    return 1.0 if expected and expected in prompt else 0.0

adapter = SynthOfflineLearningAdapter(LocalEvaluator(score_fn=score_fn))
result = optimize(
    seed_candidate={"system_prompt": "Answer briefly."},
    trainset=[{"input": "Capital of France?", "answer": "paris"}],
    adapter=adapter,
    max_metric_calls=8,
)

print(result.best_candidate)
print(result.val_aggregate_scores[result.best_idx])
```

## Notes

- This package is local-only and does not call Synth backend APIs.
- The vendored `mipro_v2` source is included as code provenance from Horizons.
- MIPRO proposer backends are now explicitly: `single_prompt` and `rlm`.
- Python MIPRO should call the shared Rust core via `rust_py` bindings for parity with Rust behavior.

## Examples

- Local Rust-core MIPRO:
  - `examples/mipro_local_example.py`
- DSPy GEPA slot-in:
  - `examples/dspy_gepa_slot_example.py`
