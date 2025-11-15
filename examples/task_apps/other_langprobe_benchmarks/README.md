# LangProBe Benchmark Task Apps

Task apps for LangProBe (Language Programs Benchmark) datasets.

## Implemented Task Apps

### Phase 1: High Priority

1. **GSM8K** (`gsm8k_task_app.py`)
   - **Dataset**: `openai/gsm8k` (config: "main")
   - **Port**: 8112
   - **Baseline**: `gsm8k_baseline.py`
   - **Description**: Grade school math word problems
   - **Status**: ✅ Validated (66.7% accuracy)

### Phase 3: Classification Tasks

2. **HeartDisease** (`heartdisease_task_app.py`)
   - **Dataset**: `buio/heart-disease`
   - **Port**: 8114
   - **Baseline**: `heartdisease_baseline.py`
   - **Description**: Medical classification task (binary: heart disease or not)
   - **Status**: ✅ Validated (60.0% accuracy)

3. **Iris** (`iris_task_app.py`)
   - **Dataset**: `scikit-learn/iris`
   - **Port**: 8115
   - **Baseline**: `iris_baseline.py`
   - **Description**: Classic ML classification dataset (3 classes: setosa, versicolor, virginica)
   - **Status**: ✅ Validated (40.0% accuracy)

## Descoped Task Apps

- **MATH** - Hendrycks dataset not available on HuggingFace Hub
- **HumanEval** - Dataset not available on HuggingFace Hub
- **AlfWorld** - C++ build dependencies fail on macOS (requires manual installation or Docker/Linux)

## Quick Start

### Start a Task App

```bash
# GSM8K
python -m examples.task_apps.other_langprobe_benchmarks.gsm8k_task_app --port 8112

# HeartDisease
python -m examples.task_apps.other_langprobe_benchmarks.heartdisease_task_app --port 8114

# Iris
python -m examples.task_apps.other_langprobe_benchmarks.iris_task_app --port 8115
```

### Run Baseline Evaluation

```bash
# GSM8K
python examples/task_apps/other_langprobe_benchmarks/gsm8k_baseline.py \
  --task-app-url http://127.0.0.1:8112 \
  --num-seeds 10 \
  --model gpt-5-nano

# HeartDisease
python examples/task_apps/other_langprobe_benchmarks/heartdisease_baseline.py \
  --task-app-url http://127.0.0.1:8114 \
  --num-seeds 10 \
  --model gpt-5-nano

# Iris
python examples/task_apps/other_langprobe_benchmarks/iris_baseline.py \
  --task-app-url http://127.0.0.1:8115 \
  --num-seeds 10 \
  --model gpt-5-nano
```

## Requirements

- Python 3.11+
- `OPENAI_API_KEY` environment variable set (or in `.env`)
- `ENVIRONMENT_API_KEY` environment variable set (or in `.env`)
- Task app running on specified port

## Baseline Scripts

Each baseline script:
- Runs evaluation on specified seeds
- Uses gpt-5-nano by default
- Reports accuracy/success rate and mean reward
- Exits with non-zero code if score is 0.0 (indicates a problem)

## Implementation Notes

- All task apps follow the `gepa_benchmarks` pattern
- Use `gepa_benchmarks.common` for shared utilities
- Single-turn tasks (GSM8K, HeartDisease, Iris) have max_turns=1
- Classification tasks (HeartDisease, Iris) use label normalization
- All validated with gpt-5-nano and show meaningful non-zero scores

## Next Steps

See [`langprobe.md`](../../../langprobe.md) for full planning document and remaining datasets.

