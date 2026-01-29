# Boosting demo (GEPA + XGBoost)

This demo uses GEPA to optimize a featurization prompt for tabular data. An LLM
turns each row into a fixed-length numeric feature vector, then a frozen XGBoost
training procedure scores the prompt by validation AUC. The goal is to see if
prompt-optimized features outperform baselines and comparisons (e.g. CAAFE).

## What it does
- Uses the Breast Cancer dataset from scikit-learn (fast, local, numeric).
- LLM featurizes each row into `BOOSTING_NUM_FEATURES` numeric features.
- XGBoost trains with fixed hyperparameters and scores on a fixed validation split.
- GEPA evolves the system prompt to maximize validation AUC.

## Quickstart
From `synth-ai/`:

```bash
uv run python demos/boosting/run_demo.py --local
```

## Requirements
```bash
uv pip install xgboost scikit-learn
```

## Environment variables
- `SYNTH_API_KEY`: Synth API key (auto-minted if missing)
- `SYNTH_BACKEND_URL`: override backend URL
- `BOOSTING_INFERENCE_URL`: inference URL for quick baseline vs optimized scoring
- `BOOSTING_NUM_FEATURES`: feature vector length (default: 10)
- `BOOSTING_TRAIN_SIZE`: number of training rows (default: 200)
- `BOOSTING_VAL_SIZE`: number of validation rows (default: 80)

## Notes
- The XGBoost training procedure is frozen (`XGB_PARAMS` in `localapi_boosting.py`).
- GEPA only optimizes the prompt; data splits and model hyperparameters stay fixed.
- `BOOSTING_INFERENCE_URL` should point at a Synth inference proxy or OpenAI-compatible
  `/v1/chat/completions` endpoint.

## Comparing to CAAFE
Use the same dataset and evaluation metric, then compare:
1) Raw XGBoost on original features
2) LLM-featurized XGBoost with baseline prompt
3) LLM-featurized XGBoost with GEPA-optimized prompt

Reference repo:
```
https://github.com/noahho/CAAFE
```
