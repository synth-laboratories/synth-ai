# Math Single-Step Task App

This directory hosts the legacy entrypoint for the math single-step task app. Prefer starting the app via:

```bash
uvx synth-ai serve math-single-step --env-file examples/rl/.env --port 8101
```

If you need to run it directly (e.g., for Modal `modal deploy` compatibility), use:

```bash
python examples/rl/task_app/math_task_app.py --env-file examples/rl/.env --port 8101
```

Environment variables:

- `MATH_DATASET_NAME` – defaults to `EleutherAI/math`
- `MATH_DATASET_CONFIG` – defaults to `algebra__linear_1d`
- `MATH_DATASET_DEFAULT_SPLIT`, `MATH_DATASET_VALIDATION_SPLIT`, `MATH_DATASET_TEST_SPLIT`

The task app enforces a single `math_submit` tool call per episode, enabling RL to reward correct final answers and penalise missing or malformed submissions.

