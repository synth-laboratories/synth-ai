# RLM MIPRO (OOLONG)

Online MIPRO demo for the MIT RLM task on OOLONG.

## Run

```bash
uv run python demos/rlm_mipro/run_demo.py
```

Common options:

```bash
uv run python demos/rlm_mipro/run_demo.py --rollouts 5 --model gpt-4.1-mini
SYNTH_BACKEND_URL=https://api-dev.usesynth.ai uv run python demos/rlm_mipro/run_demo.py
```

## Notes

- Requires `SYNTH_API_KEY` for the Synth backend.
- Online mode does not require tunneling; rollouts are driven locally.
- Uses the OOLONG dataset (`oolongbench/oolong-real`).
