### Pokemon TCG Gameplay Demo (`gepa_ptcg`)

This demo is a **Synth task app** for **playing** Pokémon TCG (not deckbuilding).

- The **LLM agent** plays as **P1**.
- The opponent is **deterministic AI v4** (engine-side).
- Reward is primarily **win rate**.

This demo runs as a LocalAPI task app and is usually executed via a Synth backend **eval job** (so you get tracing + standard job plumbing).

---

### How it works

#### Task app: `localapi_ptcg.py`
- Defines two fixed sample decks (`SAMPLE_DECK_1`, `SAMPLE_DECK_2`) and many seeded game instances.
- On each rollout:
  - Initializes a game from the selected instance (`p1_deck`, `p2_deck`, seeds).
  - Loops until game end or `max_steps`.
  - Calls the LLM (typically via the Synth **interceptor**) to choose the next action.
  - Applies the action in the engine.
  - Computes reward from the final outcome (win/loss).

#### Runner: `run_demo.py`
- Starts the task app locally.
- Submits an eval job to either:
  - local backend at `http://localhost:8000` (`--local`)
  - or the hosted backend (default)
- Prints mean reward and per-seed winners.

---

### Prerequisites

#### Backend (for interceptor evals)
- A Synth backend running (local mode expects `http://localhost:8000`) with the interceptor mounted at `/api/interceptor/v1`.

#### Engine (`tcg_py`)
This demo uses `tcg_py` from `engine-bench` (Rust engine bindings).

`engine-bench` repo: [JoshuaPurtell/engine-bench](https://github.com/JoshuaPurtell/engine-bench)

Default location this demo expects:

```bash
mkdir -p ~/Documents/GitHub
git clone https://github.com/JoshuaPurtell/engine-bench.git ~/Documents/GitHub/engine-bench
```

Override location if needed:

```bash
export ENGINE_BENCH_DIR=~/custom/path/engine-bench
```

If `maturin` is missing:

```bash
python -m pip install maturin
```

---

### Run (local backend + interceptor)

From this folder:

```bash
../../.venv/bin/python run_demo.py --local --model gpt-4.1-mini --num-games 3
```

Increase games (seeds):

```bash
../../.venv/bin/python run_demo.py --local --model gpt-4.1-mini --num-games 20
```

Notes:
- `--num-games` controls how many rollouts to run (seeds `0..N-1`).
- The task app currently runs with `concurrency=1` in `run_demo.py`.

