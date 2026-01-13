### Pokemon TCG Deckbuilder Demo (GEPA-style task app)

This folder contains a **Synth task app** that evaluates an LLM’s ability to generate a **legal 60-card Pokémon TCG deck** (from a fixed pool) that also performs in **deterministic AI v4 vs v4** battles.

There are two ways to run it:
- **Interceptor-backed eval job (recommended for tracing + Synth eval plumbing)**: `run_demo.py` submits an eval job to a Synth backend and the task app calls the LLM via the backend **interceptor**.
- **Direct generate + score (no backend, no interceptor)**: `generate_and_score.py` calls the LLM directly (OpenAI client) and then scores locally via `tcg_py`.

---

### How it works

#### 1) Task app (`localapi_deckbuilder.py`)
- Exposes a standard task app API via `synth_ai.sdk.localapi`.
- On each rollout:
  - Builds a prompt with the **card pool** and **task-specific constraints**.
  - Calls the LLM (usually through the **Synth interceptor**).
  - Parses the returned JSON: `{"deck": ["card-id", ...]}`.
  - **Fail-fast validity gate (0 reward immediately)**:
    - Non-string / empty card IDs
    - Any card ID not in `CARD_POOL`
    - Any constraint failure from the current challenge (deck size, copy limits, no-ex, evo lines, ratios, etc.)
  - If valid, computes final reward:
    - `final_reward = 0.5 * constraint_score + 0.5 * win_rate`
    - Battles are run via `tcg_py` using deterministic **AI v4 vs AI v4**.

#### 2) Eval orchestration (`run_demo.py`)
- Starts the task app locally on a port (default `8018`, will auto-pick a new port on conflict).
- Submits an eval job to a backend:
  - Local mode expects a backend at `http://localhost:8000`
  - The backend injects an **interceptor inference URL** into `policy_config`
- Polls until the job completes and prints the aggregate score.

---

### Prerequisites

#### Backend (for interceptor evals)
- A Synth backend running on `http://localhost:8000` with the interceptor mounted at `/api/interceptor/v1`.

#### `tcg_py` engine (for battle scoring)
- This demo expects the local `engine-bench` repo (default):
  - `~/Documents/GitHub/engine-bench`
- The task app will attempt to build `tcg_py` via `maturin` if needed.

##### Clone `engine-bench`
`engine-bench` lives at [JoshuaPurtell/engine-bench](https://github.com/JoshuaPurtell/engine-bench).

Default location (matches this demo’s default):

```bash
mkdir -p ~/Documents/GitHub
git clone https://github.com/JoshuaPurtell/engine-bench.git ~/Documents/GitHub/engine-bench
```

If you want it somewhere else, set `ENGINE_BENCH_DIR`:

```bash
export ENGINE_BENCH_DIR=~/custom/path/engine-bench
```

If `maturin` is missing, install it in your environment:

```bash
python -m pip install maturin
```

---

### Running: interceptor-backed eval (recommended)

From this directory:

```bash
../../.venv/bin/python run_demo.py --local --model gpt-4.1-mini --num-seeds 5
```

Larger run (more rollouts):

```bash
../../.venv/bin/python run_demo.py --local --model gpt-4.1-mini --num-seeds 30
```

Notes:
- `--num-seeds N` runs N rollouts. Seeds map onto the 5 challenges in a round-robin.
- The eval is strict: if the deck violates any challenge requirement, the rollout reward is **0**.

---

### Running: direct generate + score (no backend)

This path is useful for fast iteration and debugging without the interceptor.

It will load `OPENAI_API_KEY` from the nearest `.env` up the directory tree (e.g. `synth-ai/.env`).

Quick smoke (10 games/opponent):

```bash
../../.venv/bin/python generate_and_score.py --challenge basic-deck --n 1 --games-per-opponent 10
```

Full default scoring is **all 6 comparison decks** at **500 games each**:

```bash
../../.venv/bin/python generate_and_score.py --challenge basic-deck --n 1
```

If you want to score only against the challenge’s configured opponent set:

```bash
../../.venv/bin/python generate_and_score.py --challenge basic-deck --use-challenge-opponents
```

---

---

### Running: Eval Jobs

Run evaluation jobs to benchmark a fixed prompt across multiple challenges:

```bash
../../.venv/bin/python run_demo.py --local --model gpt-4.1-mini --num-seeds 5
```

**What it does:**
- Starts the task app locally
- Submits an eval job to the backend
- Evaluates the prompt across multiple seeds (challenges)
- Returns aggregate scores and per-seed results

**Options:**
- `--local`: Use local backend (`localhost:8000`)
- `--model`: Model to use (e.g., `gpt-4.1-mini`)
- `--num-seeds`: Number of seeds to evaluate (round-robin through challenges)
- `--port`: Task app port (default: `8018`)

**Output:**
- Mean reward across all seeds
- Per-seed results (score, deck size, constraint satisfaction)
- Detailed constraint breakdown for each seed

---

### Running: GEPA Prompt Learning Jobs

Run GEPA (Genetic Evolution of Prompt Architectures) to optimize prompts:

```bash
../../.venv/bin/python run_gepa_job.py --local --config gepa_deckbuilder.toml
```

**What it does:**
- Starts the task app locally
- Submits a GEPA optimization job to the backend
- Evolves prompts over multiple generations using genetic algorithm
- Returns the best prompt and score

**Options:**
- `--local`: Use local backend (`localhost:8000`)
- `--config`: Path to GEPA config file (default: `gepa_deckbuilder.toml`)
- `--budget`: Override rollout budget (e.g., `--budget 100`)
- `--generations`: Override number of generations (e.g., `--generations 5`)
- `--port`: Task app port (default: `8018`)

**Config File (`gepa_deckbuilder.toml`):**
- `prompt_learning.gepa.rollout.budget`: Total rollouts (e.g., `50`)
- `prompt_learning.gepa.population.num_generations`: Number of generations (e.g., `3`)
- `prompt_learning.gepa.evaluation.seeds`: Seeds for optimization (e.g., `[0, 1, 2, 3, 4]`)
- `prompt_learning.gepa.evaluation.validation_seeds`: Optional validation seeds

**Output:**
- Best prompt found during optimization
- Best score (train and validation)
- Optimization progress over generations

**Example with overrides:**
```bash
../../.venv/bin/python run_gepa_job.py --local --budget 100 --generations 5
```

---

### Files you'll care about
- `localapi_deckbuilder.py`: task app implementation (prompting, parsing, strict validity checks, scoring).
- `run_demo.py`: runs an interceptor-backed eval job via a Synth backend.
- `run_gepa_job.py`: runs a GEPA prompt learning job to optimize prompts.
- `gepa_deckbuilder.toml`: GEPA configuration file.
- `generate_and_score.py`: direct LLM deck generation + local Rust-parallel scoring via `tcg_py`.
- `JOBS_SCOPING.md`: detailed documentation on running eval and GEPA jobs.

