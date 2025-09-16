# Synth AI

Modern Compound AI System Development

**Comprehensive AI Framework for Language Models, Environments, and Observability**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.3.dev0-orange)](https://pypi.org/project/synth-ai/)
![Coverage](https://img.shields.io/badge/coverage-0.0%25-red)
![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)

A unified framework combining language model capabilities, synthetic environments, and comprehensive tracing for building and evaluating AI agents.

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install synth-ai

# With research environments (includes game environments)
pip install synth-ai[research]

# Full installation with all providers
pip install synth-ai[all]
```

### Spinning Up

Start the Synth AI service daemon (includes sqld database + environment service):

```bash
# Start both database daemon (port 8080) and environment service (port 8901)
uvx synth-ai serve
```

#### Service Command Options

```bash
uvx synth-ai serve [OPTIONS]
```

**Available Options:**
- `--db-file` - Database file path (default: "synth_ai.db")
- `--sqld-port` - Port for sqld HTTP interface (default: 8080)  
- `--env-port` - Port for environment service (default: 8901)
- `--no-sqld` - Skip starting sqld database daemon
- `--no-env` - Skip starting environment service

**Examples:**
```bash
# Start with custom ports
uvx synth-ai serve --sqld-port 8081 --env-port 8902

# Start only the environment service
uvx synth-ai serve --no-sqld

# Start only the database service
uvx synth-ai serve --no-env
```

#### What the Serve Command Provides

**sqld Database Service (port 8080)**
- Local SQLite-compatible database server with HTTP API
- Automatically downloads and installs sqld binary if needed
- Provides persistent storage for agent interactions and traces

**Environment Service (port 8901)**
- FastAPI service for managing AI environments and tasks
- Built-in environments: Crafter, Sokoban, MiniGrid, TicTacToe, Verilog, NetHack, Enron
- RESTful API for environment initialization, stepping, and termination
- Dynamic environment registry for custom environments

In another terminal, run your first example:

```bash
# Run a Crafter agent demo with Gemini
./examples/run_crafter_demo.sh
```

This will:
- Start the sqld database daemon with HTTP API on port 8080
- Launch the environment service API on port 8901  
- Run a reactive agent in the Crafter environment using Gemini 1.5 Flash

#### Demos (Eval + Finetuning)

You can run interactive demos from the repo without remembering exact commands:

```bash
# Lists all available demos under examples/, then prompts you to choose
uvx synth-ai demo
```

Today this includes:
- Eval demo: `examples/evals/run_demo.sh`
  - Prompts for models, episodes, etc.
  - Runs Crafter rollouts with v3 tracing, then analyzes and filters traces
  - Writes a JSONL like `ft_data/evals_filtered.jsonl` for downstream use
- Finetuning demo: `examples/finetuning/synth_qwen/run_demo.sh`
  - Guides you through: rollouts → filter v3 traces → prepare SFT JSONL
  - Pair with `uvpm examples.finetuning.synth_qwen.sft_kickoff` to start an SFT job when ready

Notes:
- Ensure the service is running (`uvx synth-ai serve`) so v3 traces are recorded locally.
- Set API configuration for finetuning:
  - `export LEARNING_V2_BASE_URL="http://localhost:8000/api"` (or your proxy)
  - `export SYNTH_API_KEY="sk_live_..."`
- v3 trace data is stored under `traces/v3/synth_ai.db/` by default. Inspect with `uvx synth-ai traces`.
 - LM tracing: all model calls (prompts, outputs, tool calls, token usage, latency, cost) are automatically captured via v3 tracing and stored locally; inspect with `uvx synth-ai traces`.

### One-Command Demos

Quickly browse and launch interactive demos under `examples/`:

```bash
uvx synth-ai demo
```

This lists all `run_demo.sh` scripts found in the repo (e.g., eval comparisons, finetuning flows) and lets you pick one to run.


## Changelog (migrated from CHANGELOG.md)

# Changelog

All notable changes to this project are documented in this file.

## [0.2.4.dev6] - 2025-08-18
- Added: Wordle environment for simple RL testing, including engine, environment, taskset, and curated instances with helper generator script.
- Added: Wordle README and unit/integration tests (integrity checks and gameplay coverage).
- Added: Service routes and example wiring to expose Wordle tasks through the environment service.
- Changed: Bumped development version to 0.2.4.dev6 and published to PyPI.

[0.2.4.dev6]: https://github.com/synth-laboratories/synth-ai/releases/tag/v0.2.4.dev6


## Crafter Notes (migrated from crafter.md)

uvpm src.synth_env.examples.crafter_classic.agent_demos.test_crafter_react_agent --model gemini-1.5-flash-latest
uv run uvicorn src.synth_env.service.app:app --host 0.0.0.0 --port 8901
CRAFTER

gemini-1.5-flash-8b
collect_sapling: 20 times (K=0.1, contribution=0.304)

gpt-4.1-nano
collect_drink: 8 times (K=0.1, contribution=0.220)
collect_sapling: 2 times (K=0.1, contribution=0.110)
collect_wood: 12 times (K=1.0, contribution=2.565)

gpt-4o-mini
collect_drink: 1 times (K=0.1, contribution=0.069)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_wood: 7 times (K=1.0, contribution=2.079)
eat_cow: 2 times (K=1.0, contribution=1.099)

gemini-1.5-flash
collect_drink: 5 times (K=0.1, contribution=0.179)
collect_sapling: 10 times (K=0.1, contribution=0.240)
collect_wood: 12 times (K=1.0, contribution=2.565)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_wood_pickaxe: 1 times (K=3.0, contribution=2.079)
place_table: 1 times (K=3.0, contribution=2.079)

gpt-4.1-mini
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 16 times (K=0.1, contribution=0.283)
collect_stone: 1 times (K=1.0, contribution=0.693)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_wood_pickaxe: 1 times (K=3.0, contribution=2.079)
place_table: 1 times (K=3.0, contribution=2.079)

gemini-2.5-flash
collect_coal: 5 times (K=3.0, contribution=5.375)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 9 times (K=1.0, contribution=2.303)
collect_wood: 18 times (K=1.0, contribution=2.944)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 2 times (K=10.0, contribution=10.986)
make_wood_pickaxe: 13 times (K=3.0, contribution=7.917)
place_furnace: 2 times (K=10.0, contribution=10.986)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 17 times (K=3.0, contribution=8.671)
wake_up: 2 times (K=0.1, contribution=0.110)

gemini-2.5-pro
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 4 times (K=0.1, contribution=0.161)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 6 times (K=1.0, contribution=1.946)
collect_wood: 18 times (K=1.0, contribution=2.944)
make_stone_pickaxe: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 10 times (K=3.0, contribution=7.194)
place_furnace: 3 times (K=10.0, contribution=13.863)
place_table: 18 times (K=3.0, contribution=8.833)
wake_up: 3 times (K=0.1, contribution=0.139)

gpt-4.1
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 3 times (K=0.1, contribution=0.139)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_stone: 7 times (K=1.0, contribution=2.079)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_skeleton: 1 times (K=1.0, contribution=0.693)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_stone_pickaxe: 4 times (K=10.0, contribution=16.094)
make_wood_pickaxe: 14 times (K=3.0, contribution=8.124)
place_table: 17 times (K=3.0, contribution=8.671)
wake_up: 3 times (K=0.1, contribution=0.139)

claude-sonnet-4
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 2 times (K=0.1, contribution=0.110)
collect_sapling: 11 times (K=0.1, contribution=0.248)
collect_stone: 4 times (K=1.0, contribution=1.609)
collect_wood: 15 times (K=1.0, contribution=2.773)
eat_cow: 4 times (K=1.0, contribution=1.609)
make_wood_pickaxe: 8 times (K=3.0, contribution=6.592)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 13 times (K=3.0, contribution=7.917)
wake_up: 1 times (K=0.1, contribution=0.069)

gemini-2.5-flash-lite
collect_drink: 8 times (K=0.1, contribution=0.220)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_stone: 2 times (K=1.0, contribution=1.099)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_wood_pickaxe: 7 times (K=3.0, contribution=6.238)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 11 times (K=3.0, contribution=7.455)
wake_up: 6 times (K=0.1, contribution=0.195)

o4-mini
collect_coal: 7 times (K=3.0, contribution=6.238)
collect_drink: 5 times (K=0.1, contribution=0.179)
collect_iron: 1 times (K=10.0, contribution=6.931)
collect_sapling: 9 times (K=0.1, contribution=0.230)
collect_stone: 15 times (K=1.0, contribution=2.773)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 7 times (K=10.0, contribution=20.794)
make_stone_sword: 1 times (K=10.0, contribution=6.931)
make_wood_pickaxe: 19 times (K=3.0, contribution=8.987)
place_furnace: 5 times (K=10.0, contribution=17.918)
place_plant: 3 times (K=0.1, contribution=0.139)
place_table: 19 times (K=3.0, contribution=8.987)
wake_up: 3 times (K=0.1, contribution=0.139)

o3-mini
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 10 times (K=0.1, contribution=0.240)
collect_stone: 5 times (K=1.0, contribution=1.792)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 8 times (K=1.0, contribution=2.197)
make_stone_pickaxe: 1 times (K=10.0, contribution=6.931)
make_wood_pickaxe: 9 times (K=3.0, contribution=6.908)
place_table: 13 times (K=3.0, contribution=7.917)
wake_up: 11 times (K=0.1, contribution=0.248)

qwen/qwen3-32b
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 6 times (K=0.1, contribution=0.195)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 8 times (K=1.0, contribution=2.197)
collect_wood: 20 times (K=1.0, contribution=3.045)
eat_cow: 5 times (K=1.0, contribution=1.792)
make_stone_pickaxe: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 15 times (K=3.0, contribution=8.318)
place_furnace: 3 times (K=10.0, contribution=13.863)
place_plant: 2 times (K=0.1, contribution=0.110)
place_table: 18 times (K=3.0, contribution=8.833)
wake_up: 13 times (K=0.1, contribution=0.264)

o3
collect_coal: 6 times (K=3.0, contribution=5.838)
collect_drink: 1 times (K=0.1, contribution=0.069)
collect_iron: 2 times (K=10.0, contribution=10.986)
collect_sapling: 11 times (K=0.1, contribution=0.248)
collect_stone: 9 times (K=1.0, contribution=2.303)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 7 times (K=10.0, contribution=20.794)
make_stone_sword: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 14 times (K=3.0, contribution=8.124)
make_wood_sword: 6 times (K=3.0, contribution=5.838)
place_furnace: 4 times (K=10.0, contribution=16.094)
place_plant: 5 times (K=0.1, contribution=0.179)
place_table: 15 times (K=3.0, contribution=8.318)
wake_up: 12 times (K=0.1, contribution=0.256)

uv run python src/synth_env/examples/crafter_classic/agent_demos/test_crafter_react_agent.py --config src/evals/configs/crafter.toml

episodes = 20                     # Number of episodes to run
max_steps = 50                 # Maximum steps per episode
seed = 42                        # Random seed for reproducibility
difficulty = "easy"              # Difficulty mode


groq models
- meta-llama/llama-4-scout-17b-16e-instruct
- meta-llama/llama-4-maverick-17b-128e-instruct
qwen/qwen3-32b


CRAFTER
50 steps
| Model            | Episodes | Mean Score | Avg Achievements | Unique Achievements | Shaped Reward | Mean K-Score |
|------------------|----------|------------|------------------|---------------------|---------------|--------------|
| qwen-2.5-0.5b    |       10 |       1.00 |             1.00 |                   1 |         0.240 |        0.024 |
| g-1.5-flash-8b   |       20 |       1.00 |             1.00 |                   1 |         0.304 |        0.015 |
| L4-scout-17b     |       20 |       0.20 |             0.20 |                   4 |         1.525 |        0.076 |
| gpt-4.1-nano     |       20 |       1.10 |             1.10 |                   3 |         2.895 |        0.145 |
| gpt-4o-mini      |       20 |       1.25 |             1.25 |                   4 |         3.525 |        0.176 |
| L3.1-8b-groq     |       20 |       1.45 |             1.45 |                   4 |         3.552 |        0.178 |
| L4-maverick-17b  |       20 |       2.20 |             2.20 |                   6 |         7.087 |        0.354 |
| L3.3-70b-groq    |       20 |       2.15 |             2.15 |                   6 |         7.188 |        0.359 |
| gemini-1.5-flash |       20 |       1.55 |             1.55 |                   7 |         8.529 |        0.426 |
| deepseek-chat    |       20 |       1.85 |             1.85 |                   7 |         9.458 |        0.473 |
| gpt-4.1-mini     |       20 |       2.35 |             2.35 |                   8 |        11.699 |        0.585 |
| gpt-5-nano       |       20 |       2.85 |             ???? |                  13 |        ?????? |        ??????|
| groq/kimi-k2     |       20 |       3.05 |             3.05 |                   8 |        17.952 |        0.898 |
| g-2.5-flash-lite |       20 |       3.50 |             3.50 |                   9 |        19.829 |        0.991 |
| claude-sonnet-4  |       20 |       3.00 |             3.00 |                  10 |        23.077 |        1.154 |
| gpt-5-mini       |       20 |       3.85 |             ???? |                  15 |        ?????? |        ????? |
| o3-mini          |       20 |       4.20 |             4.20 |                  10 |        33.491 |        1.675 |
| gpt-4.1          |       20 |       4.40 |             4.40 |                  12 |        43.371 |        2.169 |
| gemini-2.5-flash |       19 |       4.68 |             4.68 |                  12 |        50.520 |        2.659 |
| gemini-2.5-pro   |       20 |       4.00 |             4.00 |                  10 |        53.358 |        2.668 |
| qwen/qwen3-32b   |       20 |       5.40 |             5.40 |                  15 |        56.894 |        2.845 |
| o4-mini          |       20 |       5.70 |             5.70 |                  14 |        83.936 |        4.197 |
| o3               |       20 |       5.80 |             5.80 |                  16 |        97.293 |        4.865 | 

*o3 had trajectories terminated early

300 steps
| gemini-1.5-flash |       20 |       1.50 |             1.50 |                   6 |         7.440 |        0.372 |
| g-2.5-flash-lite |       20 |       4.90 |             4.90 |                  10 |        24.713 |        1.236 |
| kimi-k2-instruct |       20 |       4.45 |             4.45 |                  12 |        45.834 |        2.292 |
| qwen/qwen3-32b   |       20 |       6.25 |             6.25 |                  14 |        55.396 |        2.770 |

50 steps, 100 traj
| qwen/qwen3-32b   |       93 |       4.74 |             4.74 |                  14 |        94.806 |        1.019 |uvpm src.synth_env.examples.crafter_classic.agent_demos.test_crafter_react_agent --model gemini-1.5-flash-latest
uv run uvicorn src.synth_env.service.app:app --host 0.0.0.0 --port 8901
CRAFTER

gemini-1.5-flash-8b
collect_sapling: 20 times (K=0.1, contribution=0.304)

gpt-4.1-nano
collect_drink: 8 times (K=0.1, contribution=0.220)
collect_sapling: 2 times (K=0.1, contribution=0.110)
collect_wood: 12 times (K=1.0, contribution=2.565)

gpt-4o-mini
collect_drink: 1 times (K=0.1, contribution=0.069)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_wood: 7 times (K=1.0, contribution=2.079)
eat_cow: 2 times (K=1.0, contribution=1.099)

gemini-1.5-flash
collect_drink: 5 times (K=0.1, contribution=0.179)
collect_sapling: 10 times (K=0.1, contribution=0.240)
collect_wood: 12 times (K=1.0, contribution=2.565)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_wood_pickaxe: 1 times (K=3.0, contribution=2.079)
place_table: 1 times (K=3.0, contribution=2.079)

gpt-4.1-mini
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 16 times (K=0.1, contribution=0.283)
collect_stone: 1 times (K=1.0, contribution=0.693)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_wood_pickaxe: 1 times (K=3.0, contribution=2.079)
place_table: 1 times (K=3.0, contribution=2.079)

gemini-2.5-flash
collect_coal: 5 times (K=3.0, contribution=5.375)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 9 times (K=1.0, contribution=2.303)
collect_wood: 18 times (K=1.0, contribution=2.944)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 2 times (K=10.0, contribution=10.986)
make_wood_pickaxe: 13 times (K=3.0, contribution=7.917)
place_furnace: 2 times (K=10.0, contribution=10.986)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 17 times (K=3.0, contribution=8.671)
wake_up: 2 times (K=0.1, contribution=0.110)

gemini-2.5-pro
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 4 times (K=0.1, contribution=0.161)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 6 times (K=1.0, contribution=1.946)
collect_wood: 18 times (K=1.0, contribution=2.944)
make_stone_pickaxe: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 10 times (K=3.0, contribution=7.194)
place_furnace: 3 times (K=10.0, contribution=13.863)
place_table: 18 times (K=3.0, contribution=8.833)
wake_up: 3 times (K=0.1, contribution=0.139)

gpt-4.1
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 3 times (K=0.1, contribution=0.139)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_stone: 7 times (K=1.0, contribution=2.079)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_skeleton: 1 times (K=1.0, contribution=0.693)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_stone_pickaxe: 4 times (K=10.0, contribution=16.094)
make_wood_pickaxe: 14 times (K=3.0, contribution=8.124)
place_table: 17 times (K=3.0, contribution=8.671)
wake_up: 3 times (K=0.1, contribution=0.139)

claude-sonnet-4
collect_coal: 1 times (K=3.0, contribution=2.079)
collect_drink: 2 times (K=0.1, contribution=0.110)
collect_sapling: 11 times (K=0.1, contribution=0.248)
collect_stone: 4 times (K=1.0, contribution=1.609)
collect_wood: 15 times (K=1.0, contribution=2.773)
eat_cow: 4 times (K=1.0, contribution=1.609)
make_wood_pickaxe: 8 times (K=3.0, contribution=6.592)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 13 times (K=3.0, contribution=7.917)
wake_up: 1 times (K=0.1, contribution=0.069)

gemini-2.5-flash-lite
collect_drink: 8 times (K=0.1, contribution=0.220)
collect_sapling: 15 times (K=0.1, contribution=0.277)
collect_stone: 2 times (K=1.0, contribution=1.099)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 3 times (K=1.0, contribution=1.386)
make_wood_pickaxe: 7 times (K=3.0, contribution=6.238)
place_plant: 1 times (K=0.1, contribution=0.069)
place_table: 11 times (K=3.0, contribution=7.455)
wake_up: 6 times (K=0.1, contribution=0.195)

o4-mini
collect_coal: 7 times (K=3.0, contribution=6.238)
collect_drink: 5 times (K=0.1, contribution=0.179)
collect_iron: 1 times (K=10.0, contribution=6.931)
collect_sapling: 9 times (K=0.1, contribution=0.230)
collect_stone: 15 times (K=1.0, contribution=2.773)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 7 times (K=10.0, contribution=20.794)
make_stone_sword: 1 times (K=10.0, contribution=6.931)
make_wood_pickaxe: 19 times (K=3.0, contribution=8.987)
place_furnace: 5 times (K=10.0, contribution=17.918)
place_plant: 3 times (K=0.1, contribution=0.139)
place_table: 19 times (K=3.0, contribution=8.987)
wake_up: 3 times (K=0.1, contribution=0.139)

o3-mini
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 7 times (K=0.1, contribution=0.208)
collect_sapling: 10 times (K=0.1, contribution=0.240)
collect_stone: 5 times (K=1.0, contribution=1.792)
collect_wood: 17 times (K=1.0, contribution=2.890)
eat_cow: 8 times (K=1.0, contribution=2.197)
make_stone_pickaxe: 1 times (K=10.0, contribution=6.931)
make_wood_pickaxe: 9 times (K=3.0, contribution=6.908)
place_table: 13 times (K=3.0, contribution=7.917)
wake_up: 11 times (K=0.1, contribution=0.248)

qwen/qwen3-32b
collect_coal: 3 times (K=3.0, contribution=4.159)
collect_drink: 6 times (K=0.1, contribution=0.195)
collect_sapling: 12 times (K=0.1, contribution=0.256)
collect_stone: 8 times (K=1.0, contribution=2.197)
collect_wood: 20 times (K=1.0, contribution=3.045)
eat_cow: 5 times (K=1.0, contribution=1.792)
make_stone_pickaxe: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 15 times (K=3.0, contribution=8.318)
place_furnace: 3 times (K=10.0, contribution=13.863)
place_plant: 2 times (K=0.1, contribution=0.110)
place_table: 18 times (K=3.0, contribution=8.833)
wake_up: 13 times (K=0.1, contribution=0.264)

o3
collect_coal: 6 times (K=3.0, contribution=5.838)
collect_drink: 1 times (K=0.1, contribution=0.069)
collect_iron: 2 times (K=10.0, contribution=10.986)
collect_sapling: 11 times (K=0.1, contribution=0.248)
collect_stone: 9 times (K=1.0, contribution=2.303)
collect_wood: 19 times (K=1.0, contribution=2.996)
defeat_zombie: 1 times (K=1.0, contribution=0.693)
eat_cow: 1 times (K=1.0, contribution=0.693)
make_stone_pickaxe: 7 times (K=10.0, contribution=20.794)
make_stone_sword: 3 times (K=10.0, contribution=13.863)
make_wood_pickaxe: 14 times (K=3.0, contribution=8.124)
make_wood_sword: 6 times (K=3.0, contribution=5.838)
place_furnace: 4 times (K=10.0, contribution=16.094)
place_plant: 5 times (K=0.1, contribution=0.179)
place_table: 15 times (K=3.0, contribution=8.318)
wake_up: 12 times (K=0.1, contribution=0.256)

uv run python src/synth_env/examples/crafter_classic/agent_demos/test_crafter_react_agent.py --config src/evals/configs/crafter.toml

episodes = 20                     # Number of episodes to run
max_steps = 50                 # Maximum steps per episode
seed = 42                        # Random seed for reproducibility
difficulty = "easy"              # Difficulty mode


groq models
- meta-llama/llama-4-scout-17b-16e-instruct
- meta-llama/llama-4-maverick-17b-128e-instruct
qwen/qwen3-32b


CRAFTER
50 steps
| Model            | Episodes | Mean Score | Avg Achievements | Unique Achievements | Shaped Reward | Mean K-Score |
|------------------|----------|------------|------------------|---------------------|---------------|--------------|
| qwen-2.5-0.5b    |       10 |       1.00 |             1.00 |                   1 |         0.240 |        0.024 |
| g-1.5-flash-8b   |       20 |       1.00 |             1.00 |                   1 |         0.304 |        0.015 |
| L4-scout-17b     |       20 |       0.20 |             0.20 |                   4 |         1.525 |        0.076 |
| gpt-4.1-nano     |       20 |       1.10 |             1.10 |                   3 |         2.895 |        0.145 |
| gpt-4o-mini      |       20 |       1.25 |             1.25 |                   4 |         3.525 |        0.176 |
| L3.1-8b-groq     |       20 |       1.45 |             1.45 |                   4 |         3.552 |        0.178 |
| L4-maverick-17b  |       20 |       2.20 |             2.20 |                   6 |         7.087 |        0.354 |
| L3.3-70b-groq    |       20 |       2.15 |             2.15 |                   6 |         7.188 |        0.359 |
| gemini-1.5-flash |       20 |       1.55 |             1.55 |                   7 |         8.529 |        0.426 |
| deepseek-chat    |       20 |       1.85 |             1.85 |                   7 |         9.458 |        0.473 |
| gpt-4.1-mini     |       20 |       2.35 |             2.35 |                   8 |        11.699 |        0.585 |
| groq/kimi-k2     |       20 |       3.05 |             3.05 |                   8 |        17.952 |        0.898 |
| g-2.5-flash-lite |       20 |       3.50 |             3.50 |                   9 |        19.829 |        0.991 |
| claude-sonnet-4  |       20 |       3.00 |             3.00 |                  10 |        23.077 |        1.154 |
| o3-mini          |       20 |       4.20 |             4.20 |                  10 |        33.491 |        1.675 |
| gpt-4.1          |       20 |       4.40 |             4.40 |                  12 |        43.371 |        2.169 |
| gemini-2.5-flash |       19 |       4.68 |             4.68 |                  12 |        50.520 |        2.659 |
| gemini-2.5-pro   |       20 |       4.00 |             4.00 |                  10 |        53.358 |        2.668 |
| qwen/qwen3-32b   |       20 |       5.40 |             5.40 |                  12 |        56.894 |        2.845 |
| o4-mini          |       20 |       5.70 |             5.70 |                  14 |        83.936 |        4.197 |
| o3               |       20 |       5.80 |             5.80 |                  16 |        97.293 |        4.865 | 

*o3 had trajectories terminated early

300 steps
| gemini-1.5-flash |       20 |       1.50 |             1.50 |                   6 |         7.440 |        0.372 |
| g-2.5-flash-lite |       20 |       4.90 |             4.90 |                  10 |        24.713 |        1.236 |
| kimi-k2-instruct |       20 |       4.45 |             4.45 |                  12 |        45.834 |        2.292 |
| qwen/qwen3-32b   |       20 |       6.25 |             6.25 |                  14 |        55.396 |        2.770 |

50 steps, 100 traj
| qwen/qwen3-32b   |       93 |       4.74 |             4.74 |                  14 |        94.806 |        1.019 |