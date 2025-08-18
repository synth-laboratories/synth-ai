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
