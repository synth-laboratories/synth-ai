# Synth AI

Modern Compound AI System Development

**Comprehensive AI Framework for Language Models, Environments, and Observability**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.1.dev0-orange)](https://pypi.org/project/synth-ai/)
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

Start the Synth AI service daemon (includes Turso database + environment service):

```bash
# Start both database daemon (port 8080) and environment service (port 8901)
uvx synth-ai serve
```

In another terminal, run your first example:

```bash
# Run a Crafter agent demo with Gemini
./examples/run_crafter_demo.sh
```

This will:
- Start the Turso database daemon with 2-second sync intervals
- Launch the environment service API on port 8901
- Run a reactive agent in the Crafter environment using Gemini 1.5 Flash