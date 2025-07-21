# Synth AI

Modern Compound AI System Development

**Comprehensive AI Framework for Language Models, Environments, and Observability**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.1.9-orange)](https://pypi.org/project/synth-ai/)
![Coverage](https://img.shields.io/badge/coverage-11.3%25-red)
![Tests](https://img.shields.io/badge/tests-17/17 passing-brightgreen)
A unified framework combining language model capabilities, synthetic environments, and comprehensive tracing for building and evaluating AI agents.

## 🎯 Key Features

- **🤖 Language Models** - Unified LM interface for OpenAI, Anthropic, Gemini, Groq, and more
- **🏗️ Synthetic Environments** - Comprehensive framework for agent training and evaluation  
- **📊 Observability & Tracing** - Built-in monitoring, logging, and performance tracking
- **🔌 Provider Support** - Enhanced client wrappers with automatic tracing integration
- **🌐 RESTful APIs** - HTTP access for remote training and evaluation
- **🛠️ Agent Tools** - Simple abstractions for agent-environment interaction

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

### Basic Usage

```python
from synth_ai import LM

# Create language model
lm = LM("gpt-4o-mini", temperature=0.7)

# Generate structured output
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

result = lm("What is the capital of France?", response_format=Response)
print(result.structured_output.answer)  # "Paris"
```

### Environment Usage

```python
from synth_ai.environments.examples.tictactoe.environment import TicTacToeEnvironment

# Create environment
env = TicTacToeEnvironment()

# Run agent
state = env.reset()
while not env.done:
    action = agent.act(state)
    state = env.step(action)
```

### Tracing & Observability

```python
from synth_ai.tracing import trace_event_sync

@trace_event_sync
def my_agent_step(observation):
    # Your agent logic here
    return action

# Automatic tracing of execution
```

## 🎮 Supported Environments

| Environment | Status | Description |
|-------------|---------|-------------|
| **TicTacToe** | ✅ Stable | Simple strategic game for testing |
| **Sokoban** | ✅ Stable | Classic puzzle game for planning |
| **Hendryks Math** | ✅ Stable | Mathematical reasoning tasks |
| **Crafter** | 🔄 Research | Minecraft-like survival environment |
| **NetHack** | 🔄 Research | Complex dungeon exploration |
| **MiniGrid** | 🔄 Research | Grid world navigation |

## 🤖 Supported LM Providers

| Provider | Status | Features |
|----------|--------|----------|
| **OpenAI** | ✅ Full | GPT-4, structured outputs, tools |
| **Anthropic** | ✅ Full | Claude models, structured outputs |
| **Google** | ✅ Full | Gemini models, structured outputs |
| **Groq** | ✅ Full | Fast inference, structured outputs |
| **Together** | ✅ Full | Open source models |
| **DeepSeek** | ✅ Full | Code and reasoning models |

## 📖 Documentation

- **[API Reference](docs/api.md)** - Complete API documentation
- **[Environment Guide](docs/environments.md)** - Detailed environment descriptions
- **[LM Provider Guide](docs/providers.md)** - Language model configuration
- **[Tracing Guide](docs/tracing.md)** - Observability and monitoring

## 🔧 Development

### Health Check
```bash
# Check codebase health
python scripts/check_health.py
```

### Testing
```bash
# Fast tests (~3 seconds)
uv run pytest tests/ -x --tb=short

# With research environments
uv run --extra research pytest tests/ 

# Full test suite with coverage
uv run pytest tests/ --cov=synth_ai --cov-report=html
```

### Code Quality
```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
uvx ty check

# Run all checks
uvx ty check && ruff check . && ruff format --check .
```

### Performance Profiling
```bash
# Profile LM inference
uv run python -m synth_ai.lm.core.profiler

# Profile environment execution  
uv run python -m synth_ai.environments.profiler
```

## 📊 Test Coverage & Metrics

Run comprehensive tests and generate metrics:

```bash
# Generate test coverage report
uv run pytest tests/ --cov=synth_ai --cov-report=html --cov-report=term

# Update README metrics
python dev/update_readme_metrics.py

# Fast metric update (unit tests only)
python dev/update_readme_metrics.py --fast
```

## 🏗️ Architecture

### Monorepo Structure

```
synth_ai/
├── lm/                    # Language model core (formerly zyk)
│   ├── core/             # Main LM interface and clients
│   ├── vendors/          # Provider-specific implementations  
│   ├── provider_support/ # Enhanced client wrappers
│   └── structured_outputs/ # Structured generation
├── environments/         # Environment framework (formerly synth_env)
│   ├── examples/         # Built-in environments
│   ├── service/          # REST API service
│   └── stateful/         # State management
├── tracing/              # Observability (formerly synth_sdk)
│   ├── decorators/       # Function tracing
│   ├── events/           # Event management
│   └── upload/           # Remote logging
└── zyk/                  # Deprecated compatibility layer
```

### Migration Guide

The package consolidates functionality from multiple repos:

```python
# Old imports (deprecated, but still work)
from synth_ai.zyk import LM  # ⚠️ Shows deprecation warning
from synth_env import tasks  # ⚠️ Shows deprecation warning

# New imports (recommended)
from synth_ai import LM, environments
from synth_ai.environments import tasks
from synth_ai.lm.core.main import LM  # Direct import
```

## 🚢 Publishing & Releases

### Build Package
```bash
# Build for PyPI
uv build

# Check package contents
tar -tzf dist/synth-ai-*.tar.gz | head -20
```

### Publish
```bash
# Test on TestPyPI
uv publish --index-url https://test.pypi.org/legacy/

# Publish to PyPI
uv publish
```

## 🤝 Contributing

We welcome contributions! Please see our **[Contributing Guide](dev/contributing.md)** for:
- Development setup with `uv`
- Code style guidelines (`ruff`, `uvx ty`)
- Testing requirements
- Pull request process

### Pre-Commit Checklist
```bash
# Run all checks before committing
uvx ty check && \
ruff check . && \
ruff format --check . && \
uv run pytest tests/ -x --tb=short
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the teams at OpenAI, Anthropic, Google, and other contributors to the AI ecosystem that made this framework possible.

---

**⚠️ Development Status**: This is an active consolidation of multiple AI frameworks. While core LM and tracing functionality is production-ready, some environments may have breaking changes.

## 📈 Recent Updates

- **v0.1.9**: Consolidated monorepo with LM, environments, and tracing
- **Migration**: Moved from `synth_ai.zyk` → `synth_ai.lm` with backward compatibility
- **Integration**: Combined synth-sdk tracing and synth-env environments
- **Dependencies**: Added optional `[research]` extras for heavy game environments