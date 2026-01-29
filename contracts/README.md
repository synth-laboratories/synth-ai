# Synth AI Contracts

**Polyglot API contracts for implementing Synth services in any language.**

This directory contains OpenAPI specifications that define the HTTP interfaces
for Synth's various services. Implement these contracts in your language of
choice to integrate with Synth.

## Available Contracts

### Task App Contract (`task_app.yaml`)

The core contract for prompt optimization. Implements:
- `GET /health` - Health check
- `POST /rollout` - Evaluate a prompt (returns reward)
- `GET /info` - Task metadata (optional)

Use this to build Task Apps for MIPRO and GEPA prompt optimization in any
language (Python, TypeScript, Go, Rust, etc.).

## Usage

### Python
```python
from synth_ai.contracts import get_task_app_contract

# Get the contract YAML
yaml_content = get_task_app_contract()

# Or get the file path
from synth_ai.contracts import TASK_APP_CONTRACT_PATH
print(f"Contract at: {TASK_APP_CONTRACT_PATH}")
```

### CLI
```bash
# Show the contract
synth contracts show task-app
```

### Other Languages

Download the contract file and use OpenAPI codegen tools:
```bash
# TypeScript (using openapi-generator)
npx @openapitools/openapi-generator-cli generate \
  -i https://raw.githubusercontent.com/SynthAILabs/synth-ai/main/contracts/task_app.yaml \
  -g typescript-fetch \
  -o ./generated

# Go
openapi-generator generate \
  -i task_app.yaml \
  -g go-server \
  -o ./generated
```

## Future Contracts (Planned)

- `sft.yaml` - Supervised Fine-Tuning data submission
- `rl.yaml` - Reinforcement Learning environment interface
- `verifier.yaml` - LLM verifier evaluation API

## Design Principles

1. **Language-agnostic**: OpenAPI specs work with any language
2. **Explicit schemas**: Every request/response is fully typed
3. **Rich documentation**: Inline docs explain implementation details
4. **Versioned**: Contracts are versioned for stability
# Synth AI Contracts

**Polyglot API contracts for implementing Synth services in any language.**

This directory contains OpenAPI specifications that define the HTTP interfaces
for Synth's various services. Implement these contracts in your language of
choice to integrate with Synth.

## Available Contracts

### Task App Contract (`task_app.yaml`)

The core contract for prompt optimization. Implements:
- `GET /health` - Health check
- `POST /rollout` - Evaluate a prompt (returns reward)
- `GET /info` - Task metadata (optional)

Use this to build Task Apps for MIPRO and GEPA prompt optimization in any
language (Python, TypeScript, Go, Rust, etc.).

## Usage

### Python
```python
from pathlib import Path

yaml_content = Path("contracts/task_app.yaml").read_text()
print(yaml_content[:200])
```

### CLI
```bash
# Show the contract
synth contracts show task-app
```

### Other Languages

Download the contract file and use OpenAPI codegen tools:
```bash
# TypeScript (using openapi-generator)
npx @openapitools/openapi-generator-cli generate \
  -i https://raw.githubusercontent.com/SynthAILabs/synth-ai/main/contracts/task_app.yaml \
  -g typescript-fetch \
  -o ./generated

# Go
openapi-generator generate \
  -i task_app.yaml \
  -g go-server \
  -o ./generated
```

## Future Contracts (Planned)

- `sft.yaml` - Supervised Fine-Tuning data submission
- `rl.yaml` - Reinforcement Learning environment interface
- `verifier.yaml` - LLM verifier evaluation API

## Design Principles

1. **Language-agnostic**: OpenAPI specs work with any language
2. **Explicit schemas**: Every request/response is fully typed
3. **Rich documentation**: Inline docs explain implementation details
4. **Versioned**: Contracts are versioned for stability
