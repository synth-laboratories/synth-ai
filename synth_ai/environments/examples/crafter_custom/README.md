# Custom Crafter Environments

Test Crafter agents with configurable world generation parameters.

## Quick Start

```bash
# Run with easy world configuration
python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --model gpt-4.1-nano \
    --world-config easy \
    --episodes 3

# Run with custom world config
python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --world-config-path world_configs/abundant_resources.json \
    --evaluate-traces
```

## Dataset

The `dataset/instances.json` file contains pre-configured task instances with different world configurations:
- Peaceful worlds (no enemies)
- Easy worlds (more resources, fewer enemies)
- Normal worlds (standard Crafter)
- Hard worlds (scarce resources, many enemies)
- Custom configurations

The run script filters instances based on your selected configuration.

## Command Line Options

- `--model`: Model name (e.g., gpt-4.1-nano)
- `--episodes`: Number of episodes to run
- `--max-turns`: Maximum turns per episode
- `--difficulty`: Filter by difficulty (easy, medium, hard)
- `--world-config`: Filter by world config (easy, normal, hard, peaceful)
- `--world-config-path`: Path to custom world config JSON
- `--evaluate-traces`: Run trace evaluation after episodes

## World Configurations

- **peaceful**: No enemies, abundant resources
- **easy**: More resources, fewer enemies  
- **normal**: Standard Crafter experience
- **hard**: Scarce resources, many enemies

## Directory Structure

```
crafter_custom/
├── README.md
├── dataset/
│   └── instances.json              # Pre-configured task instances
└── agent_demos/
    ├── test_crafter_custom_agent.py
    ├── trace_eval.py
    ├── world_configs/
    │   ├── abundant_resources.json
    │   └── survival_challenge.json
    └── traces/
```