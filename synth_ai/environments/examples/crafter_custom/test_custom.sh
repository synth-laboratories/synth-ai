#!/bin/bash
# Example script to test custom Crafter environments

echo "Testing Custom Crafter Environments"
echo "=================================="

# Test peaceful world
echo -e "\n1. Testing PEACEFUL world (no enemies):"
python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --model gpt-4.1-nano \
    --world-config peaceful \
    --episodes 1 \
    --max-turns 10 \
    --evaluate-traces

# Test hard world
echo -e "\n2. Testing HARD world (many enemies, few resources):"
python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --model gpt-4.1-nano \
    --world-config hard \
    --episodes 1 \
    --max-turns 10 \
    --evaluate-traces

# Test custom config
echo -e "\n3. Testing CUSTOM config (abundant resources):"
python -m synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent \
    --model gpt-4.1-nano \
    --world-config-path agent_demos/world_configs/abundant_resources.json \
    --episodes 1 \
    --max-turns 10 \
    --evaluate-traces