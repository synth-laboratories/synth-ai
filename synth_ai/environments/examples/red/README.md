# Pokemon Red Environment

A stateful Game Boy Pokemon Red environment for AI agents, implementing dense reward tracking and memory state extraction similar to the Gemini Plays Pokemon benchmark.

## Setup

### 1. Install Dependencies

```bash
pip install pyboy numpy
```

### 2. Obtain Pokemon Red ROM

You need a Pokemon Red ROM file. Place it in one of these locations:

- `src/examples/red/roms/pokemon_red.gb` (recommended)
- `src/examples/red/roms/PokemonRed.gb`  
- `src/examples/red/vendor/pokemon_red.gb`
- `~/Games/pokemon_red.gb`

**Note**: You must legally own Pokemon Red to use the ROM file. We cannot provide the ROM due to copyright restrictions.

### 3. Optional: Save States

Place initial save states in `src/examples/red/snapshots/`:
- `pewter_start.state` - Starting position near Pewter Gym

## Features

### Dense Reward Components
- **Badge Rewards** (+1.0) - Earning gym badges
- **Map Transitions** (+0.1) - Moving between areas  
- **Battle Victories** (+0.5) - Winning Pokemon battles
- **Level Ups** (+0.3) - Pokemon gaining levels
- **XP Gains** (+0.001 per XP) - Experience point accumulation
- **Step Penalty** (-0.001) - Encourages efficiency

### Memory State Tracking
- Player position (map ID, X/Y coordinates)
- Badge collection (bitfield tracking)
- Battle states and outcomes
- Pokemon stats (HP, level, XP)
- Inventory and items
- Menu states

### Tools Available
- `press_button` - Press Game Boy buttons (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT)

## Usage

```python
from examples.red.environment import PokemonRedEnvironment
from examples.red.taskset import INSTANCE as POKEMON_TASK
from environment.tools import EnvToolCall

# Initialize environment
env = PokemonRedEnvironment(POKEMON_TASK)
obs = await env.initialize()

# Press buttons
button_call = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
obs = await env.step(button_call)

print(f"Position: {obs['position']}")
print(f"Badges: {obs['badges_earned']}")
print(f"HP: {obs['hp_status']}")
```

## Testing

Run the test suite:

```bash
pytest src/examples/red/agent_demos/test_synth_react.py -v
pytest src/examples/red/units/ -v
```

## Architecture

- `engine.py` - Core PyBoy integration with reward tracking
- `environment.py` - StatefulEnvironment wrapper with tool interface
- `taskset.py` - Task definitions (default: beat Brock)
- `engine_helpers/` - Memory extraction and reward components
- `agent_demos/` - Example agent interactions
- `units/` - Comprehensive unit tests

## Memory Addresses

Key Pokemon Red memory locations tracked:

- `0xD356` - Badge flags (bitfield)
- `0xD35E` - Current map ID  
- `0xD361/0xD362` - Player Y/X position
- `0xD16C` - Current HP of first Pokemon
- `0xD18C` - Level of first Pokemon
- `0xD179` - XP of first Pokemon (3 bytes)

Based on the Gemini Plays Pokemon benchmark research and PokemonRedExperiments by Paul Whidden.