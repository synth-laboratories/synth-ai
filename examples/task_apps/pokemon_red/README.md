# Pokémon Red Task App

A reinforcement learning environment for Pokémon Red using PyBoy emulation with VLM support.

## Features

- **Full Game Boy Emulation**: Uses PyBoy to run authentic Pokémon Red ROM
- **VLM Support**: Base64-encoded PNG frames for vision models (GPT-4V, Qwen-VL, etc.)
- **Policy Proxy**: OpenAI/Groq API integration for LLM-driven gameplay
- **Rich State Extraction**: Comprehensive game state from RAM (HP, position, party, battle data)
- **Reward Shaping**: Ultra-dense reward functions for RL training
- **Instant Start**: Pre-configured init state skips intro (starts in Red's bedroom)

## Quick Start

### 1. Start the Task App Server

```bash
# From synth-ai root
uv run -m synth_ai task-app serve pokemon_red --port 8913
```

### 2. Run a Random Rollout

```python
import httpx
import asyncio

async def test_rollout():
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://127.0.0.1:8913/rollout",
            json={
                "ops": [
                    {"button": "DOWN", "frames": 10},
                    {"button": "A", "frames": 20},
                    {"button": "RIGHT", "frames": 15},
                ],
                "policy": {"config": {}},
            },
        )
        result = response.json()
        print(f"Steps: {len(result['steps'])}")

asyncio.run(test_rollout())
```

### 3. Run with VLM Policy

```bash
# Using Qwen-VL via Groq
uv run python examples/task_apps/pokemon_red/test_pallet_town_rewards.py
```

## Reward Functions

### Pallet Town Progression (Recommended for Beginners)

**Location**: `synth_ai/environments/examples/red/engine_helpers/reward_library/pallet_town_progression.py`

Ultra-rich reward shaping for the opening sequence:

| Milestone | Reward | Description |
|-----------|--------|-------------|
| Leave bedroom | +20 | Go downstairs |
| Exit house | +30 | Enter Pallet Town |
| Find Oak's lab | +40 | Discover and enter lab |
| Talk to Oak | +50 | First dialogue |
| Get starter | +100 | Receive your first Pokémon |
| Enter battle | +75 | Start rival battle |
| Deal damage | +50 | Attack rival (10×5) |
| Half HP | +25 | Reduce enemy to <50% HP |
| Low HP | +35 | Reduce enemy to <25% HP |
| Win battle | +150 | Defeat rival |
| Exit lab | +60 | Leave with Pokémon |
| **Efficiency bonuses** | +100 | Fast navigation, healthy Pokémon |

**Total: ~600-700 points**

See [`PALLET_TOWN_REWARDS.md`](../../../synth_ai/environments/examples/red/engine_helpers/reward_library/PALLET_TOWN_REWARDS.md) for full documentation.

### Usage in Training

```toml
# pallet_town_rl_config.toml
[reward]
reward_type = "composite"
reward_class = "synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression.PalletTownProgressionCompositeReward"

[training]
algorithm = "ppo"
max_steps_per_episode = 500
num_episodes = 1000
```

## State Schema

The environment exposes comprehensive game state:

```python
{
    # Position
    "map_id": int,              # Current location
    "player_x": int,
    "player_y": int,
    
    # Party
    "party_count": int,
    "party_pokemon": [
        {
            "species_id": int,
            "level": int,
            "hp_current": int,
            "hp_max": int,
            "hp_percentage": float,
            "xp": int,
        }
    ],
    
    # Battle
    "in_battle": bool,
    "battle_outcome": int,      # 0=ongoing, 1=win, 2=lose
    "enemy_hp_current": int,
    "enemy_hp_max": int,
    "enemy_hp_percentage": float,
    "enemy_level": int,
    "enemy_species_id": int,
    "battle_turn": int,
    
    # Dialogue & UI
    "text_box_active": bool,
    "menu_state": int,
    
    # Progress
    "badges": int,              # Bitfield of earned badges
    "money": int,
    
    # VLM Support
    "observation_image_base64": str,  # PNG frame for vision models
}
```

## Action Space

### Button Actions

```python
{
    "button": "A" | "B" | "START" | "SELECT" | "UP" | "DOWN" | "LEFT" | "RIGHT",
    "frames": int,  # How long to hold the button (60fps)
}
```

### Policy-Driven Actions

When using LLM policies, the task app proxies requests to OpenAI/Groq:

```python
{
    "policy": {
        "config": {
            "model": "gpt-4-turbo",
            "api_key": "...",
            # or for Groq:
            # "model": "qwen-2.5-7b",
            # "base_url": "https://api.groq.com/v1",
        }
    }
}
```

## Files

- **`task_app.py`**: Main task app entry point
- **`pallet_town_rl_config.toml`**: Training config for Pallet Town sequence
- **`test_pallet_town_rewards.py`**: Reward function test/demo script
- **`create_red_init_state.py`** (repo root): Script to generate init state
- **`Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb`**: Your ROM (not committed)

## Creating Init States

The default init state starts in Red's bedroom with intro skipped. To create custom states:

```python
# See /Users/joshpurtell/Documents/GitHub/synth-ai/create_red_init_state.py
from pyboy import PyBoy

emulator = PyBoy("path/to/rom.gb", window="null")

# Navigate to desired starting point
# ... (button presses)

# Save state
with open("custom_init.state", "wb") as f:
    emulator.save_state(f)
```

## Memory Addresses

Key RAM addresses are defined in `synth_ai/environments/examples/red/engine_helpers/memory_map.py`:

- `MAP_ID = 0xD35E`
- `PLAYER_X/Y = 0xD362/0xD361`
- `IN_BATTLE_FLAG = 0xD057`
- `ENEMY_HP_CURRENT = 0xCFE6`
- `PARTY_COUNT = 0xD163`
- `BADGE_FLAGS = 0xD356`
- (and many more)

## Troubleshooting

### ROM Not Found

```bash
# Set environment variable
export POKEMON_RED_ROM_PATH="/path/to/pokemon_red.gb"

# Or copy ROM to expected location
cp "Pokemon - Red Version.gb" synth_ai/environments/examples/red/roms/pokemon_red.gb
```

### PyBoy Not Installed

```bash
uv add pyboy
```

### Server Won't Start (Port in Use)

```bash
# Kill existing server
lsof -ti :8913 | xargs -r kill -9

# Or use a different port
uv run -m synth_ai task-app serve pokemon_red --port 8914
```

## Examples

### Test Script (Random Actions)

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python test_pokemon_red_rollout.py
```

### Reward Function Demo

```bash
uv run python examples/task_apps/pokemon_red/test_pallet_town_rewards.py
```

Output:
```
======================================================================
PALLET TOWN PROGRESSION - REWARD SIMULATION
======================================================================

✓ Leave bedroom (Map 1→2):                    +20 points
✓ Exit house to Pallet Town (Map 2→0):        +30 points
✓ Find and enter Oak's Lab (Map 0→3):         +40 points
...
======================================================================
TOTAL REWARD: 705 points
======================================================================
```

## Future Work

- [ ] Route 1 exploration rewards
- [ ] Wild Pokémon encounter rewards
- [ ] Capture mechanics rewards
- [ ] Gym battle rewards
- [ ] Badge collection rewards
- [ ] Multi-environment curriculum (Pallet → Viridian → Pewter)

## Credits

- **PyBoy**: Game Boy emulator - https://github.com/Baekalfen/PyBoy
- **Pokémon Red Disassembly**: RAM map reference - https://github.com/pret/pokered
- **Datacrystal.org**: Memory address documentation

