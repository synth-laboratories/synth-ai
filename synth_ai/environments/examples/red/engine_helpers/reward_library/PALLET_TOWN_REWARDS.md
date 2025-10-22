# Pallet Town Progression Rewards

## Overview

The **Pallet Town Progression** reward function provides ultra-rich, fine-grained reward shaping for the opening sequence of Pokémon Red. This reward function is specifically designed to provide dense feedback for RL agents learning to navigate the game's initial tutorial sequence.

## Total Possible Reward: **~600+ points**

## Reward Breakdown

### Navigation Milestones

| Component | Reward | Description |
|-----------|--------|-------------|
| `LeaveBedroomReward` | **+20** | Going downstairs from bedroom to main floor (map 1→2) |
| `ExitHouseFirstTimeReward` | **+30** | Leaving starting house and entering Pallet Town (map 2→0) |
| `FindOakLabReward` | **+40** | Discovering and entering Oak's Lab (map 0→3) |
| `ExitLabAfterBattleReward` | **+60** | Leaving Oak's Lab with a Pokemon (completes sequence) |

**Subtotal: 150 points**

---

### Story Progression

| Component | Reward | Description |
|-----------|--------|-------------|
| `TalkToOakReward` | **+50** | First conversation with Professor Oak in the lab |
| `ReceiveStarterPokemonReward` | **+100** | Receiving your first Pokemon from Oak (party 0→1) |

**Subtotal: 150 points**

---

### Battle Sequence

| Component | Reward | Description |
|-----------|--------|-------------|
| `EnterFirstBattleReward` | **+75** | Entering the first rival battle in Oak's lab |
| `DealDamageToRivalReward` | **+50** | Dealing damage to rival's Pokemon (10 instances × +5) |
| `ReduceEnemyHPByHalfReward` | **+25** | Reducing enemy HP below 50% |
| `ReduceEnemyHPToLowReward` | **+35** | Reducing enemy HP below 25% (critical) |
| `WinFirstBattleReward` | **+150** | Winning the first battle against the rival |

**Subtotal: 335 points**

---

### Efficiency Bonuses

| Component | Reward | Description |
|-----------|--------|-------------|
| `FirstBattleEfficiencyReward` | **+20** | Winning battle in ≤5 turns (+10 if ≤8 turns) |
| `KeepPokemonHealthyReward` | **+30** | Keeping Pokemon HP >50% during first battle |
| `NavigationSpeedReward` | **+50** | Completing sequence in ≤40 steps (+30 if ≤60, +15 if ≤80) |

**Subtotal: 100 points**

---

## State Requirements

The reward function requires the following state fields to be tracked:

### Position & Navigation
- `map_id` - Current map/location ID
- `player_x`, `player_y` - Player coordinates
- `prev_map_id` - Previous map ID (for transitions)

### Battle State
- `in_battle` - Boolean indicating if in battle
- `battle_outcome` - 0=ongoing, 1=win, 2=lose
- `enemy_hp_current` - Enemy Pokemon's current HP
- `enemy_hp_max` - Enemy Pokemon's max HP
- `enemy_hp_percentage` - Enemy HP as percentage
- `battle_turn` - Current turn number in battle
- `prev_in_battle` - Previous battle state
- `prev_enemy_hp_current` - Previous enemy HP
- `prev_enemy_hp_percentage` - Previous enemy HP percentage

### Party State
- `party_count` - Number of Pokemon in party
- `party_pokemon` - List of party Pokemon with HP data
- `prev_party_count` - Previous party count

### Dialogue State
- `text_box_active` - Boolean indicating if text box is shown
- `prev_text_box_active` - Previous text box state

## Usage

### Basic Usage

```python
from synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression import (
    PalletTownProgressionCompositeReward
)

# Create reward component
reward_fn = PalletTownProgressionCompositeReward()

# Use in step function
reward = await reward_fn.score(state=current_state, action=action_dict)
```

### Configuration File

See `examples/task_apps/pokemon_red/pallet_town_rl_config.toml` for a complete RL training configuration using this reward function.

```toml
[reward]
reward_type = "composite"
reward_class = "synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression.PalletTownProgressionCompositeReward"
```

### Individual Components

You can also use individual reward components for more specific objectives:

```python
from synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression import (
    LeaveBedroomReward,
    FindOakLabReward,
    WinFirstBattleReward,
)

# Mix and match components
custom_reward = CompositeReward([
    LeaveBedroomReward(),
    FindOakLabReward(),
    WinFirstBattleReward(),
])
```

## Design Philosophy

### Dense Reward Shaping
Rather than only rewarding the final goal (winning the battle and exiting the lab), this reward function provides intermediate rewards for each meaningful milestone. This helps RL agents learn faster by:
- Providing clear feedback at each step
- Encouraging exploration of correct actions
- Breaking down complex sequences into manageable sub-goals

### One-Time vs. Cumulative
- **One-time rewards** (most milestones) prevent reward farming and ensure progress
- **Cumulative rewards** (damage dealt) encourage repeated beneficial actions during battles
- **Efficiency bonuses** reward optimal strategies without penalizing learning

### Balanced Weights
Reward weights are calibrated such that:
- Early milestones (leaving house) give modest rewards to encourage exploration
- Major story events (getting Pokemon) give large rewards (100+ points)
- Battle progression gives incremental rewards (5-35 points)
- Winning the battle gives the highest single reward (150 points)
- Efficiency bonuses are significant but optional (20-50 points)

## Testing

The reward function has been tested with:
- Unit tests for individual components
- Integration tests with full environment
- RL training runs with PPO and GRPO algorithms

## Future Enhancements

Potential additions for expanded coverage:
- Route 1 exploration rewards
- Viridian City navigation rewards
- Pokemon Center discovery rewards
- Wild encounter rewards
- Capture mechanics rewards

See `story_rewards.py` and `novelty_rewards.py` for complementary reward functions covering later game sections.

