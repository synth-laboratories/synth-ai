# Pokemon Red Environment

A comprehensive reinforcement learning environment for Pokemon Red with advanced reward shaping and v3 tracing support.

## Overview

This environment provides a deterministic reward system for Pokemon Red that guides agents toward meaningful progress through exploration, training, and gym challenges. The system includes:

- **Rich reward shaping** with 10+ reward components
- **V3 tracing** with automatic reward/event logging to Turso database
- **Multi-model evaluation** support (GPT-4o-mini, GPT-4o, etc.)
- **Achievement tracking** with detailed performance analytics

## Quick Start

### 1. Install Dependencies

```bash
cd /path/to/synth-ai
pip install -e .
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run Basic Evaluation

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
python test_new_reward_system.py
```

This runs 5 episodes with 20 tool calls each using GPT-4o-mini.

### 4. Check Results in Database

```bash
python -c "
import asyncio
from synth_ai.tracing_v3.session_tracer import SessionTracer

async def check_results():
    tracer = SessionTracer()
    await tracer.initialize()
    # Query results...
    results = await tracer.db.query_traces('SELECT * FROM session_traces WHERE session_id LIKE \"pokemon_red_eval%\"')
    print(results)

asyncio.run(check_results())
"
```

## Reward System

### Components

The environment uses a stacked reward system with 10+ components:

#### Exploration Rewards
- **New Area Discovery** (+0.02): Moving to unexplored locations
- **Quick Exploration** (+0.04): Discovering multiple areas efficiently
- **Route Exploration** (+2.0): Entering major routes (Route 1, Viridian Forest, etc.)
- **City Discovery** (+1.5): Entering cities (Viridian, Pewter)
- **Area Transitions** (+1.0): Moving between major zones

#### Training & Battle Rewards
- **Pokemon Level Up** (+0.2): Any Pokemon gains experience
- **Level Milestones** (+0.3): Reaching power levels (8, 12, 15)
- **Pokemon Ready** (+3.0): Pokemon strong enough for gym battle (level ≥10)
- **Battle Engagement** (+0.1): Starting Pokemon battles

#### Resource Management
- **Item Collection** (+0.5): Finding valuable items (Pokeballs, Potions, TMs)
- **Pokemon Center** (+0.8): Restoring significant HP
- **Health Maintenance** (+0.05): Keeping Pokemon healthy

#### Major Achievements
- **Gym Entry** (+5.0): Entering Pewter Gym building
- **Defeat Brock** (+50.0): Winning Boulder Badge

### Key Features

- **No negative penalties** - pure positive reinforcement
- **Deterministic rewards** - same actions always yield same rewards
- **Progressive difficulty** - rewards scale with game progression
- **Exploration encouragement** - small rewards for discovering new areas

## V3 Tracing & Data Collection

### Automatic Data Logging

The environment automatically logs all events and rewards to the Turso database:

```python
# Environment automatically creates:
# - EnvironmentEvent objects for each step
# - EventReward records for reward analysis
# - OutcomeReward records for episode summaries
# - Complete session traces with metadata
```

### Database Schema

#### Session Traces
```sql
CREATE TABLE session_traces (
    session_id TEXT PRIMARY KEY,
    created_at DATETIME,
    num_timesteps INTEGER,
    num_events INTEGER,
    num_messages INTEGER,
    metadata JSON
);
```

#### Events
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    event_type TEXT, -- 'environment', 'runtime', 'cais'
    reward FLOAT,    -- Reward value for environment events
    terminated BOOLEAN,
    truncated BOOLEAN,
    -- Additional metadata and state information
);
```

#### Event Rewards
```sql
CREATE TABLE event_rewards (
    id INTEGER PRIMARY KEY,
    event_id INTEGER,
    session_id TEXT,
    reward_value FLOAT,
    reward_type TEXT, -- 'sparse', 'shaped', 'achievement'
    key TEXT,         -- Achievement name
    annotation JSON   -- Detailed reward metadata
);
```

### Querying Results

#### Get all Pokemon Red sessions:
```python
sessions = await db.query_traces("""
    SELECT * FROM session_traces
    WHERE session_id LIKE 'pokemon_red_eval_%'
    ORDER BY created_at DESC
""")
```

#### Get reward performance:
```python
rewards = await db.query_traces("""
    SELECT s.session_id, SUM(e.reward) as total_reward,
           COUNT(CASE WHEN e.reward > 0 THEN 1 END) as achievements
    FROM session_traces s
    JOIN events e ON s.session_id = e.session_id
    WHERE s.session_id LIKE 'pokemon_red_eval_%'
    GROUP BY s.session_id
    ORDER BY total_reward DESC
""")
```

#### Get achievement breakdown:
```python
achievements = await db.query_traces("""
    SELECT er.key, COUNT(*) as count, AVG(er.reward_value) as avg_reward
    FROM event_rewards er
    JOIN session_traces s ON er.session_id = s.session_id
    WHERE s.session_id LIKE 'pokemon_red_eval_%' AND er.reward_value > 0
    GROUP BY er.key
    ORDER BY count DESC
""")
```

## Running Evaluations

### Single Model Evaluation

```bash
# Run with GPT-4o-mini (default)
python test_new_reward_system.py

# Run with specific model
# Edit test_new_reward_system.py line 316: model_name = "gpt-4o"
python test_new_reward_system.py
```

### Multi-Model Comparison

```python
# Edit seeds and model in test_new_reward_system.py
seeds = [0, 1, 2, 3, 4]  # Different seeds
model_name = "gpt-4o"     # Different model

python test_new_reward_system.py
```

### Custom Evaluation Parameters

Modify `test_new_reward_system.py`:

```python
# Change these parameters:
seeds = [100, 101, 102]    # Custom seeds
max_steps = 50             # More/less steps per episode
model_name = "gpt-4o"      # Different model

# Then run:
python test_new_reward_system.py
```

## Performance Metrics

### Achievement Efficiency
```
Achievement Efficiency = (Positive Reward Events / Total Actions) × 100%
```

Typical results:
- **GPT-4o-mini**: ~21% achievement efficiency
- **Exploration focus**: Agents prioritize area discovery
- **Reward range**: 0.02-50.0 depending on achievement

### Benchmark Scores

| Model | Best Score | Avg Score | Achievement Rate |
|-------|------------|-----------|------------------|
| GPT-4o-mini | 0.32 | 0.16 | 21.4% |
| GPT-4o | TBD | TBD | TBD |

## Architecture

### Environment Structure
```
PokemonRedEnvironment
├── PokemonRedEngine (game logic & state)
├── RewardStack (10+ reward components)
├── Trace Hooks (automatic event logging)
└── V3 Tracing (Turso database integration)
```

### Reward Flow
1. **Action executed** → Engine updates game state
2. **Reward calculated** → Stack evaluates all components
3. **Event logged** → EnvironmentEvent created with reward
4. **Trace hooks fired** → EventReward records created
5. **Database saved** → All data persisted to Turso

### Key Files

- `environment.py` - Main environment class with tracing integration
- `engine.py` - Game logic and reward stack implementation
- `engine_helpers/reward_components.py` - Individual reward components
- `trace_hooks_v3.py` - Automatic event/reward logging hooks
- `test_new_reward_system.py` - Evaluation script

## Troubleshooting

### Database Connection Issues
```bash
# Check Turso connection
python -c "
from synth_ai.tracing_v3.session_tracer import SessionTracer
import asyncio
async def test(): tracer = SessionTracer(); await tracer.initialize(); print('✅ Connected' if tracer.db else '❌ Failed')
asyncio.run(test())
"
```

### Missing Rewards
- Check that `priv_state.reward_last_step` exists
- Verify tracer is passed to environment: `PokemonRedEnvironment(tracer=tracer)`
- Check database for events: `SELECT * FROM events WHERE reward > 0`

### Performance Issues
- Reduce `max_steps` for faster testing
- Use fewer seeds for quicker iterations
- Check OpenAI API rate limits

## Contributing

### Adding New Reward Components

1. Create component in `engine_helpers/reward_components.py`:
```python
class NewReward(RewardComponent):
    async def score(self, state: Dict, action: Dict) -> float:
        # Your reward logic here
        return reward_value
```

2. Add to reward stack in `engine.py`:
```python
self.reward_stack = RewardStack([
    # ... existing components
    NewReward(),
])
```

3. Update achievement mapping in `trace_hooks_v3.py`

### Modifying Evaluation Script

Edit `test_new_reward_system.py` to:
- Change models, seeds, or step counts
- Add new evaluation metrics
- Modify reward analysis logic

## Future Improvements

- [ ] Add more reward components (NPC interactions, item usage)
- [ ] Implement curriculum learning (progressive difficulty)
- [ ] Add multi-agent support
- [ ] Create web dashboard for result visualization
- [ ] Add automated benchmarking suite