# Sokoban Task App

A task app for training and evaluating LLM agents on Sokoban puzzles.

Sokoban is a classic puzzle game where the player must push boxes onto target locations. It's a good benchmark for spatial reasoning, planning, and sequential decision-making.

## Features

- 🎮 Multiple difficulty levels (easy, medium, hard)
- 🤖 LLM policy support (GPT-5-mini, Qwen)
- 📊 Supports both RL training and evaluation rollouts
- 🎯 Rich observations with ASCII grid visualization
- ⚡ Batched actions (up to 8 actions per LLM call)

## Quick Start

### 1. Start the Server

```bash
cd /path/to/synth-ai

# Start the Sokoban task app on port 8911
uvx synth-ai task-app deploy --runtime uvicorn sokoban --port 8911
```

The server will be available at `http://localhost:8911`.

### 2. Run a Test Rollout

#### Option A: Using GPT-5-mini

```bash
export OPENAI_API_KEY="your-api-key"

python3 << 'EOF'
import httpx
import asyncio

async def test_gpt5mini():
    async with httpx.AsyncClient(timeout=600.0) as client:  # Longer timeout
        print("🎮 Testing with GPT-5-mini (slower due to reasoning tokens)...\n")
        
        response = await client.post(
            "http://localhost:8911/rollout",
            json={
                "run_id": "test_gpt5mini",
                "env": {"seed": 123, "config": {"difficulty": "easy", "max_steps": 100}},
                "ops": ["policy"] * 5,  # Fewer calls due to slowness
                "policy": {
                    "config": {
                        "provider": "openai",
                        "model": "gpt-5-mini",
                        "max_actions_per_call": 8
                    }
                }
            },
            headers={"Authorization": "Bearer sk_env_your_key_here"}
        )
        
        result = response.json()
        traj = result["trajectories"][0]
        final = traj["final"]["observation"]
        
        print(f"Boxes: {final['boxes_on_target']}/{final['num_boxes']}")
        print(f"Steps: {final['steps_taken']}")

asyncio.run(test_gpt5mini())
EOF
```

#### Option B: Using Qwen via Groq (Fast & Cheap)

```bash
export GROQ_API_KEY="your-groq-key"

python3 << 'EOF'
import httpx
import asyncio

async def test_qwen():
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:8911/rollout",
            json={
                "run_id": "test_qwen",
                "env": {"seed": 123, "config": {"difficulty": "easy", "max_steps": 100}},
                "ops": ["policy"] * 15,
                "policy": {
                    "config": {
                        "provider": "groq",
                        "model": "qwen-2.5-7b",
                        "max_actions_per_call": 8
                    }
                }
            },
            headers={"Authorization": "Bearer sk_env_your_key_here"}
        )
        
        result = response.json()
        traj = result["trajectories"][0]
        final = traj["final"]["observation"]
        
        print(f"Result: {'✅ SOLVED!' if final['boxes_on_target'] == final['num_boxes'] else '❌ Not solved'}")
        print(f"Boxes: {final['boxes_on_target']}/{final['num_boxes']}")

asyncio.run(test_qwen())
EOF
```

## Configuration Options

### Environment Config

```python
{
    "seed": 123,              # Random seed for puzzle generation
    "config": {
        "difficulty": "easy",  # "easy", "medium", or "hard"
        "max_steps": 100      # Maximum steps before truncation
    }
}
```

### Policy Config

```python
{
    "provider": "openai",           # "openai" or "groq"
    "model": "gpt-5-mini",          # Model name
    "max_actions_per_call": 8,      # Actions per policy call (1-8)
    "temperature": 0.7,             # Temperature (optional)
    "max_completion_tokens": 4000   # Max tokens (optional)
}
```

## Model Recommendations

| Model | Status | Speed | Notes |
|-------|--------|-------|-------|
| **gpt-5-mini** | ✅ Recommended | Slow (30-50s/call) | Uses 1500-2750 reasoning tokens per call |
| **gpt-5** | ❌ Not supported | N/A | Doesn't support tool calling |
| **gpt-5-nano** | ❌ Not supported | N/A | Doesn't support tool calling |
| **qwen-2.5-7b** (Groq) | ✅ Works | Very fast | Cheap and fast alternative |

### Why is GPT-5-mini slow?

GPT-5-mini uses extensive internal reasoning (1500-2750 reasoning tokens per call) before generating actions. While this could lead to better puzzle-solving, it makes each policy call take 30-50 seconds.

Example usage breakdown:
```json
{
  "usage": {
    "completion_tokens": 2465,
    "reasoning_tokens": 2432,  // Deep thinking!
    "prompt_tokens": 470
  }
}
```

## Observation Format

Each observation includes:

```python
{
    "room_text": str,           # ASCII visualization of the puzzle
    "player_position": [x, y],  # Player coordinates
    "boxes_on_target": int,     # Number of boxes on target squares
    "num_boxes": int,           # Total number of boxes
    "steps_taken": int,         # Steps taken so far
    "max_steps": int,           # Maximum allowed steps
    "last_action": str,         # Last action taken
    "reward_last": float,       # Reward from last step
    "total_reward": float,      # Cumulative reward
    "terminated": bool,         # Puzzle solved?
    "truncated": bool           # Max steps reached?
}
```

### ASCII Legend

- `P` = Player
- `O` = Box
- `X` = Target square
- `@` = Box on target
- `+` = Player on target
- `#` = Wall
- `_` = Floor

## Action Space

The agent uses the `interact_many` tool to execute multiple actions in sequence:

```python
{
    "tool": "interact_many",
    "args": {
        "actions": [0, 1, 2, 3]  # 0=left, 1=up, 2=right, 3=down
    }
}
```

Or with string names:
```python
{
    "actions": ["left", "up", "right", "down"]
}
```

## Training with RL

The Sokoban task app supports RL training. Example config:

```toml
# sokoban_rl_config.toml
[task_app]
url = "http://localhost:8911"
auth_token = "sk_env_your_key_here"

[rl]
algorithm = "grpo"
num_episodes = 1000
batch_size = 32

[policy]
provider = "groq"
model = "qwen-2.5-7b"
max_actions_per_call = 8

[env]
difficulty = "easy"
max_steps = 100
```

Run training:
```bash
uvx synth-ai train --config sokoban_rl_config.toml
```

## Debugging

### Check server health
```bash
curl http://localhost:8911/health
```

### View server logs
```bash
# If running with nohup
tail -f nohup_sokoban.log

# Filter for important logs
tail -f nohup_sokoban.log | grep -E "extract|debug|error"
```

### Test with explicit actions
```python
# Instead of "policy", provide explicit actions
"ops": [
    {"button": "right", "count": 3},
    {"button": "down", "count": 2}
]
```

## Troubleshooting

### Empty responses from LLM
- **GPT-5/GPT-5-nano**: These models don't support tool calling reliably. Use GPT-5-mini instead.
- **Timeout errors**: GPT-5-mini is slow. Increase client timeout to 600+ seconds or use fewer policy calls.

### Puzzle not solving
- Try more policy calls (15-30)
- Use a different seed
- Try "easy" difficulty first
- Check if the agent is stuck in a loop (repeating same actions)

### Server won't start
```bash
# Check if port is in use
lsof -i :8911

# Kill existing process
kill -9 $(lsof -ti :8911)

# Restart
uvx synth-ai task-app deploy --runtime uvicorn sokoban --port 8911
```

## Examples

See the `examples/workflows/` directory for:
- RL training scripts
- Evaluation scripts
- Multi-episode parallel evaluation

## Contributing

To add new features:
1. Edit `task_app.py` for core logic
2. Update `_base_task_info()` for new observation/action specs
3. Modify `rollout_executor()` for custom rollout behavior
4. Add tests in `tests/integration/`

## License

MIT
