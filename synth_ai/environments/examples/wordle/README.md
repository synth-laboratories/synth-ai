# Wordle Example Environment

A lightweight Wordle implementation following Synth-AI’s stateful environment pattern.
It includes engine/env abstractions, tool-based interaction, and a procedural taskset.

## Features
- Stateful engine with reproducible snapshots and seeds
- Reward stack: +1 on win, -1 on invalid guess, 0 otherwise
- Toggle for invalid attempts consuming turns (default: consume)
- Procedural `TaskInstanceSet` with fixed targets and splits
- Text-first observations suitable for LLM prompting

## Quick Start
```python
import asyncio
from synth_ai.environments.examples.wordle import create_wordle_taskset, WordleEnvironment

async def main():
    taskset = await create_wordle_taskset(sample_size=3)
    inst = taskset.instances[0]
    env = WordleEnvironment(inst)
    obs = await env.initialize()
    print(obs["text"])  # text summary with board and status

    # Submit guesses via the standard tool-call format
    obs = await env.step({"guess": "trace"})
    obs = await env.step({"guess": inst.metadata.target_word})
    print("status:", obs["status"], "total_reward:", obs["total_reward"])  # won, 1.0

asyncio.run(main())
```

## Instances and Seeds
- Each `WordleTaskInstance` metadata includes:
  - `word_length`, `max_guesses`, `target_word`, `enforce_wordlist`, `seed`,
  - `consume_invalid_attempts` (see below).
- If `target_word` is empty, the engine selects deterministically from a small pool using `seed`.

## Invalid Attempts Toggle
- Behavior is controlled by `consume_invalid_attempts` (default: `True`).
  - `True`: Invalid guesses consume a turn and apply -1 reward. They are not added to the board.
  - `False`: Invalid guesses do not consume a turn but still apply -1 reward.
- Configure via taskset:
```python
# Do not consume invalid attempts
taskset = await create_wordle_taskset(sample_size=5, consume_invalid_attempts=False)
```

## Serialization
Serialize engine state and restore later:
```python
snapshot = await env._serialize_engine()
rehydrated = await WordleEnvironment._deserialize_engine(snapshot, inst)
obs2 = await rehydrated.step({"guess": "slate"})
```

## Testing
Run the Wordle unit tests only:
```bash
pytest tests/environments/unit/test_wordle.py -q
```

## Large Word Bank (English-only)
- Default instances file: `synth_ai/environments/examples/wordle/instances.json`.
- To expand to 500+ English words without adding runtime dependencies, generate once with `wordfreq` and commit the JSON:
  - `pip install wordfreq`
  - `python -m synth_ai.environments.examples.wordle.helpers.generate_instances_wordfreq --count 500 --min-zipf 3.0 --outfile synth_ai/environments/examples/wordle/instances.json`
  - Commit the updated `instances.json` so it’s used by default.
- To test a different bank without touching the repo, point the loader at a custom file:
  - `export WORDLE_INSTANCES_JSON=/path/to/my_instances.json`

## Tool API
- Tool: `interact`
- Args schema: `{ "guess": str }`
- Returns: Standard `ToolResult` with `public_state`/`private_state` used by the env to form observations.

## Notes
- Built-in solution pool is intentionally small to avoid heavy dependencies; swap in a larger list by extending the engine or generating instances from an external corpus.
- Board shows only accepted (valid) guesses; invalid guesses never appear on the board regardless of the toggle.
