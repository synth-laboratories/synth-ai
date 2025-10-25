# Pokemon Battle Task App

This example shows how to expose a Horizons-compatible Pokémon Showdown battle
environment through the Synth AI task app scaffolding. The adapter runs fully
locally by driving pokechamp’s deterministic `LocalSim`, so battles can be
snapshotted and restored without a live Showdown server.

## Local setup (Track 1)

1. Clone and install **PokeChamp** together with its `poke-env` fork:

   ```bash
   git clone https://github.com/sethkarten/pokechamp.git
   cd pokechamp
   pip install -r requirements.txt
   ```

2. Export environment variables so the task app can locate the cloned repo:

   ```bash
   export POKECHAMP_ROOT=/path/to/pokechamp
   export POKEMON_SHOWDOWN_ROOT=/path/to/pokemon-showdown
   ```

3. Run a rollout to sanity-check the wiring:

   ```bash
   uv run python -m synth_ai.task.describe pokemon_showdown
   uv run python -m synth_ai.task.rollout pokemon_showdown --seed 1001
   ```

The adapter uses the pokechamp dataset teams bundled with the repository to
instantiate deterministic Gen 9 OU battles. You can point `POKECHAMP_ROOT` at a
fork with custom teams to experiment with other formats.

## Modal deployment

A ready-to-use deployment helper is available at
`examples/task_apps/pokemon_battle/modal_app.py`. It mirrors the above manual
steps (cloning `pokechamp`, installing requirements, and mounting the Synth AI
repo). Deploy with:

```bash
modal deploy examples/task_apps/pokemon_battle/modal_app.py
```

The resulting URL can be plugged into Synth AI workflows via `TASK_APP_URL`.

## Notes

- The dataset catalog resolves team files from the PokeChamp repo when available
  (`POKECHAMP_ROOT`). If the assets are missing, `/info` marks the scenario as
  unavailable.
- Snapshots serialise the entire deterministic battle state, allowing training
  algorithms to branch or reset mid-match.
- Deterministic RNG seeding (Python, NumPy, PyTorch) keeps rollouts reproducible
  across Modal replicas and local runs.
- The opponent policy now favours super-effective moves to provide a stronger
  baseline; swap it out with a pokechamp minimax bot for ladder-level play.
- A `/healthz` endpoint is exposed in the Modal service for liveness probes.

## Status & Next Steps

- **Observation polish**: expose richer per-turn summaries (hazards, stat boosts, tera states) and compact text strings tailored for language agents.
- **Action helpers**: surface explicit target slots/tera/mega toggles so higher formats (doubles, VGC) can plug in with minimal code.
- **Benchmark opponent**: replace the heuristic opponent with a pokechamp bot (e.g. minimax) or hook into the official PokéAgent ladder for eval parity.
- **Integration tests**: add pytest smoke tests covering `/snapshot` → `/restore` loops and multi-step rollouts.
- **Agent wiring**: ship a reference RL/LLM policy config (Synth CLI or Modal job) that exercises the adapter end-to-end and logs battle traces.
