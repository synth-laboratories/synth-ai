import os
from pathlib import Path

import pytest

from examples.task_apps.pokemon_emerald.task_app.pokemon_emerald import (
    DATASET_SPEC,
    PokemonEmeraldAdapter,
    PokemonEmeraldDataset,
)

pytest.importorskip("mgba.core", reason="mGBA emulator bindings are required for this test.")

_DEFAULT_SPEEDRUN_ROOT = (
    Path("examples/task_apps/dev/pokemon_emerald/external/pokeagent-speedrun").resolve()
)
if "POKEAGENT_SPEEDRUN_ROOT" not in os.environ and _DEFAULT_SPEEDRUN_ROOT.exists():
    os.environ["POKEAGENT_SPEEDRUN_ROOT"] = str(_DEFAULT_SPEEDRUN_ROOT)


def _resolve_rom(scenario: dict[str, object]) -> Path | None:
    candidates: list[Path] = []
    rom_env = os.getenv("POKEMON_EMERALD_ROM")
    if rom_env:
        candidates.append(Path(rom_env).expanduser())
    assets_root = os.getenv("POKEMON_EMERALD_ASSETS")
    if assets_root:
        candidates.append(Path(assets_root).expanduser() / "emerald.gba")
    checkpoint_ref = scenario.get("checkpoint_ref")
    if isinstance(checkpoint_ref, str):
        candidates.append(Path(checkpoint_ref).resolve().parent / "rom.gba")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(
    not os.getenv("POKEAGENT_SPEEDRUN_ROOT"),
    reason="POKEAGENT_SPEEDRUN_ROOT must point at the pokeagent-speedrun repository.",
)
def test_snapshot_restore_round_trip():
    dataset = PokemonEmeraldDataset(DATASET_SPEC)
    seeds = dataset.seeds
    assert seeds, "Dataset must expose at least one seed."
    scenario = dataset.describe_seed(seeds[0])
    if not scenario.get("assets_ready"):
        pytest.skip("Scenario assets are unavailable.")

    rom_path = _resolve_rom(scenario)
    if rom_path is None:
        pytest.skip("Pokémon Emerald ROM not found.")

    adapter = PokemonEmeraldAdapter(scenario=scenario, rom_path=rom_path)

    try:
        initial_obs = adapter.reset()
        frame = initial_obs.get("frame_png")
        assert frame is not None and frame.startswith("data:image/png;base64,"), "Frame PNG must be present."

        snapshot = adapter.snapshot()

        step_obs, reward, done, _info = adapter.step({"macro": "noop", "frames": 6})
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert step_obs.get("frame_png"), "Frame PNG should remain available after stepping."

        restored_obs = adapter.restore(snapshot)
        assert restored_obs["player_state"] == initial_obs["player_state"]
        assert restored_obs["frame_png"] == initial_obs["frame_png"]
    finally:
        adapter.close()
