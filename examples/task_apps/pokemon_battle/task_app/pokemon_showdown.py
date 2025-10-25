"""Task App configuration for a Horizons-backed Pokémon Showdown battle environment."""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import numpy as np
from fastapi import HTTPException, Request
from poke_env.data.gen_data import GenData
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player.battle_order import BattleOrder
from poke_env.player.local_simulation import LocalSim
from poke_env.player.baselines import AbyssalPlayer
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.server import ProxyConfig, TaskAppConfig

logger = logging.getLogger(__name__)


DATASET_SPEC = TaskDatasetSpec(
    id="pokemon_showdown_reference",
    name="Pokémon Showdown Reference Matches",
    version="0.1.0",
    splits=["train", "eval"],
    default_split="train",
    description=(
        "Seeded Gen 9 OU matches derived from the PokeChamp benchmark packs and "
        "PokéAgent Track 1 starter kit."
    ),
)


def _resolve_repo_root(env_key: str, repo_dir: str) -> Path | None:
    env_path = os.getenv(env_key)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for ancestor in here.parents:
        candidates.append(ancestor / "external" / repo_dir)
        candidates.append(ancestor / repo_dir)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:  # pragma: no cover - path resolution edge cases
            continue
        if resolved.exists():
            return resolved
    return None


def _ensure_on_path(path: Path | None) -> None:
    if not path:
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _read_text_if_exists(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return path.read_text()
    except Exception:
        return None


@dataclass(frozen=True)
class PokemonBattleScenario:
    seed: int
    name: str
    format_id: str
    player_team_ref: str
    opponent_team_ref: str
    description: str
    source: str
    tags: tuple[str, ...] = ()


class PokemonBattleDataset:
    """In-memory catalogue of deterministic battle scenarios."""

    def __init__(self, spec: TaskDatasetSpec) -> None:
        self.spec = spec
        self.repo_root = _resolve_repo_root("POKECHAMP_ROOT", "pokechamp")
        _ensure_on_path(self.repo_root)

        self._team_roots: list[Path] = []
        if self.repo_root:
            self._team_roots.extend(
                [
                    self.repo_root / "poke_env" / "data" / "static" / "gen9" / "ou",
                    self.repo_root / "poke_env" / "data" / "static" / "teams",
                    self.repo_root / "resource" / "teams",
                ]
            )

        scenarios: list[PokemonBattleScenario] = [
            PokemonBattleScenario(
                seed=1001,
                name="hazard_balance_vs_pivot_offense",
                format_id="gen9ou",
                player_team_ref="poke_env/data/static/gen9/ou/gen9ou-1825.txt",
                opponent_team_ref="poke_env/data/static/gen9/ou/gen9ou-1500.txt",
                description=(
                    "Balanced hazard stack roster into a pivot-heavy opponent. "
                    "Mirrors the PokeChamp ICML 2025 evaluation seed."
                ),
                source="https://github.com/sethkarten/pokechamp",
                tags=("benchmark", "pokechamp", "gen9"),
            ),
            PokemonBattleScenario(
                seed=2002,
                name="sunroom_vs_rainroom",
                format_id="gen9ou",
                player_team_ref="poke_env/data/static/gen9/ou/gen9ou-1500.txt",
                opponent_team_ref="poke_env/data/static/gen9/ou/gen9ou-0.txt",
                description="Weather control showdown drawn from the PokéAgent ladder starter.",
                source="https://pokeagent.github.io/track1.html",
                tags=("ladder", "weather", "gen9"),
            ),
            PokemonBattleScenario(
                seed=3003,
                name="stall_vs_hyper_offense",
                format_id="gen9ou",
                player_team_ref="poke_env/data/static/gen9/ou/gen9ou-0.txt",
                opponent_team_ref="poke_env/data/static/gen9/ou/gen9ou-1825.txt",
                description="Long-horizon stall versus hyper-offense curriculum seed.",
                source="https://github.com/sethkarten/pokechamp",
                tags=("curriculum", "gen9"),
            ),
        ]
        self._scenarios: dict[int, PokemonBattleScenario] = {s.seed: s for s in scenarios}
        self.default_seed = scenarios[0].seed

    @property
    def seeds(self) -> list[int]:
        return sorted(self._scenarios)

    @property
    def formats(self) -> list[str]:
        return sorted({scenario.format_id for scenario in self._scenarios.values()})

    @property
    def count(self) -> int:
        return len(self._scenarios)

    def resolve_seed(self, seed: int | None) -> int:
        if seed is None:
            return self.default_seed
        if seed not in self._scenarios:
            raise KeyError(f"Unknown battle seed: {seed}")
        return seed

    def describe_seed(self, seed: int) -> dict[str, Any]:
        scenario = self._scenarios.get(seed)
        if not scenario:
            raise KeyError(f"Unknown battle seed: {seed}")

        player_team_text = self._load_team_text(scenario.player_team_ref)
        opponent_team_text = self._load_team_text(scenario.opponent_team_ref)

        return {
            "seed": seed,
            "name": scenario.name,
            "format_id": scenario.format_id,
            "player_team_ref": scenario.player_team_ref,
            "player_team": player_team_text,
            "opponent_team_ref": scenario.opponent_team_ref,
            "opponent_team": opponent_team_text,
            "description": scenario.description,
            "source": scenario.source,
            "tags": list(scenario.tags),
            "assets_ready": bool(player_team_text and opponent_team_text),
        }

    def _load_team_text(self, reference: str) -> str | None:
        raw_ref = reference.strip()
        if not raw_ref:
            return None

        if raw_ref.startswith("text:"):
            return raw_ref.split(":", 1)[1]

        candidates: list[Path] = []
        ref_path = Path(raw_ref)
        if ref_path.is_absolute():
            candidates.append(ref_path)

        if self.repo_root:
            candidates.append(self.repo_root / raw_ref)

        for base in self._team_roots:
            candidates.append(base / ref_path.name)
            candidates.append(base / raw_ref)

        for candidate in candidates:
            if candidate.exists():
                return _read_text_if_exists(candidate)
        return None


def _build_dataset_registry() -> tuple[TaskDatasetRegistry, PokemonBattleDataset]:
    registry = TaskDatasetRegistry()
    dataset = PokemonBattleDataset(DATASET_SPEC)
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: PokemonBattleDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "pokemon_showdown", "name": "Pokémon Showdown Battle", "version": "0.1.0"},
        environment="pokemon_showdown",
        action_space={
            "type": "structured",
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"enum": ["move", "switch", "team-preview"]},
                    "index": {"type": "integer", "minimum": 0},
                    "target": {"type": "integer", "minimum": 0, "nullable": True},
                    "metadata": {"type": "object"},
                },
                "required": ["action"],
            },
            "notes": "Legal indices are surfaced in observation['legal_actions'].",
        },
        observation={
            "summary": "Structured Showdown state and a text rendering per turn.",
            "keys": ["structured", "legal_actions", "text"],
            "text_role": "Battle transcript for language agents.",
        },
        dataset={
            **DATASET_SPEC.model_dump(),
            "seed_count": dataset.count,
            "seeds": dataset.seeds,
            "formats": dataset.formats,
            "source_repos": [
                "https://github.com/sethkarten/pokechamp",
                "https://pokeagent.github.io/track1.html",
            ],
            "pokechamp_root": str(dataset.repo_root) if dataset.repo_root else None,
        },
        rubric={
            "version": "1",
            "criteria_count": 2,
            "source": "inline",
            "summary": "Win/loss outcome plus faint differential.",
        },
        inference={
            "supports_proxy": True,
            "tool": {"name": "battle_action", "parallel_tool_calls": False},
            "endpoints": {
                "openai": "/proxy/v1/chat/completions",
                "groq": "/proxy/groq/v1/chat/completions",
            },
        },
        limits={"max_turns": 200, "max_time_s": 1800, "max_ops": 4096},
        task_metadata={
            "preferred_engine": "pokechamp",
            "supports_remote_server": True,
            "documentation": "https://github.com/sethkarten/pokechamp",
        },
    )


def describe_taskset(dataset: PokemonBattleDataset) -> dict[str, Any]:
    return {
        **DATASET_SPEC.model_dump(),
        "count": dataset.count,
        "seeds": dataset.seeds,
        "formats": dataset.formats,
        "assets_ready": all(dataset.describe_seed(seed)["assets_ready"] for seed in dataset.seeds),
    }


def provide_task_instances(
    dataset: PokemonBattleDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    base_observation = getattr(base_info, "observation", None)
    if hasattr(base_observation, "model_dump"):
        observation_template = base_observation.model_dump()
    elif isinstance(base_observation, dict):
        observation_template = dict(base_observation)
    else:
        observation_template = {}

    for seed_value in seeds:
        resolved_seed = dataset.resolve_seed(seed_value)
        details = dataset.describe_seed(resolved_seed)
        infos.append(
            TaskInfo(
                task=base_info.task,
                environment=base_info.environment,
                action_space=base_info.action_space,
                observation={
                    **observation_template,
                    "seed": resolved_seed,
                    "format_id": details["format_id"],
                    "player_team_ref": details["player_team_ref"],
                    "opponent_team_ref": details["opponent_team_ref"],
                    "description": details["description"],
                },
                dataset={
                    **base_info.dataset.model_dump(),
                    "seed": resolved_seed,
                    "scenario": details,
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
                limits=base_info.limits,
                task_metadata={
                    **base_info.task_metadata,
                    "source": details["source"],
                    "tags": details["tags"],
                    "assets_ready": details["assets_ready"],
                },
            )
        )
        return infos


class PokechampAssets:
    """Lazy loader for pokechamp static data used by the local simulator."""

    move_effect: dict[str, Any] = {}
    pokemon_move_dict: dict[str, Any] = {}
    ability_effect: dict[str, Any] = {}
    pokemon_ability_dict: dict[str, Any] = {}
    item_effect: dict[str, Any] = {}
    pokemon_item_dict: dict[str, Any] = {}
    loaded = False

    @classmethod
    def ensure_loaded(cls, repo_root: Path) -> None:
        if cls.loaded:
            return

        def _require_json(rel_path: str) -> dict[str, Any]:
            path = repo_root / rel_path
            if not path.exists():
                raise FileNotFoundError(
                    f"Required pokechamp asset missing at {path}. "
                    "Ensure POKECHAMP_ROOT is mounted with the repository assets."
                )
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        cls.move_effect = _require_json("poke_env/data/static/moves/moves_effect.json")
        cls.pokemon_move_dict = _require_json("poke_env/data/static/moves/gen8pokemon_move_dict.json")
        cls.ability_effect = _require_json("poke_env/data/static/abilities/ability_effect.json")
        cls.pokemon_ability_dict = _require_json("poke_env/data/static/abilities/gen8pokemon_ability_dict.json")
        cls.item_effect = _require_json("poke_env/data/static/items/item_effect.json")
        cls.pokemon_item_dict = {}
        cls.loaded = True


class PokemonShowdownAdapter:
    """Local deterministic battle adapter powered by pokechamp's LocalSim."""

    STEP_PENALTY = 0.05
    WIN_REWARD = 1.0
    LOSS_PENALTY = -1.0

    def __init__(self, *, scenario: dict[str, Any], repo_root: Path, seed: int | None = None):
        if not repo_root.exists():
            raise FileNotFoundError(
                f"Pokechamp repository root not found at {repo_root}. "
                "Set POKECHAMP_ROOT to the cloned repository."
            )

        if not scenario.get("assets_ready"):
            raise ValueError(
                f"Scenario '{scenario['name']}' is missing team assets. "
                "Ensure the pokechamp dataset files are present."
            )

        PokechampAssets.ensure_loaded(repo_root)
        self.scenario = scenario
        self.repo_root = repo_root
        self.format_id = scenario["format_id"]

        seed_value = seed if seed is not None else scenario.get("seed", 0)
        random.seed(seed_value)
        np.random.seed(seed_value)
        try:  # pragma: no cover - optional dependency
            import torch

            torch.manual_seed(seed_value)
        except Exception:
            pass

        self.random = random.Random(seed_value)
        self.gen_data = GenData.from_format(self.format_id)

        player_team_text = scenario.get("player_team")
        opponent_team_text = scenario.get("opponent_team")
        if not player_team_text or not opponent_team_text:
            raise ValueError(
                f"Scenario '{scenario['name']}' is missing team definitions."
            )

        self._base_battle = self._build_base_battle(
            player_team_text=player_team_text,
            opponent_team_text=opponent_team_text,
        )
        self.sim = self._create_sim(deepcopy(self._base_battle))
        self.battle = self.sim.battle
        self._sync_available_actions()

        self._prev_score = self._score()
        self.turn = 0
        self.done = False
        self.outcome = 0.0

    def reset(self) -> dict[str, Any]:
        self.sim = self._create_sim(deepcopy(self._base_battle))
        self.battle = self.sim.battle
        self._sync_available_actions()
        self._prev_score = self._score()
        self.turn = 0
        self.done = False
        self.outcome = 0.0
        self._abyssal = AbyssalPlayer(
            battle_format=self.format_id,
            team=self._opponent_packed_team,
            save_replays=False,
            log_level=logging.WARNING,
        )
        return self._build_observation()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Battle finished. Call reset() before stepping again.")

        agent_order = self._action_to_order(action)
        opponent_order = self._opponent_policy()

        self.sim.step(agent_order, opponent_order)
        self.battle = self.sim.battle
        self._sync_available_actions()

        reward = -self.STEP_PENALTY
        current = self._score()
        reward += current - self._prev_score
        self._prev_score = current
        self.turn += 1

        self.done = self._check_finished()
        if self.done:
            reward += self.outcome

        observation = self._build_observation()
        info = {"legal_actions": self._legal_actions(), "turn": self.turn}
        return observation, reward, self.done, info

    def snapshot(self) -> bytes:
        import pickle

        state = {
            "battle": deepcopy(self.battle),
            "turn": self.turn,
            "prev_score": self._prev_score,
            "random_state": self.random.getstate(),
            "outcome": self.outcome,
        }
        return pickle.dumps(state)

    def restore(self, snapshot_bytes: bytes) -> dict[str, Any]:
        import pickle

        state = pickle.loads(snapshot_bytes)
        self.sim = self._create_sim(state["battle"])
        self.battle = self.sim.battle
        self._sync_available_actions()
        self.turn = state["turn"]
        self._prev_score = state["prev_score"]
        self.random.setstate(state["random_state"])
        self.outcome = state["outcome"]
        self.done = False
        return self._build_observation()

    # -- battle construction -------------------------------------------------
    def _build_base_battle(self, *, player_team_text: str, opponent_team_text: str) -> Battle:
        battle = Battle(
            battle_tag="battle-local",
            username="agent",
            logger=logging.getLogger("pokemon_showdown_env"),
            gen=self.gen_data.gen,
            save_replays=False,
        )
        battle._format = self.format_id
        battle._player_role = "p1"

        self._player_packed_team = self._apply_team(
            battle, player_team_text, prefix="p1", is_player=True
        )
        self._opponent_packed_team = self._apply_team(
            battle, opponent_team_text, prefix="p2", is_player=False
        )

        battle._opponent_username = "opponent"
        battle._player_username = "agent"

        self._abyssal = AbyssalPlayer(
            battle_format=self.format_id,
            team=self._opponent_packed_team,
            save_replays=False,
            log_level=logging.WARNING,
        )
        return battle

    def _apply_team(self, battle: Battle, team_text: str, *, prefix: str, is_player: bool) -> str:
        mons = Teambuilder.parse_showdown_team(team_text or "")
        if not mons:
            raise ValueError(f"Showdown team for prefix {prefix} is empty.")

        team_dict = getattr(battle, "_team" if is_player else "_opponent_team")

        packed_entries: list[str] = []
        for mon in mons:
            entry = TeambuilderPokemon()
            entry.species = mon.species
            entry.item = mon.item
            entry.ability = mon.ability
            entry.moves = mon.moves[:]
            entry.level = mon.level or 80
            entry.shiny = mon.shiny
            packed_entries.append(entry.formatted)

        packed_team = "]".join(packed_entries)

        for idx, mon in enumerate(mons):
            species = mon.species or mon.nickname or f"Slot{idx+1}"
            ident = f"{prefix}: {species}"
            details = f"{species}, L{mon.level or 80}"
            pokemon = battle.get_pokemon(
                ident,
                force_self_team=is_player,
                force_opp_team=not is_player,
                details=details,
            )

            pokemon._level = mon.level or 80
            if mon.item:
                pokemon.item = mon.item
            if mon.ability:
                pokemon.ability = mon.ability
            for move in mon.moves:
                pokemon._add_move(move)
            pokemon.set_hp_status("300/300")
            pokemon._shiny = mon.shiny
            team_dict[ident] = pokemon

            if idx == 0:
                battle.switch(f"{prefix}a: {species}", details, "300/300")

        return packed_team

    def _create_sim(self, battle: Battle) -> LocalSim:
        return LocalSim(
            battle=battle,
            move_effect=PokechampAssets.move_effect,
            pokemon_move_dict=PokechampAssets.pokemon_move_dict,
            ability_effect=PokechampAssets.ability_effect,
            pokemon_ability_dict=PokechampAssets.pokemon_ability_dict,
            item_effect=PokechampAssets.item_effect,
            pokemon_item_dict=PokechampAssets.pokemon_item_dict,
            gen=self.gen_data,
            _dynamax_disable=True,
            format=self.format_id,
            prompt_translate=None,
        )

    # -- helpers -------------------------------------------------------------
    def _sync_available_actions(self) -> None:
        active = self.battle.active_pokemon
        if active:
            self.battle._available_moves = list(active.available_moves)
        else:
            self.battle._available_moves = []

        switches: list[Pokemon] = []
        for mon in self.battle.team.values():
            if mon and not mon.active and not mon.fainted:
                switches.append(mon)
        self.battle._available_switches = switches

    def _action_to_order(self, action: dict[str, Any]) -> BattleOrder:
        if not action or "action" not in action:
            raise ValueError("Action payload must include 'action' key.")

        action_type = action["action"]
        if action_type == "move":
            moves = self.battle.available_moves
            if not moves:
                raise ValueError("No moves available.")
            index = int(action.get("index", 0))
            if index < 0 or index >= len(moves):
                raise IndexError(f"Move index {index} out of range.")
            move = moves[index]
            target = action.get("target")
            move_target = target if isinstance(target, int) else None
            return BattleOrder(move, move_target=move_target)

        if action_type == "switch":
            switches = self.battle.available_switches
            if not switches:
                raise ValueError("No switches available.")
            index = int(action.get("index", 0))
            if index < 0 or index >= len(switches):
                raise IndexError(f"Switch index {index} out of range.")
            pokemon = switches[index]
            return BattleOrder(pokemon)

        raise ValueError(f"Unsupported action type '{action_type}'.")

    def _opponent_policy(self) -> BattleOrder:
        try:
            order = self._abyssal.choose_move(self.battle)
            if isinstance(order, BattleOrder):
                return order
        except Exception as exc:
            logger.warning("Abyssal opponent failed: %s", exc, exc_info=True)

        opponent = self.battle.opponent_active_pokemon
        agent_active = self.battle.active_pokemon

        moves = opponent.available_moves if opponent else []
        if moves and agent_active:
            def _move_score(move: Move) -> tuple[float, float]:
                try:
                    multiplier = move.type.damage_multiplier(
                        agent_active.type_1 or PokemonType.NORMAL,
                        agent_active.type_2,
                    )
                except Exception:
                    multiplier = 1.0
                base_power = move.base_power or 0
                return multiplier, base_power

            best_move = max(moves, key=_move_score)
            return BattleOrder(best_move)

        bench = [
            mon for mon in self.battle.opponent_team.values() if mon and not mon.active and not mon.fainted
        ]
        if bench:
            return BattleOrder(bench[0])

        fallback_moves = self.battle.available_moves
        if fallback_moves:
            return BattleOrder(fallback_moves[0])
        raise RuntimeError("Opponent policy could not determine a valid action.")

    def _legal_actions(self) -> dict[str, Any]:
        moves = [
            {
                "index": idx,
                "id": move.id,
                "name": move.id,
                "type": str(move.type),
                "base_power": move.base_power or 0,
                "accuracy": move.accuracy or 0,
                "pp": move.current_pp if move.current_pp is not None else move.max_pp,
            }
            for idx, move in enumerate(self.battle.available_moves)
        ]
        switches = [
            {
                "index": idx,
                "species": mon.species,
                "hp_fraction": mon.current_hp_fraction,
                "status": mon.status.name if mon.status else None,
            }
            for idx, mon in enumerate(self.battle.available_switches)
        ]
        return {"moves": moves, "switches": switches}

    def _build_observation(self) -> dict[str, Any]:
        active = self.battle.active_pokemon
        opponent = self.battle.opponent_active_pokemon

        observation = {
            "structured": {
                "scenario": self.scenario["name"],
                "format_id": self.format_id,
                "turn": self.turn,
                "ally_active": self._serialize_pokemon(active),
                "opponent_active": self._serialize_pokemon(opponent),
                "ally_team": self._serialize_team(self.battle.team),
                "opponent_team": self._serialize_team(self.battle.opponent_team),
            },
            "legal_actions": self._legal_actions(),
            "text": self._build_text_summary(active, opponent),
        }
        return observation

    def _serialize_pokemon(self, pokemon: Pokemon | None) -> dict[str, Any] | None:
        if pokemon is None:
            return None
        return {
            "species": pokemon.species,
            "hp_fraction": pokemon.current_hp_fraction,
            "status": pokemon.status.name if pokemon.status else None,
            "moves": [move.id for move in pokemon.moves.values()],
        }

    def _serialize_team(self, team: dict[str, Pokemon]) -> list[dict[str, Any]]:
        bundle: list[dict[str, Any]] = []
        for mon in team.values():
            if mon is None:
                continue
            bundle.append(
                {
                    "species": mon.species,
                    "hp_fraction": mon.current_hp_fraction,
                    "status": mon.status.name if mon.status else None,
                    "active": mon.active,
                    "fainted": mon.fainted,
                }
            )
        return bundle

    def _build_text_summary(self, active: Pokemon | None, opponent: Pokemon | None) -> str:
        player_hp = active.current_hp_fraction if active else 0.0
        opponent_hp = opponent.current_hp_fraction if opponent else 0.0
        player_status = active.status.name if active and active.status else "OK"
        opponent_status = opponent.status.name if opponent and opponent.status else "OK"
        return (
            f"Turn {self.turn}: "
            f"{active.species if active else 'None'} ({player_hp:.1f}%, {player_status}) "
            f"vs {opponent.species if opponent else 'None'} ({opponent_hp:.1f}%, {opponent_status})"
        )

    def _score(self) -> float:
        def team_score(team: dict[str, Pokemon]) -> float:
            return sum((mon.current_hp_fraction or 0.0) for mon in team.values() if mon is not None)

        return team_score(self.battle.team) - team_score(self.battle.opponent_team)

    def _check_finished(self) -> bool:
        ours_alive = any(mon and not mon.fainted for mon in self.battle.team.values())
        opp_alive = any(mon and not mon.fainted for mon in self.battle.opponent_team.values())

        if not ours_alive and not opp_alive:
            self.outcome = 0.0
            return True
        if not ours_alive:
            self.outcome = self.LOSS_PENALTY
            return True
        if not opp_alive:
            self.outcome = self.WIN_REWARD
            return True
        return False


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: PokemonBattleDataset | None = fastapi_request.app.state.get("battle_dataset")
    if dataset is None:
        raise HTTPException(status_code=500, detail="Battle dataset missing from app state.")

    seed = dataset.resolve_seed(request.env.seed)
    scenario = dataset.describe_seed(seed)
    if not scenario["assets_ready"]:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario '{scenario['name']}' is missing required assets. "
            "Ensure pokechamp static files are present.",
        )

    adapter = PokemonShowdownAdapter(
        scenario=scenario,
        repo_root=dataset.repo_root if dataset.repo_root else Path("."),
        seed=request.env.seed,
    )

    obs0 = adapter.reset()
    steps: list[RolloutStep] = [
        RolloutStep(
            obs=obs0,
            tool_calls=[],
            reward=0.0,
            done=False,
            info={"legal_actions": adapter._legal_actions()},
        ),
    ]

    total_reward = 0.0
    done = False

    def _normalise_op(raw: Any) -> dict[str, Any]:
        payload = raw
        if isinstance(raw, dict) and "arguments" in raw:
            try:
                payload = json.loads(raw["arguments"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in arguments field: {exc}") from exc
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unsupported op payload: {payload!r}")

    for op in request.ops or []:
        if done:
            break
        try:
            action_payload = _normalise_op(op)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        obs, reward, done, info = adapter.step(action_payload)
        total_reward += reward
        steps.append(
            RolloutStep(obs=obs, tool_calls=[], reward=reward, done=done, info=info),
        )

    final_obs = steps[-1].obs if steps else obs0
    metrics = RolloutMetrics(
        episode_returns=[total_reward],
        mean_return=total_reward,
        num_steps=max(len(steps) - 1, 0),
        num_episodes=1,
        outcome_score=total_reward,
        details={
            "seed": seed,
            "scenario": scenario["name"],
            "assets_ready": scenario["assets_ready"],
        },
    )

    # Extract inference_url from policy config
    inference_url = (request.policy.config or {}).get("inference_url")
    
    trajectory = RolloutTrajectory(
        env_id="pokemon_showdown",
        policy_id=request.policy.policy_id or "policy",
        steps=steps,
        final={"observation": final_obs, "reward": total_reward, "done": done},
        length=len(steps),
        inference_url=inference_url,  # NEW: Required for trace correlation
    )

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(request.ops or []),
        trace=None,
    )


def build_config() -> TaskAppConfig:
    registry, dataset = _build_dataset_registry()
    base_info = _base_task_info(dataset)
    config = TaskAppConfig(
        app_id="pokemon_showdown",
        name="Pokémon Showdown Task App",
        description="Expose deterministic Pokémon Showdown battles via the Synth AI task framework.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, base_info, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        proxy=ProxyConfig(
            enable_openai=True,
            enable_groq=True,
            system_hint="Respond with legal Pokémon Showdown actions encoded as JSON.",
        ),
        app_state={"battle_dataset": dataset},
        require_api_key=True,
        expose_debug_env=True,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="pokemon_showdown",
        description="Pokémon Showdown (Track 1) task app skeleton.",
        config_factory=build_config,
        aliases=("pokemon_battle", "pokemon_track1"),
        env_files=(),
        modal=ModalDeploymentConfig(
            app_name="pokemon-showdown-task-app",
            python_version="3.11",
            pip_packages=("horizons-ai",),
            extra_local_dirs=(
                ("repo", "/opt/synth_ai_repo"),
                ("pokechamp", "/external/pokechamp"),
                ("pokemon_showdown", "/external/pokemon-showdown"),
            ),
            secret_names=("ENVIRONMENT_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"),
            timeout=900,
            memory=8192,
            cpu=4.0,
        ),
    )
)


__all__ = ["build_config"]
