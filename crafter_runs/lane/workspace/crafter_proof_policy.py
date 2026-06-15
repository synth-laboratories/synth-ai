from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from crafter import constants as crafter_constants

ACTION_NAMES = list(crafter_constants.actions)
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}

MATERIALS = list(crafter_constants.materials)
MATERIAL_TO_ID = {name: index + 1 for index, name in enumerate(MATERIALS)}
ID_TO_MATERIAL = {index: name for name, index in MATERIAL_TO_ID.items()}

PLAYER_ID = len(MATERIALS) + 1
OBJECT_IDS = {
    "player": PLAYER_ID,
    "cow": PLAYER_ID + 1,
    "zombie": PLAYER_ID + 2,
    "skeleton": PLAYER_ID + 3,
    "arrow": PLAYER_ID + 4,
    "plant": PLAYER_ID + 5,
}

WALKABLE_IDS = {
    MATERIAL_TO_ID["grass"],
    MATERIAL_TO_ID["path"],
    MATERIAL_TO_ID["sand"],
}
DELTAS = {
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up": (0, -1),
    "move_down": (0, 1),
}
DELTA_TO_ACTION = {delta: action for action, delta in DELTAS.items()}
CARDINAL_DELTAS = list(DELTA_TO_ACTION)

TREE_ID = MATERIAL_TO_ID["tree"]
WATER_ID = MATERIAL_TO_ID["water"]
STONE_ID = MATERIAL_TO_ID["stone"]
COAL_ID = MATERIAL_TO_ID["coal"]
IRON_ID = MATERIAL_TO_ID["iron"]
DIAMOND_ID = MATERIAL_TO_ID["diamond"]
TABLE_ID = MATERIAL_TO_ID["table"]
FURNACE_ID = MATERIAL_TO_ID["furnace"]


def _inside(pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < 64 and 0 <= pos[1] < 64


def _to_tuple(pos: Any) -> tuple[int, int]:
    if isinstance(pos, np.ndarray):
        return int(pos[0]), int(pos[1])
    return int(pos[0]), int(pos[1])


def _tile_id(semantic: np.ndarray | None, pos: tuple[int, int]) -> int | None:
    if semantic is None or not _inside(pos):
        return None
    return int(semantic[pos])


def _is_walkable(tile_id: int | None) -> bool:
    return tile_id in WALKABLE_IDS


def _is_occupied(tile_id: int | None) -> bool:
    return tile_id is not None and tile_id >= PLAYER_ID and tile_id != PLAYER_ID


def _is_passable(tile_id: int | None, *, allow_lava: bool = False) -> bool:
    if tile_id is None:
        return False
    if tile_id in WALKABLE_IDS:
        return True
    return bool(allow_lava and tile_id == MATERIAL_TO_ID["lava"])


def _bfs(
    semantic: np.ndarray,
    start: tuple[int, int],
    *,
    allow_lava: bool = False,
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[tuple[int, int], int]]:
    queue: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while queue:
        pos = queue.popleft()
        for delta in CARDINAL_DELTAS:
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if nxt in dist or not _inside(nxt):
                continue
            if nxt != start:
                tile_id = _tile_id(semantic, nxt)
                if _is_occupied(tile_id):
                    continue
                if not _is_passable(tile_id, allow_lava=allow_lava):
                    continue
            prev[nxt] = pos
            dist[nxt] = dist[pos] + 1
            queue.append(nxt)
    return prev, dist


def _first_step(
    prev: dict[tuple[int, int], tuple[int, int] | None],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, int] | None:
    if goal not in prev:
        return None
    pos = goal
    while prev[pos] is not None and prev[pos] != start:
        pos = prev[pos]
    parent = prev[pos]
    if parent is None:
        return None
    return pos[0] - parent[0], pos[1] - parent[1]


def _all_positions(semantic: np.ndarray, ids: set[int]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for tile_id in ids:
        xs, ys = np.where(semantic == tile_id)
        hits.extend((int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist(), strict=False))
    return hits


def _action_for_delta(delta: tuple[int, int]) -> int:
    return ACTION_TO_ID[DELTA_TO_ACTION[delta]]


def _step_toward(
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    goal: tuple[int, int],
    *,
    allow_lava: bool = False,
) -> int | None:
    if semantic is None or start is None:
        return None
    prev, _dist = _bfs(semantic, start, allow_lava=allow_lava)
    delta = _first_step(prev, start, goal)
    if delta is None:
        return None
    return _action_for_delta(delta)


def _collect_plan(
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    target_ids: set[int],
    *,
    action_name: str,
    facing: tuple[int, int],
    target_positions: list[tuple[int, int]] | None = None,
    allow_lava: bool = False,
) -> int | None:
    if semantic is None or start is None:
        return None
    prev, dist = _bfs(semantic, start, allow_lava=allow_lava)
    best: tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]] | None = None
    targets = (
        target_positions if target_positions is not None else _all_positions(semantic, target_ids)
    )
    for target in targets:
        for delta in CARDINAL_DELTAS:
            interact = (target[0] - delta[0], target[1] - delta[1])
            approach = (target[0] - 2 * delta[0], target[1] - 2 * delta[1])
            if not _inside(interact) or not _inside(approach):
                continue
            if not _is_walkable(_tile_id(semantic, interact)):
                continue
            if not _is_walkable(_tile_id(semantic, approach)):
                continue
            if (
                _tile_id(semantic, interact) == PLAYER_ID
                or _tile_id(semantic, approach) == PLAYER_ID
            ):
                continue
            if approach not in dist:
                continue
            candidate = (dist[approach], target, interact, delta)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _dist, target, interact, delta = best
    if start == interact and facing == delta:
        return ACTION_TO_ID[action_name]
    if start == (target[0] - 2 * delta[0], target[1] - 2 * delta[1]):
        return _action_for_delta(delta)
    return _step_toward(semantic, start, (target[0] - 2 * delta[0], target[1] - 2 * delta[1]))


def _craft_plan(
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    station_ids: set[int],
    *,
    action_name: str,
) -> int | None:
    if semantic is None or start is None:
        return None
    prev, dist = _bfs(semantic, start)
    station_positions = _all_positions(semantic, station_ids)
    if not station_positions:
        return None
    goals: list[tuple[int, int]] = []
    for x in range(64):
        for y in range(64):
            goal = (x, y)
            if not _is_walkable(_tile_id(semantic, goal)):
                continue
            if goal not in dist:
                continue
            if any(
                abs(goal[0] - station[0]) + abs(goal[1] - station[1]) != 1
                for station in station_positions
            ):
                continue
            goals.append(goal)
    if not goals:
        return None
    goal = min(goals, key=lambda pos: dist[pos])
    if start == goal:
        return ACTION_TO_ID[action_name]
    return _step_toward(semantic, start, goal)


def _furnace_targets(semantic: np.ndarray | None) -> list[tuple[int, int]]:
    if semantic is None:
        return []
    targets: list[tuple[int, int]] = []
    for table in _all_positions(semantic, {TABLE_ID}):
        for dx in (-1, 1):
            for dy in (-1, 1):
                furnace = (table[0] + dx, table[1] + dy)
                if not _inside(furnace):
                    continue
                if not _is_walkable(_tile_id(semantic, furnace)):
                    continue
                targets.append(furnace)
    # Deduplicate while preserving order.
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for pos in targets:
        if pos in seen:
            continue
        seen.add(pos)
        unique.append(pos)
    return unique


def _choose_interact_objective(
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    target_ids: set[int],
    *,
    action_name: str,
    target_positions: list[tuple[int, int]] | None = None,
    allow_lava: bool = False,
) -> dict[str, Any] | None:
    if semantic is None or start is None:
        return None
    prev, dist = _bfs(semantic, start, allow_lava=allow_lava)
    targets = (
        target_positions if target_positions is not None else _all_positions(semantic, target_ids)
    )
    best: tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = (
        None
    )
    for target in targets:
        for delta in CARDINAL_DELTAS:
            interact = (target[0] - delta[0], target[1] - delta[1])
            approach = (target[0] - 2 * delta[0], target[1] - 2 * delta[1])
            if not _inside(interact) or not _inside(approach):
                continue
            if not _is_walkable(_tile_id(semantic, interact)):
                continue
            if not _is_walkable(_tile_id(semantic, approach)):
                continue
            if (
                _tile_id(semantic, interact) == PLAYER_ID
                or _tile_id(semantic, approach) == PLAYER_ID
            ):
                continue
            if approach not in dist:
                continue
            candidate = (dist[approach], target, interact, approach, delta)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _distance, target, interact, approach, delta = best
    return {
        "kind": "interact",
        "action_name": action_name,
        "target": target,
        "interact": interact,
        "approach": approach,
        "delta": delta,
        "target_id": _tile_id(semantic, target),
    }


def _follow_interact_objective(
    objective: dict[str, Any],
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    *,
    facing: tuple[int, int],
) -> int | None:
    if semantic is None or start is None:
        return None
    target = objective.get("target")
    interact = objective.get("interact")
    approach = objective.get("approach")
    delta = objective.get("delta")
    action_name = objective.get("action_name")
    target_id = objective.get("target_id")
    if (
        not isinstance(target, tuple)
        or not isinstance(interact, tuple)
        or not isinstance(approach, tuple)
    ):
        return None
    if target_id is not None and _tile_id(semantic, target) != int(target_id):
        return None
    if start == interact and facing == delta:
        return ACTION_TO_ID[str(action_name)]
    if start == approach:
        return _action_for_delta(delta)
    return _step_toward(semantic, start, approach)


def _choose_craft_objective(
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
    station_ids: set[int],
    *,
    action_name: str,
) -> dict[str, Any] | None:
    if semantic is None or start is None:
        return None
    prev, dist = _bfs(semantic, start)
    stations = _all_positions(semantic, station_ids)
    if not stations:
        return None
    goals: list[tuple[int, int]] = []
    for x in range(64):
        for y in range(64):
            goal = (x, y)
            if not _is_walkable(_tile_id(semantic, goal)):
                continue
            if goal not in dist:
                continue
            if any(
                abs(goal[0] - station[0]) + abs(goal[1] - station[1]) != 1 for station in stations
            ):
                continue
            goals.append(goal)
    if not goals:
        return None
    goal = min(goals, key=lambda pos: dist[pos])
    return {"kind": "craft", "action_name": action_name, "goal": goal, "stations": tuple(stations)}


def _follow_craft_objective(
    objective: dict[str, Any],
    semantic: np.ndarray | None,
    start: tuple[int, int] | None,
) -> int | None:
    if semantic is None or start is None:
        return None
    goal = objective.get("goal")
    action_name = objective.get("action_name")
    if not isinstance(goal, tuple):
        return None
    stations = objective.get("stations") or ()
    if any(
        abs(goal[0] - int(station[0])) + abs(goal[1] - int(station[1])) != 1 for station in stations
    ):
        return None
    if start == goal:
        return ACTION_TO_ID[str(action_name)]
    return _step_toward(semantic, start, goal)


class Policy:
    def __init__(self) -> None:
        self._facing = (1, 0)
        self._bootstrapped = False
        self._objective: dict[str, Any] | None = None

    def _inventory(self, observation: dict[str, Any]) -> dict[str, int]:
        inventory = observation.get("inventory")
        return dict(inventory) if isinstance(inventory, dict) else {}

    def _achievements(self, observation: dict[str, Any]) -> dict[str, bool]:
        achievements = observation.get("achievements_status")
        return dict(achievements) if isinstance(achievements, dict) else {}

    def _sleep_ok(self, inventory: dict[str, int], health: float, step: int) -> bool:
        if health <= 4 and inventory.get("food", 0) > 0 and inventory.get("drink", 0) > 0:
            return True
        return (
            inventory.get("energy", 0) <= 1
            and inventory.get("food", 0) > 0
            and inventory.get("drink", 0) > 0
            and step % 3 == 0
        )

    def _update_facing(self, action_id: int) -> None:
        for name, delta in DELTAS.items():
            if ACTION_TO_ID[name] == action_id:
                self._facing = delta
                return

    def _objective_valid(
        self, objective: dict[str, Any] | None, semantic: np.ndarray | None
    ) -> bool:
        if objective is None or semantic is None:
            return False
        kind = objective.get("kind")
        if kind == "interact":
            target = objective.get("target")
            target_id = objective.get("target_id")
            if not isinstance(target, tuple):
                return False
            return not (target_id is not None and _tile_id(semantic, target) != int(target_id))
        if kind == "craft":
            stations = objective.get("stations") or ()
            if not stations:
                return False
            for station in stations:
                pos = (int(station[0]), int(station[1]))
                if _tile_id(semantic, pos) not in {TABLE_ID, FURNACE_ID}:
                    return False
            return True
        return False

    def _select_objective(
        self,
        inventory: dict[str, int],
        achievements: dict[str, bool],
        semantic: np.ndarray | None,
        start: tuple[int, int] | None,
    ) -> dict[str, Any] | None:
        if semantic is None or start is None:
            return None

        if inventory.get("drink", 0) <= 4:
            objective = _choose_interact_objective(
                semantic,
                start,
                {WATER_ID},
                action_name="do",
            )
            if objective is not None:
                return objective

        if inventory.get("food", 0) <= 4:
            objective = _choose_interact_objective(
                semantic,
                start,
                {OBJECT_IDS["cow"]},
                action_name="do",
            )
            if objective is not None:
                return objective

        if not achievements.get("place_table"):
            if inventory.get("wood", 0) >= 2:
                objective = _choose_interact_objective(
                    semantic,
                    start,
                    {MATERIAL_TO_ID["grass"], MATERIAL_TO_ID["path"], MATERIAL_TO_ID["sand"]},
                    action_name="place_table",
                )
                if objective is not None:
                    return objective
            return _choose_interact_objective(
                semantic,
                start,
                {TREE_ID},
                action_name="do",
            )

        if not achievements.get("make_wood_pickaxe"):
            if inventory.get("wood", 0) >= 1:
                objective = _choose_craft_objective(
                    semantic,
                    start,
                    {TABLE_ID},
                    action_name="make_wood_pickaxe",
                )
                if objective is not None:
                    return objective
            return _choose_interact_objective(
                semantic,
                start,
                {TREE_ID},
                action_name="do",
            )

        if not achievements.get("make_stone_pickaxe"):
            if inventory.get("wood", 0) < 1:
                return _choose_interact_objective(
                    semantic,
                    start,
                    {TREE_ID},
                    action_name="do",
                )
            if inventory.get("stone", 0) >= 1:
                objective = _choose_craft_objective(
                    semantic,
                    start,
                    {TABLE_ID},
                    action_name="make_stone_pickaxe",
                )
                if objective is not None:
                    return objective
            return _choose_interact_objective(
                semantic,
                start,
                {STONE_ID},
                action_name="do",
            )

        if not achievements.get("place_furnace"):
            if inventory.get("stone", 0) >= 4:
                objective = _choose_interact_objective(
                    semantic,
                    start,
                    {MATERIAL_TO_ID["grass"], MATERIAL_TO_ID["path"], MATERIAL_TO_ID["sand"]},
                    action_name="place_furnace",
                    target_positions=_furnace_targets(semantic),
                )
                if objective is not None:
                    return objective
            return _choose_interact_objective(
                semantic,
                start,
                {STONE_ID},
                action_name="do",
            )

        if not achievements.get("make_iron_pickaxe"):
            if inventory.get("wood", 0) < 1:
                return _choose_interact_objective(
                    semantic,
                    start,
                    {TREE_ID},
                    action_name="do",
                )
            if inventory.get("coal", 0) < 1:
                return _choose_interact_objective(
                    semantic,
                    start,
                    {COAL_ID},
                    action_name="do",
                )
            if inventory.get("iron", 0) < 1:
                return _choose_interact_objective(
                    semantic,
                    start,
                    {IRON_ID},
                    action_name="do",
                )
            objective = _choose_craft_objective(
                semantic,
                start,
                {TABLE_ID, FURNACE_ID},
                action_name="make_iron_pickaxe",
            )
            if objective is not None:
                return objective

        if achievements.get("make_iron_pickaxe"):
            objective = _choose_interact_objective(
                semantic,
                start,
                {DIAMOND_ID},
                action_name="do",
            )
            if objective is not None:
                return objective

        return None

    def act(self, observation: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        info = info if isinstance(info, dict) else {}
        inventory = self._inventory(observation)
        achievements = self._achievements(observation)
        health = float(observation.get("health") or inventory.get("health") or 0)
        step = int(observation.get("step_count") or 0)
        semantic = info.get("semantic")
        semantic = semantic if isinstance(semantic, np.ndarray) else None
        player_pos = info.get("player_pos")
        start = _to_tuple(player_pos) if player_pos is not None else None

        if not self._bootstrapped:
            self._bootstrapped = True
            return ACTION_TO_ID["move_right"]

        if self._sleep_ok(inventory, health, step):
            self._objective = None
            return ACTION_TO_ID["sleep"]

        objective = self._objective
        if not self._objective_valid(objective, semantic):
            objective = None

        preferred = self._select_objective(inventory, achievements, semantic, start)
        if preferred is not None:
            if (
                objective is None
                or objective.get("kind") != preferred.get("kind")
                or objective.get("action_name") != preferred.get("action_name")
            ):
                objective = preferred
        else:
            objective = None

        self._objective = objective
        if objective is not None:
            if objective.get("kind") == "interact":
                action = _follow_interact_objective(objective, semantic, start, facing=self._facing)
            else:
                action = _follow_craft_objective(objective, semantic, start)
            if action is not None:
                self._update_facing(action)
                return action
            self._objective = None

        roam = ["move_right", "move_down", "move_left", "move_up"]
        action = ACTION_TO_ID[roam[step % len(roam)]]
        self._update_facing(action)
        return action
