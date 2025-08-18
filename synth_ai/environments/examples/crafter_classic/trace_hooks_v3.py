"""
Trace hooks for Crafter Classic environment - v3 version.
Updated to use the v3 tracing architecture with async support.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Set

from synth_ai.tracing_v3.abstractions import BaseEvent, EnvironmentEvent
from synth_ai.tracing_v3.hooks import HookManager

# Achievement categorization by difficulty
EASY_ACHIEVEMENTS = {
    "collect_wood",
    "collect_stone",
    "collect_sapling",
    "collect_drink",
    "place_stone",
    "place_table",
    "wake_up",
    "eat_plant",
}

MEDIUM_ACHIEVEMENTS = {
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "collect_coal",
    "collect_iron",
    "eat_cow",
}

HARD_ACHIEVEMENTS = {
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_iron_pickaxe",
    "make_iron_sword",
    "collect_diamond",
    "defeat_skeleton",
    "defeat_zombie",
}


async def check_easy_achievements(event_obj: BaseEvent, **kwargs) -> Optional[Dict[str, Any]]:
    """Hook that fires when an easy achievement is unlocked."""
    # Only process EnvironmentEvents
    if not isinstance(event_obj, EnvironmentEvent):
        return None

    # Get achievements before and after
    before_achievements = {}
    after_achievements = {}

    state_before = event_obj.system_state_before
    state_after = event_obj.system_state_after

    if isinstance(state_before, dict):
        before_achievements = state_before.get("public_state", {}).get("achievements_status", {})
    if isinstance(state_after, dict):
        after_achievements = state_after.get("public_state", {}).get("achievements_status", {})

    # Find new easy achievements
    new_easy_achievements = []
    for achievement, status in after_achievements.items():
        if status and not before_achievements.get(achievement, False):
            if achievement in EASY_ACHIEVEMENTS:
                new_easy_achievements.append(achievement)

    if new_easy_achievements:
        # print(f"🎯 Easy achievement(s) unlocked: {', '.join(new_easy_achievements)}")  # Disabled for clean output
        return {
            "achievements": new_easy_achievements,
            "difficulty": "easy",
            "timestamp": datetime.now().isoformat(),
        }
    return None


async def check_medium_achievements(event_obj: BaseEvent, **kwargs) -> Optional[Dict[str, Any]]:
    """Hook that fires when a medium achievement is unlocked."""
    # Only process EnvironmentEvents
    if not isinstance(event_obj, EnvironmentEvent):
        return None

    # Get achievements before and after
    before_achievements = {}
    after_achievements = {}

    state_before = event_obj.system_state_before
    state_after = event_obj.system_state_after

    if isinstance(state_before, dict):
        before_achievements = state_before.get("public_state", {}).get("achievements_status", {})
    if isinstance(state_after, dict):
        after_achievements = state_after.get("public_state", {}).get("achievements_status", {})

    # Find new medium achievements
    new_medium_achievements = []
    for achievement, status in after_achievements.items():
        if status and not before_achievements.get(achievement, False):
            if achievement in MEDIUM_ACHIEVEMENTS:
                new_medium_achievements.append(achievement)

    if new_medium_achievements:
        # print(f"⭐ Medium achievement(s) unlocked: {', '.join(new_medium_achievements)}")  # Disabled for clean output
        return {
            "achievements": new_medium_achievements,
            "difficulty": "medium",
            "timestamp": datetime.now().isoformat(),
        }
    return None


async def check_hard_achievements(event_obj: BaseEvent, **kwargs) -> Optional[Dict[str, Any]]:
    """Hook that fires when a hard achievement is unlocked."""
    # Only process EnvironmentEvents
    if not isinstance(event_obj, EnvironmentEvent):
        return None

    # Get achievements before and after
    before_achievements = {}
    after_achievements = {}

    state_before = event_obj.system_state_before
    state_after = event_obj.system_state_after

    if isinstance(state_before, dict):
        before_achievements = state_before.get("public_state", {}).get("achievements_status", {})
    if isinstance(state_after, dict):
        after_achievements = state_after.get("public_state", {}).get("achievements_status", {})

    # Find new hard achievements
    new_hard_achievements = []
    for achievement, status in after_achievements.items():
        if status and not before_achievements.get(achievement, False):
            if achievement in HARD_ACHIEVEMENTS:
                new_hard_achievements.append(achievement)

    if new_hard_achievements:
        # print(f"🏆 Hard achievement(s) unlocked: {', '.join(new_hard_achievements)}")  # Disabled for clean output
        return {
            "achievements": new_hard_achievements,
            "difficulty": "hard",
            "timestamp": datetime.now().isoformat(),
        }
    return None


async def log_invalid_actions(event_obj: BaseEvent, **kwargs) -> Optional[Dict[str, Any]]:
    """Hook that logs invalid actions."""
    from synth_ai.tracing_v3.abstractions import RuntimeEvent

    # Only process RuntimeEvents
    if not isinstance(event_obj, RuntimeEvent):
        return None

    # Check if action was invalid
    if event_obj.metadata.get("valid") is False:
        action_name = event_obj.metadata.get("action_name", "unknown")
        # print(f"⚠️  Invalid action attempted: {action_name}")  # Disabled for clean output
        return {
            "action": action_name,
            "valid": False,
            "timestamp": datetime.now().isoformat(),
        }
    return None


async def track_reward_milestones(event_obj: BaseEvent, **kwargs) -> Optional[Dict[str, Any]]:
    """Hook that tracks significant reward events."""
    # Only process EnvironmentEvents
    if not isinstance(event_obj, EnvironmentEvent):
        return None

    reward = event_obj.reward
    if reward and reward >= 1.0:  # Significant positive reward
        # print(f"💰 High reward received: {reward}")  # Disabled for clean output
        return {
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }
    return None


# Create the global CRAFTER_HOOKS instance
CRAFTER_HOOKS = HookManager()

# Register all hooks
CRAFTER_HOOKS.register(
    "event_recorded",
    check_easy_achievements,
    name="easy_achievements",
    priority=10,
    event_types=["environment"],
)

CRAFTER_HOOKS.register(
    "event_recorded",
    check_medium_achievements,
    name="medium_achievements",
    priority=10,
    event_types=["environment"],
)

CRAFTER_HOOKS.register(
    "event_recorded",
    check_hard_achievements,
    name="hard_achievements",
    priority=10,
    event_types=["environment"],
)

CRAFTER_HOOKS.register(
    "event_recorded",
    log_invalid_actions,
    name="invalid_actions",
    priority=5,
    event_types=["runtime"],
)

CRAFTER_HOOKS.register(
    "event_recorded",
    track_reward_milestones,
    name="reward_milestones",
    priority=5,
    event_types=["environment"],
)
