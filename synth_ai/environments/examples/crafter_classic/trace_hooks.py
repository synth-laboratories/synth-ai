"""
Trace hooks for Crafter Classic environment.
"""
from typing import Any, Optional, Dict, Set
from datetime import datetime

from synth_ai.tracing_v2.hooks import TraceStateHook, HookResult
try:
    from synth_ai.tracing_v2.abstractions import SessionEvent, CAISEvent, EnvironmentEvent
except ImportError:
    # Fallback to any SessionEvent type
    SessionEvent = object
    CAISEvent = None
    EnvironmentEvent = None


# Achievement categorization by difficulty
EASY_ACHIEVEMENTS = {
    'collect_wood', 'collect_stone', 'collect_sapling', 'collect_drink',
    'place_stone', 'place_table', 'wake_up', 'eat_plant'
}

MEDIUM_ACHIEVEMENTS = {
    'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
    'collect_coal', 'collect_iron', 'eat_cow'
}

HARD_ACHIEVEMENTS = {
    'make_stone_pickaxe', 'make_stone_sword', 'make_iron_pickaxe', 
    'make_iron_sword', 'collect_diamond', 'defeat_skeleton', 'defeat_zombie'
}


class EasyAchievementHook(TraceStateHook):
    """Hook that fires when an easy achievement is unlocked."""
    
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        # Only process EnvironmentEvents
        if not (hasattr(event, '__class__') and event.__class__.__name__ == 'EnvironmentEvent'):
            return None
        return super().check(event)
    
    def analyze_state(self, state_before: Any, state_after: Any, event: SessionEvent) -> Optional[HookResult]:
        # Get achievements before and after
        before_achievements = {}
        after_achievements = {}
        
        if isinstance(state_before, dict):
            before_achievements = state_before.get('public_state', {}).get('achievements_status', {})
        if isinstance(state_after, dict):
            after_achievements = state_after.get('public_state', {}).get('achievements_status', {})
            
        # Find new easy achievements
        new_easy_achievements = []
        for achievement, status in after_achievements.items():
            if status and not before_achievements.get(achievement, False):
                if achievement in EASY_ACHIEVEMENTS:
                    new_easy_achievements.append(achievement)
                    
        if new_easy_achievements:
            return HookResult(
                hook_name="easy_achievement",
                description=f"Easy achievement(s) unlocked: {', '.join(new_easy_achievements)}",
                data={
                    "achievements": new_easy_achievements,
                    "difficulty": "easy"
                },
                timestamp=datetime.now().isoformat(),
                code="E",  # E for Easy
                priority=3  # All achievements have same base priority
            )
        return None


class MediumAchievementHook(TraceStateHook):
    """Hook that fires when a medium achievement is unlocked."""
    
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        # Only process EnvironmentEvents
        if not (hasattr(event, '__class__') and event.__class__.__name__ == 'EnvironmentEvent'):
            return None
        return super().check(event)
    
    def analyze_state(self, state_before: Any, state_after: Any, event: SessionEvent) -> Optional[HookResult]:
        # Get achievements before and after
        before_achievements = {}
        after_achievements = {}
        
        if isinstance(state_before, dict):
            before_achievements = state_before.get('public_state', {}).get('achievements_status', {})
        if isinstance(state_after, dict):
            after_achievements = state_after.get('public_state', {}).get('achievements_status', {})
            
        # Find new medium achievements
        new_medium_achievements = []
        for achievement, status in after_achievements.items():
            if status and not before_achievements.get(achievement, False):
                if achievement in MEDIUM_ACHIEVEMENTS:
                    new_medium_achievements.append(achievement)
                    
        if new_medium_achievements:
            return HookResult(
                hook_name="medium_achievement",
                description=f"Medium achievement(s) unlocked: {', '.join(new_medium_achievements)}",
                data={
                    "achievements": new_medium_achievements,
                    "difficulty": "medium"
                },
                timestamp=datetime.now().isoformat(),
                code="M",  # M for Medium
                priority=4  # Slightly higher priority than easy
            )
        return None


class HardAchievementHook(TraceStateHook):
    """Hook that fires when a hard achievement is unlocked."""
    
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        # Only process EnvironmentEvents
        if not (hasattr(event, '__class__') and event.__class__.__name__ == 'EnvironmentEvent'):
            return None
        return super().check(event)
    
    def analyze_state(self, state_before: Any, state_after: Any, event: SessionEvent) -> Optional[HookResult]:
        # Get achievements before and after
        before_achievements = {}
        after_achievements = {}
        
        if isinstance(state_before, dict):
            before_achievements = state_before.get('public_state', {}).get('achievements_status', {})
        if isinstance(state_after, dict):
            after_achievements = state_after.get('public_state', {}).get('achievements_status', {})
            
        # Find new hard achievements
        new_hard_achievements = []
        for achievement, status in after_achievements.items():
            if status and not before_achievements.get(achievement, False):
                if achievement in HARD_ACHIEVEMENTS:
                    new_hard_achievements.append(achievement)
                    
        if new_hard_achievements:
            return HookResult(
                hook_name="hard_achievement",
                description=f"Hard achievement(s) unlocked: {', '.join(new_hard_achievements)}",
                data={
                    "achievements": new_hard_achievements,
                    "difficulty": "hard"
                },
                timestamp=datetime.now().isoformat(),
                code="H",  # H for Hard
                priority=5  # Highest priority
            )
        return None


class InvalidActionHook(TraceStateHook):
    """Hook that detects when actions don't have their expected effect."""
    
    def __init__(self):
        super().__init__()
        # Track invalid actions by type
        self.invalid_actions = {}
        # Track total actions by type
        self.total_actions = {}
        
    def check(self, event: SessionEvent) -> Optional[HookResult]:
        # Only process RuntimeEvents (which contain the action taken)
        if not (hasattr(event, '__class__') and event.__class__.__name__ == 'RuntimeEvent'):
            return None
        return super().check(event)
    
    def analyze_state(self, state_before: Any, state_after: Any, event: SessionEvent) -> Optional[HookResult]:
        # Get the action from the event
        if not hasattr(event, 'actions') or not event.actions:
            return None
            
        action = event.actions[0] if isinstance(event.actions, list) else event.actions
        
        # Map action index to name
        action_names = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 
                       'do', 'sleep', 'place_stone', 'place_table', 'place_furnace', 
                       'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 
                       'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 
                       'make_iron_sword']
        
        if action >= len(action_names):
            return None
            
        action_name = action_names[action]
        
        # Track total actions
        if action_name not in self.total_actions:
            self.total_actions[action_name] = 0
        self.total_actions[action_name] += 1
        
        # Get states
        before_obs = state_before.get('observation', {}) if isinstance(state_before, dict) else {}
        after_obs = state_after.get('observation', {}) if isinstance(state_after, dict) else {}
        
        # Check if action had expected effect
        is_invalid = False
        reason = ""
        
        # Movement actions
        if action_name in ['move_left', 'move_right', 'move_up', 'move_down']:
            before_pos = before_obs.get('player_position', [0, 0])
            after_pos = after_obs.get('player_position', [0, 0])
            
            expected_change = {
                'move_left': [-1, 0],
                'move_right': [1, 0],
                'move_up': [0, -1],
                'move_down': [0, 1]
            }
            
            expected = expected_change.get(action_name, [0, 0])
            actual_change = [after_pos[0] - before_pos[0], after_pos[1] - before_pos[1]]
            
            if actual_change != expected:
                is_invalid = True
                reason = f"Expected position change {expected}, got {actual_change}"
        
        # Do action (collect/attack)
        elif action_name == 'do':
            # Check if any inventory item increased or achievement unlocked
            before_inv = before_obs.get('inventory', {})
            after_inv = after_obs.get('inventory', {})
            before_achievements = before_obs.get('achievements_status', {})
            after_achievements = after_obs.get('achievements_status', {})
            
            # Check for any positive change
            inventory_increased = any(after_inv.get(item, 0) > before_inv.get(item, 0) 
                                    for item in after_inv)
            new_achievement = any(after_achievements.get(ach, False) and not before_achievements.get(ach, False)
                                for ach in after_achievements)
            
            if not inventory_increased and not new_achievement:
                is_invalid = True
                reason = "No resource collected or achievement unlocked"
        
        # Sleep action
        elif action_name == 'sleep':
            # Should restore energy
            before_inv = before_obs.get('inventory', {})
            after_inv = after_obs.get('inventory', {})
            before_energy = before_inv.get('energy', 0)
            after_energy = after_inv.get('energy', 0)
            
            if after_energy <= before_energy:
                is_invalid = True
                reason = f"Energy not restored (before: {before_energy}, after: {after_energy})"
        
        # Place actions
        elif action_name.startswith('place_'):
            resource = action_name.replace('place_', '')
            before_inv = before_obs.get('inventory', {})
            after_inv = after_obs.get('inventory', {})
            
            # Check if the resource was consumed
            resource_consumed = False
            if resource == 'table':
                # Placing table consumes wood
                before_wood = before_inv.get('wood', 0)
                after_wood = after_inv.get('wood', 0)
                resource_consumed = after_wood < before_wood
            elif resource == 'furnace':
                # Placing furnace consumes stone
                before_stone = before_inv.get('stone', 0)
                after_stone = after_inv.get('stone', 0)
                resource_consumed = after_stone < before_stone
            elif resource == 'stone':
                # Placing stone consumes stone
                before_stone = before_inv.get('stone', 0)
                after_stone = after_inv.get('stone', 0)
                resource_consumed = after_stone < before_stone
            elif resource == 'plant':
                # Placing plant consumes sapling
                before_sapling = before_inv.get('sapling', 0)
                after_sapling = after_inv.get('sapling', 0)
                resource_consumed = after_sapling < before_sapling
            
            if not resource_consumed:
                is_invalid = True
                reason = f"Resource not consumed when placing {resource}"
        
        # Make actions
        elif action_name.startswith('make_'):
            before_inv = before_obs.get('inventory', {})
            after_inv = after_obs.get('inventory', {})
            
            # Check if the item was created
            item_name = action_name.replace('make_', '')
            before_count = before_inv.get(item_name, 0)
            after_count = after_inv.get(item_name, 0)
            
            if after_count <= before_count:
                is_invalid = True
                reason = f"{item_name} not created"
        
        # Track invalid actions
        if is_invalid:
            if action_name not in self.invalid_actions:
                self.invalid_actions[action_name] = 0
            self.invalid_actions[action_name] += 1
            
            return HookResult(
                hook_name="invalid_action",
                description=f"Invalid {action_name}: {reason}",
                data={
                    "action": action_name,
                    "reason": reason,
                    "before_state": before_obs,
                    "after_state": after_obs
                },
                timestamp=datetime.now().isoformat(),
                code="X",  # X for invalid
                priority=2  # Medium priority
            )
        
        return None


# Collection of all Crafter hooks
CRAFTER_HOOKS = [
    EasyAchievementHook(),
    MediumAchievementHook(),
    HardAchievementHook(),
    InvalidActionHook()
]