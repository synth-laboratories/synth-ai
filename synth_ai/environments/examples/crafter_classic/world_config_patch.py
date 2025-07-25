"""
Monkey patch to add configurable world generation to crafter.Env
Allows JSON-based configuration of world difficulty and resource abundance.
"""

import json
import os
from typing import Dict, Any, Optional
import crafter
from crafter import worldgen
import numpy as np

print("[PATCH] Attempting to apply Crafter world configuration patch...")

# Default world configurations
WORLD_CONFIGS = {
    "easy": {
        "name": "Easy Mode",
        "description": "Abundant resources, fewer enemies",
        "materials": {
            "coal": {"threshold": -0.2, "probability": 0.25},      # More coal (was 0, 0.15)
            "iron": {"threshold": 0.2, "probability": 0.35},       # More iron (was 0.4, 0.25)
            "diamond": {"threshold": 0.15, "probability": 0.02},   # More diamonds (was 0.18, 0.006)
            "tree": {"threshold": -0.2, "probability": 0.35},      # More trees (was 0, 0.2)
            "lava": {"threshold": 0.5, "probability": 1.0}         # Less lava area (was 0.35)
        },
        "initial_spawns": {
            "cow": {"min_distance": 2, "probability": 0.03},       # More cows closer (was 3, 0.015)
            "zombie": {"min_distance": 15, "probability": 0.003},  # Fewer zombies farther (was 10, 0.007)
            "skeleton": {"probability": 0.01}                      # Fewer skeletons (was 0.05)
        },
        "dynamic_spawns": {
            "zombie": {
                "count_range": [0, 1.5],      # Fewer zombies (was 0-3.5)
                "min_distance": 10,           # Farther away (was 6)
                "spawn_prob": 0.1,            # Less likely (was 0.3)
                "despawn_prob": 0.6           # More likely to despawn (was 0.4)
            },
            "skeleton": {
                "count_range": [0, 0.5],      # Fewer skeletons (was 0-2)
                "min_distance": 12,           # Farther away (was 7)
                "spawn_prob": 0.02,           # Much less likely (was 0.1)
                "despawn_prob": 0.3           # More likely to despawn (was 0.1)
            },
            "cow": {
                "count_range": [0, 4],        # More cows (was 0-2.5)
                "min_distance": 3,            # Can be closer (was 5)
                "spawn_prob": 0.03,           # More likely (was 0.01)
                "despawn_prob": 0.05          # Less likely to despawn (was 0.1)
            }
        }
    },
    "normal": {
        "name": "Normal Mode",
        "description": "Standard crafter experience",
        "materials": {
            "coal": {"threshold": 0, "probability": 0.15},
            "iron": {"threshold": 0.4, "probability": 0.25},
            "diamond": {"threshold": 0.18, "probability": 0.006},
            "tree": {"threshold": 0, "probability": 0.2},
            "lava": {"threshold": 0.35, "probability": 1.0}
        },
        "initial_spawns": {
            "cow": {"min_distance": 3, "probability": 0.015},
            "zombie": {"min_distance": 10, "probability": 0.007},
            "skeleton": {"probability": 0.05}
        },
        "dynamic_spawns": {
            "zombie": {
                "count_range": [0, 3.5],
                "min_distance": 6,
                "spawn_prob": 0.3,
                "despawn_prob": 0.4
            },
            "skeleton": {
                "count_range": [0, 2],
                "min_distance": 7,
                "spawn_prob": 0.1,
                "despawn_prob": 0.1
            },
            "cow": {
                "count_range": [0, 2.5],
                "min_distance": 5,
                "spawn_prob": 0.01,
                "despawn_prob": 0.1
            }
        }
    },
    "hard": {
        "name": "Hard Mode",
        "description": "Scarce resources, many enemies",
        "materials": {
            "coal": {"threshold": 0.2, "probability": 0.08},       # Less coal (was 0, 0.15)
            "iron": {"threshold": 0.5, "probability": 0.15},       # Less iron (was 0.4, 0.25)
            "diamond": {"threshold": 0.25, "probability": 0.002},  # Much rarer (was 0.18, 0.006)
            "tree": {"threshold": 0.2, "probability": 0.1},        # Fewer trees (was 0, 0.2)
            "lava": {"threshold": 0.25, "probability": 1.0}        # More lava (was 0.35)
        },
        "initial_spawns": {
            "cow": {"min_distance": 5, "probability": 0.005},      # Fewer cows (was 3, 0.015)
            "zombie": {"min_distance": 6, "probability": 0.02},    # More zombies closer (was 10, 0.007)
            "skeleton": {"probability": 0.15}                      # Many more skeletons (was 0.05)
        },
        "dynamic_spawns": {
            "zombie": {
                "count_range": [0, 6],        # Many more zombies (was 0-3.5)
                "min_distance": 4,            # Much closer (was 6)
                "spawn_prob": 0.5,            # Much more likely (was 0.3)
                "despawn_prob": 0.2           # Less likely to leave (was 0.4)
            },
            "skeleton": {
                "count_range": [0, 4],        # Many more skeletons (was 0-2)
                "min_distance": 5,            # Closer (was 7)
                "spawn_prob": 0.3,            # Much more likely (was 0.1)
                "despawn_prob": 0.05          # Rarely despawn (was 0.1)
            },
            "cow": {
                "count_range": [0, 1],        # Fewer cows (was 0-2.5)
                "min_distance": 8,            # Farther away (was 5)
                "spawn_prob": 0.002,          # Very rare (was 0.01)
                "despawn_prob": 0.2           # More likely to leave (was 0.1)
            }
        }
    },
    "peaceful": {
        "name": "Peaceful Mode",
        "description": "No enemies, abundant resources",
        "materials": {
            "coal": {"threshold": -0.3, "probability": 0.3},
            "iron": {"threshold": 0.1, "probability": 0.4},
            "diamond": {"threshold": 0.1, "probability": 0.03},
            "tree": {"threshold": -0.3, "probability": 0.4},
            "lava": {"threshold": 0.6, "probability": 1.0}
        },
        "initial_spawns": {
            "cow": {"min_distance": 2, "probability": 0.05},
            "zombie": {"min_distance": 100, "probability": 0},     # No zombies
            "skeleton": {"probability": 0}                         # No skeletons
        },
        "dynamic_spawns": {
            "zombie": {
                "count_range": [0, 0],
                "min_distance": 100,
                "spawn_prob": 0,
                "despawn_prob": 1.0
            },
            "skeleton": {
                "count_range": [0, 0],
                "min_distance": 100,
                "spawn_prob": 0,
                "despawn_prob": 1.0
            },
            "cow": {
                "count_range": [0, 5],
                "min_distance": 2,
                "spawn_prob": 0.05,
                "despawn_prob": 0.02
            }
        }
    }
}

# Store active configuration
_active_world_config = None

def load_world_config(config_name: str = "normal", config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load world configuration from predefined configs or JSON file.
    
    Args:
        config_name: Name of predefined config ('easy', 'normal', 'hard', 'peaceful')
        config_path: Path to custom JSON config file (overrides config_name)
    
    Returns:
        World configuration dictionary
    """
    global _active_world_config
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            _active_world_config = json.load(f)
        #print(f"[PATCH] Loaded custom world config from {config_path}")
    elif config_name in WORLD_CONFIGS:
        _active_world_config = WORLD_CONFIGS[config_name]
        #print(f"[PATCH] Loaded '{config_name}' world configuration")
    else:
        _active_world_config = WORLD_CONFIGS["normal"]
        #print(f"[PATCH] Unknown config '{config_name}', using 'normal'")
    
    return _active_world_config

# Patch the world generation functions
original_set_material = worldgen._set_material
original_set_object = worldgen._set_object
original_balance_chunk = crafter.Env._balance_chunk

def patched_set_material(world, materials):
    """Patched version of _set_material that uses configuration."""
    config = _active_world_config or WORLD_CONFIGS["normal"]
    mat_config = config.get("materials", {})
    
    # Store original constants
    original_values = {}
    
    # Temporarily modify generation parameters
    if hasattr(worldgen, '_material_config'):
        original_values = worldgen._material_config.copy()
    
    # Apply configuration by modifying the function behavior
    worldgen._material_config = mat_config
    
    # Call original function
    result = original_set_material(world, materials)
    
    # Restore original values
    if original_values:
        worldgen._material_config = original_values
    
    return result

def patched_set_object(world, materials):
    """Patched version of _set_object that uses configuration."""
    config = _active_world_config or WORLD_CONFIGS["normal"]
    spawn_config = config.get("initial_spawns", {})
    
    # Apply spawn configuration
    worldgen._spawn_config = spawn_config
    
    # Call original function
    result = original_set_object(world, materials)
    
    return result

def patched_balance_chunk(self, chunk):
    """Patched version of _balance_chunk that uses configuration."""
    config = _active_world_config or WORLD_CONFIGS["normal"]
    dynamic_config = config.get("dynamic_spawns", {})
    
    # Store original values for restoration
    original_balance_config = getattr(self, '_balance_config', None)
    
    # Apply dynamic spawn configuration
    self._balance_config = dynamic_config
    
    # Call original function
    result = original_balance_chunk(self, chunk)
    
    # Restore original config if it existed
    if original_balance_config is not None:
        self._balance_config = original_balance_config
    
    return result

# Apply patches
worldgen._set_material = patched_set_material
worldgen._set_object = patched_set_object
crafter.Env._balance_chunk = patched_balance_chunk

# Also need to patch the actual generation logic
original_generate_world = worldgen.generate_world

def patched_generate_world(world, player):
    """Enhanced world generation with configuration support."""
    config = _active_world_config or WORLD_CONFIGS["normal"]
    
    # Use the original world generation but with modified parameters
    # First, temporarily modify the worldgen module's behavior
    import functools
    import opensimplex
    
    # Create simplex noise generator
    simplex_raw = opensimplex.OpenSimplex(seed=world.random.randint(0, 2**31 - 1))
    
    # Helper function for noise (from original crafter)
    def _simplex_helper(simplex_obj, x, y, z, sizes, normalize=True):
        if not isinstance(sizes, dict):
            sizes = {sizes: 1}
        value = 0
        for size, weight in sizes.items():
            if hasattr(simplex_obj, 'noise3d'):
                noise = simplex_obj.noise3d(x / size, y / size, z)
            else:
                noise = simplex_obj.noise3(x / size, y / size, z)
            value += weight * noise
        if normalize:
            value /= sum(sizes.values())
        return value
    
    simplex = functools.partial(_simplex_helper, simplex_raw)
    
    # Get config
    mat_config = config.get("materials", {})
    
    # Create materials and tunnels arrays
    area = world.area
    materials = np.zeros(area, np.uint8)
    tunnels = np.zeros(area, bool)
    
    # Get player position
    px, py = player.pos
    
    for x in range(area[0]):
        for y in range(area[1]):
            # Get noise values using the helper
            terrain = simplex(x, y, 0, {30: 1})
            moisture = simplex(x, y, 1000, {20: 1})
            
            # Determine material based on noise and config
            dist_from_spawn = np.sqrt((x - px)**2 + (y - py)**2)
            
            # Clear area around spawn
            if dist_from_spawn < 4:
                materials[x, y] = world._mat_ids['grass']
            # Water bodies
            elif moisture < -0.3:
                materials[x, y] = world._mat_ids['water']
            # Sand near water
            elif moisture < -0.2:
                materials[x, y] = world._mat_ids['sand']
            # Mountains and caves
            elif terrain > 0.15:
                # Deep mountain
                if terrain > mat_config.get("diamond", {}).get("threshold", 0.18):
                    if world.random.random() < mat_config.get("diamond", {}).get("probability", 0.006):
                        materials[x, y] = world._mat_ids['diamond']
                    elif simplex(x, y, 2000, {10: 1}) > mat_config.get("lava", {}).get("threshold", 0.35):
                        materials[x, y] = world._mat_ids['lava']
                    else:
                        materials[x, y] = world._mat_ids['stone']
                # Regular mountain
                else:
                    if terrain > mat_config.get("iron", {}).get("threshold", 0.4) and world.random.random() < mat_config.get("iron", {}).get("probability", 0.25):
                        materials[x, y] = world._mat_ids['iron']
                    elif terrain > mat_config.get("coal", {}).get("threshold", 0) and world.random.random() < mat_config.get("coal", {}).get("probability", 0.15):
                        materials[x, y] = world._mat_ids['coal']
                    else:
                        materials[x, y] = world._mat_ids['stone']
                # Caves/tunnels
                cave_noise = simplex(x, y, 3000, {5: 1})
                if cave_noise > 0.6:
                    materials[x, y] = world._mat_ids['path']
                    tunnels[x, y] = True
            else:
                materials[x, y] = world._mat_ids['grass']
    
    # Set materials in world
    worldgen._set_material(world, materials)
    
    # Place objects with config
    spawn_config = config.get("initial_spawns", {})
    
    for x in range(area[0]):
        for y in range(area[1]):
            if materials[x, y] == world._mat_ids['grass']:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                
                # Trees
                tree_config = mat_config.get("tree", {})
                if (simplex(x, y, 0, {10: 1}) > tree_config.get("threshold", 0) and 
                    world.random.random() < tree_config.get("probability", 0.2) and
                    dist > 3):
                    world.add(crafter.objects.Tree(world, (x, y)))
                
                # Cows
                cow_config = spawn_config.get("cow", {})
                if (dist > cow_config.get("min_distance", 3) and 
                    world.random.random() < cow_config.get("probability", 0.015)):
                    world.add(crafter.objects.Cow(world, (x, y)))
                
                # Zombies
                zombie_config = spawn_config.get("zombie", {})
                if (dist > zombie_config.get("min_distance", 10) and 
                    world.random.random() < zombie_config.get("probability", 0.007)):
                    world.add(crafter.objects.Zombie(world, (x, y), player))
            
            # Skeletons in caves
            elif materials[x, y] == world._mat_ids['path']:
                skeleton_config = spawn_config.get("skeleton", {})
                if world.random.random() < skeleton_config.get("probability", 0.05):
                    world.add(crafter.objects.Skeleton(world, (x, y), player))

# Replace the world generation function
worldgen.generate_world = patched_generate_world

# Enhanced _balance_chunk implementation
def enhanced_balance_chunk(self, chunk):
    """Enhanced chunk balancing with configuration support."""
    config = _active_world_config or WORLD_CONFIGS["normal"]
    dynamic_config = config.get("dynamic_spawns", {})
    
    # Get counts of each entity type in chunk
    counts = {'zombie': 0, 'skeleton': 0, 'cow': 0}
    entities = []
    
    for obj in self._world._chunks[chunk]:
        if isinstance(obj, crafter.objects.Zombie):
            counts['zombie'] += 1
            entities.append(('zombie', obj))
        elif isinstance(obj, crafter.objects.Skeleton):
            counts['skeleton'] += 1
            entities.append(('skeleton', obj))
        elif isinstance(obj, crafter.objects.Cow):
            counts['cow'] += 1
            entities.append(('cow', obj))
    
    # Balance each entity type
    for entity_type, entity_config in dynamic_config.items():
        current_count = counts[entity_type]
        
        # Calculate target count based on config and conditions
        min_count, max_count = entity_config['count_range']
        if entity_type in ['zombie', 'cow']:
            # Vary by daylight
            target = min_count + (max_count - min_count) * (1 - self._world.daylight)
        else:
            target = max_count
        
        # Spawn new entities if below target
        if current_count < target:
            spawn_prob = entity_config['spawn_prob']
            if self._world.random.random() < spawn_prob:
                # Find valid spawn location
                attempts = 10
                while attempts > 0:
                    x = self._world.random.randint(chunk[0] * 12, (chunk[0] + 1) * 12)
                    y = self._world.random.randint(chunk[1] * 12, (chunk[1] + 1) * 12)
                    
                    # Check distance from player
                    dist = np.sqrt((x - self._player.pos[0])**2 + (y - self._player.pos[1])**2)
                    if dist < entity_config['min_distance']:
                        attempts -= 1
                        continue
                    
                    # Check terrain type
                    mat = self._world._mat_map[x, y]
                    valid_spawn = False
                    
                    if entity_type == 'zombie' and mat == self._world._mat_ids['grass']:
                        valid_spawn = True
                    elif entity_type == 'skeleton' and mat == self._world._mat_ids['path']:
                        valid_spawn = True
                    elif entity_type == 'cow' and mat == self._world._mat_ids['grass']:
                        valid_spawn = True
                    
                    if valid_spawn and self._world._obj_map[x, y] == 0:
                        # Spawn entity
                        if entity_type == 'zombie':
                            self._world.add(crafter.objects.Zombie(self._world, (x, y), self._player))
                        elif entity_type == 'skeleton':
                            self._world.add(crafter.objects.Skeleton(self._world, (x, y), self._player))
                        elif entity_type == 'cow':
                            self._world.add(crafter.objects.Cow(self._world, (x, y)))
                        break
                    
                    attempts -= 1
        
        # Despawn entities if above target
        elif current_count > target:
            despawn_prob = entity_config['despawn_prob']
            for etype, entity in entities:
                if etype == entity_type and self._world.random.random() < despawn_prob:
                    dist = np.sqrt((entity.pos[0] - self._player.pos[0])**2 + 
                                 (entity.pos[1] - self._player.pos[1])**2)
                    if dist > entity_config['min_distance']:
                        self._world.remove(entity)
                        break

# Replace balance chunk with enhanced version
crafter.Env._balance_chunk = enhanced_balance_chunk

# Extend Env.__init__ to accept world_config parameter
original_env_init = crafter.Env.__init__

def patched_env_init(self, area=(64, 64), view=(9, 9), length=10000, seed=None, world_config="normal", world_config_path=None):
    """Extended Env.__init__ that accepts world configuration."""
    # Load world configuration
    load_world_config(world_config, world_config_path)
    
    # Call original init
    original_env_init(self, area=area, view=view, length=length, seed=seed)
    
    # Store config name for reference
    self._world_config_name = world_config if not world_config_path else "custom"

crafter.Env.__init__ = patched_env_init

print("[PATCH] Crafter world configuration patch complete.")
print("[PATCH] Available configs: easy, normal, hard, peaceful")
print("[PATCH] Usage: crafter.Env(world_config='easy') or crafter.Env(world_config_path='my_config.json')")

# Example custom config structure:
EXAMPLE_CUSTOM_CONFIG = {
    "name": "Custom World",
    "description": "My custom world configuration",
    "materials": {
        "coal": {"threshold": 0.1, "probability": 0.2},
        "iron": {"threshold": 0.3, "probability": 0.3},
        "diamond": {"threshold": 0.2, "probability": 0.01},
        "tree": {"threshold": -0.1, "probability": 0.25},
        "lava": {"threshold": 0.4, "probability": 1.0}
    },
    "initial_spawns": {
        "cow": {"min_distance": 4, "probability": 0.02},
        "zombie": {"min_distance": 8, "probability": 0.01},
        "skeleton": {"probability": 0.03}
    },
    "dynamic_spawns": {
        "zombie": {
            "count_range": [0, 2.5],
            "min_distance": 7,
            "spawn_prob": 0.25,
            "despawn_prob": 0.35
        },
        "skeleton": {
            "count_range": [0, 1.5],
            "min_distance": 8,
            "spawn_prob": 0.08,
            "despawn_prob": 0.15
        },
        "cow": {
            "count_range": [0, 3],
            "min_distance": 4,
            "spawn_prob": 0.02,
            "despawn_prob": 0.08
        }
    }
}

def save_example_config(path: str = "example_world_config.json"):
    """Save an example configuration file."""
    with open(path, 'w') as f:
        json.dump(EXAMPLE_CUSTOM_CONFIG, f, indent=2)
    print(f"[PATCH] Example config saved to {path}")