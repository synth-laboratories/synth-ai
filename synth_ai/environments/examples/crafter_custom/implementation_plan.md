# Crafter Custom Implementation Plan

## Overview
Create a customizable fork of the Crafter environment that allows JSON-based configuration of world generation parameters including resource abundance, enemy spawn rates, and difficulty settings.

## Directory Structure
```
crafter_custom/
├── implementation_plan.md (this file)
├── test_configs.py (test script)
├── engine.py (CrafterCustomEngine)
├── environment.py (CrafterCustomEnvironment)
└── crafter/
    ├── __init__.py
    ├── env.py (modified from crafter)
    ├── worldgen.py (modified from crafter) 
    ├── constants.py (extended with config support)
    ├── objects.py (copied from crafter)
    ├── engine.py (copied from crafter)
    ├── recorder.py (copied from crafter)
    ├── data.yaml (copied from crafter)
    ├── assets/ (copied from crafter)
    ├── config.py (NEW - configuration system)
    └── config/
        ├── easy.json
        ├── normal.json
        ├── hard.json
        └── peaceful.json
```

## Implementation Steps

### 1. Copy Core Crafter Files
- Copy the essential crafter source files to create our custom version
- Files to copy from `.venv/lib/python3.11/site-packages/crafter/`:
  - `env.py` - Main environment class
  - `worldgen.py` - World generation logic
  - `constants.py` - Game constants
  - `objects.py` - Game objects (Player, Zombie, etc.)
  - `engine.py` - Core engine components
  - `recorder.py` - Recording functionality
  - `gui.py` - GUI rendering (if needed)
  - `data.yaml` - Game data definitions

### 2. Add Configuration System

#### 2.1 Create `config.py`
```python
# config.py
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class WorldGenConfig:
    # Material generation thresholds and probabilities
    coal_threshold: float = 0.0
    coal_probability: float = 0.15
    iron_threshold: float = 0.4
    iron_probability: float = 0.25
    diamond_threshold: float = 0.18
    diamond_probability: float = 0.006
    tree_threshold: float = 0.0
    tree_probability: float = 0.2
    lava_threshold: float = 0.35
    
    # Initial spawn configuration
    cow_spawn_probability: float = 0.015
    cow_min_distance: int = 3
    zombie_spawn_probability: float = 0.007
    zombie_min_distance: int = 10
    skeleton_spawn_probability: float = 0.05
    
    # Dynamic spawn configuration
    zombie_spawn_rate: float = 0.3
    zombie_despawn_rate: float = 0.4
    zombie_min_spawn_distance: int = 6
    zombie_max_count: float = 3.5
    
    skeleton_spawn_rate: float = 0.1
    skeleton_despawn_rate: float = 0.1
    skeleton_min_spawn_distance: int = 7
    skeleton_max_count: float = 2.0
    
    cow_spawn_rate: float = 0.01
    cow_despawn_rate: float = 0.1
    cow_min_spawn_distance: int = 5
    cow_max_count: float = 2.5
    
    @classmethod
    def from_json(cls, path: str) -> 'WorldGenConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
```

#### 2.2 Modify `env.py`
```python
# Add to __init__:
def __init__(self, area=(64, 64), view=(9, 9), length=10000, seed=None, 
             world_config=None):
    # ... existing init code ...
    
    # Load world configuration
    if isinstance(world_config, str):
        self._world_config = WorldGenConfig.from_json(world_config)
    elif isinstance(world_config, dict):
        self._world_config = WorldGenConfig(**world_config)
    elif isinstance(world_config, WorldGenConfig):
        self._world_config = world_config
    else:
        self._world_config = WorldGenConfig()  # Default config
    
    # Pass config to world
    self._world._config = self._world_config
```

#### 2.3 Modify `worldgen.py`
Update the generation functions to use configuration values:

```python
def _set_material(world, pos, player, tunnels, simplex):
    # Get config from world
    config = getattr(world, '_config', WorldGenConfig())
    
    # Use config values instead of hardcoded constants
    if material > config.coal_threshold:
        if world.random.random() < config.coal_probability:
            material = world._mat_ids['coal']
    # ... etc for other materials
```

### 3. Create Preset Configurations

#### 3.1 `config/easy.json`
```json
{
  "coal_threshold": -0.2,
  "coal_probability": 0.25,
  "iron_threshold": 0.2,
  "iron_probability": 0.35,
  "diamond_threshold": 0.15,
  "diamond_probability": 0.02,
  "tree_threshold": -0.2,
  "tree_probability": 0.35,
  "lava_threshold": 0.5,
  
  "cow_spawn_probability": 0.03,
  "cow_min_distance": 2,
  "zombie_spawn_probability": 0.003,
  "zombie_min_distance": 15,
  "skeleton_spawn_probability": 0.01,
  
  "zombie_spawn_rate": 0.1,
  "zombie_despawn_rate": 0.6,
  "zombie_min_spawn_distance": 10,
  "zombie_max_count": 1.5,
  
  "skeleton_spawn_rate": 0.02,
  "skeleton_despawn_rate": 0.3,
  "skeleton_min_spawn_distance": 12,
  "skeleton_max_count": 0.5,
  
  "cow_spawn_rate": 0.03,
  "cow_despawn_rate": 0.05,
  "cow_min_spawn_distance": 3,
  "cow_max_count": 4.0
}
```

### 4. Integration with Synth-AI

#### 4.1 Create `CrafterCustomEngine`
- Extend the existing CrafterEngine to use crafter_custom instead of crafter
- Support world_config parameter from task metadata

#### 4.2 Update imports
- Change `import crafter` to `from . import env as crafter`
- Ensure all references use our custom version

### 5. Testing
- Create test script to verify different configurations work as expected
- Test that serialization still works with custom crafter
- Verify MCTS can work with different difficulty levels

## Benefits of This Approach
1. **Clean separation** - No monkey patching required
2. **Full control** - Can modify any aspect of generation
3. **Type safety** - Configuration is properly typed
4. **Maintainability** - Easy to update and extend
5. **Compatibility** - Preserves all existing functionality

## Next Steps
1. Copy crafter source files to crafter_custom/
2. Implement configuration system
3. Update generation code to use config values
4. Create preset configuration files
5. Test with different settings
6. Document usage