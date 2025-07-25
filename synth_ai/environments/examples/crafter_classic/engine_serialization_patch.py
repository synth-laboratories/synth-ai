"""
Monkey patch to add save/load functionality to crafter.Env
Import this module before using CrafterEngine to enable serialization.
"""

import collections
import pickle
import numpy as np
import crafter
from crafter import objects
from synth_ai.environments.examples.crafter_classic.engine_helpers.serialization import (
    serialize_world_object, 
    deserialize_world_object
)

print("[PATCH] Attempting to apply Crafter serialization patch...")

# Check if already patched
if not hasattr(crafter.Env, 'save'):
    print("[PATCH] Adding save/load methods to crafter.Env...")
    
    def _save(self):
        """Save complete environment state."""
        # Save world objects with full state
        objects_data = []
        for i, obj in enumerate(self._world._objects):
            if obj is None:
                objects_data.append(None)
            else:
                # Use the existing serialization helper
                obj_data = serialize_world_object(obj)
                obj_data['_index'] = i  # Track position in array
                objects_data.append(obj_data)
        
        # Save world chunks (mapping of positions to object indices)
        chunks_data = {}
        for key, chunk_objs in self._world._chunks.items():
            # Store indices of objects in this chunk
            chunk_indices = []
            for obj in chunk_objs:
                if obj is not None:
                    try:
                        idx = self._world._objects.index(obj)
                        chunk_indices.append(idx)
                    except ValueError:
                        pass  # Object not in main list
            chunks_data[key] = chunk_indices
        
        # Get player index
        player_idx = None
        if self._player is not None:
            try:
                player_idx = self._world._objects.index(self._player)
            except ValueError:
                pass
        
        return {
            'step': self._step,
            'seed': self._seed,
            'length': self._length,
            'episode': self._episode,
            'player_idx': player_idx,
            'world': {
                'objects': objects_data,
                'chunks': chunks_data,
                'daylight': self._world.daylight,
                'random_state': pickle.dumps(self._world.random.get_state()),
                'area': self._world.area,
                '_mat_map': self._world._mat_map.tolist(),
                '_obj_map': self._world._obj_map.tolist(),
            }
        }
    
    def _load(self, state):
        """Load environment state from saved data."""
        # Restore basic attributes
        self._step = state['step']
        self._seed = state['seed']
        self._length = state['length']
        self._episode = state['episode']
        
        # Clear current world objects
        self._world._objects = []
        self._world._chunks.clear()
        
        # Restore world objects
        world_data = state['world']
        objects_by_idx = {}
        
        # First pass: create all objects
        for obj_data in world_data['objects']:
            if obj_data is None:
                self._world._objects.append(None)
            else:
                idx = obj_data.get('_index', len(self._world._objects))
                # Deserialize object
                obj = deserialize_world_object(obj_data, self._world)
                # Ensure list is long enough
                while len(self._world._objects) <= idx:
                    self._world._objects.append(None)
                self._world._objects[idx] = obj
                objects_by_idx[idx] = obj
        
        # Restore player reference
        if state['player_idx'] is not None and state['player_idx'] in objects_by_idx:
            self._player = objects_by_idx[state['player_idx']]
        else:
            # Find player in objects
            for obj in self._world._objects:
                if obj is not None and isinstance(obj, objects.Player):
                    self._player = obj
                    break
        
        # Restore chunks
        for key, obj_indices in world_data['chunks'].items():
            chunk_objects = set()
            for idx in obj_indices:
                if idx in objects_by_idx:
                    chunk_objects.add(objects_by_idx[idx])
            if chunk_objects:
                self._world._chunks[key] = chunk_objects
        
        # Restore world state
        self._world.daylight = world_data.get('daylight', 0.0)
        self._world.area = world_data['area']
        self._world._mat_map = np.array(world_data['_mat_map'])
        self._world._obj_map = np.array(world_data['_obj_map'])
        
        # Restore random state
        random_state = pickle.loads(world_data['random_state'])
        self._world.random.set_state(random_state)
    
    # Attach methods to Env class
    crafter.Env.save = _save
    crafter.Env.load = _load
    
    print("[PATCH] crafter.Env.save() and load() methods added.")
else:
    print("[PATCH] crafter.Env already has save/load methods.")

print("[PATCH] Crafter serialization patch complete.")