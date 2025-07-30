import importlib
import numpy as np
from typing import Any, Dict

# Minimal attributes to serialize per object type
BASIC_ATTRS: Dict[str, list] = {
    "Player": [
        "pos",
        "facing",
        "health",
        "inventory",
        "achievements",
        "action",
        "sleeping",
        "_last_health",
        "_hunger",
        "_thirst",
        "_fatigue",
        "_recover",
    ],
    "Cow": ["pos", "health"],
    "Zombie": ["pos", "health", "cooldown"],
    "Skeleton": ["pos", "health", "reload"],
    "Arrow": ["pos", "facing"],
    "Plant": ["pos", "health", "grown"],
    "Stone": ["pos"],
    "Table": ["pos"],
    "Furnace": ["pos"],
    # Add other types as needed
}


def serialize_world_object(obj: Any) -> Dict[str, Any]:
    """Convert a crafter object into a JSON-friendly dict."""
    cls_name = obj.__class__.__name__
    fields = BASIC_ATTRS.get(cls_name, ["pos"])
    payload: Dict[str, Any] = {}
    for field in fields:
        val = getattr(obj, field)
        if isinstance(val, np.ndarray):
            payload[field] = val.tolist()
        else:
            payload[field] = val
    return {
        "type": f"{obj.__class__.__module__}.{cls_name}",
        "state": payload,
    }


def deserialize_world_object(blob: Dict[str, Any], world: Any) -> Any:
    """Reconstruct a crafter object from its serialized dict."""
    type_str = blob.get("type", "")
    module_name, cls_name = type_str.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    # Bypass __init__; create empty instance
    state = blob.get("state", {})
    obj = cls.__new__(cls)
    # Initialize required base attributes
    obj.world = world
    obj.random = world.random
    obj.removed = False
    # Ensure inventory exists for health setter
    obj.inventory = {"health": 0}
    # Set attributes from state
    for field, value in state.items():
        if field == "pos":
            # restore position as numpy array
            obj.pos = np.array(value)
        else:
            # restore other attributes (including property setters)
            # convert lists back to arrays only for known ndarray fields
            setattr(obj, field, value)
    return obj
