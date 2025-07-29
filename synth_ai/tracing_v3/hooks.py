"""Hook system for extending tracing functionality."""
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import asyncio
import inspect

from .abstractions import SessionTrace, SessionTimeStep, BaseEvent, SessionEventMessage


@dataclass
class Hook:
    """A hook that can be registered with the tracer."""
    name: str
    callback: Callable
    event_types: Optional[List[str]] = None
    priority: int = 0
    enabled: bool = True


class HookManager:
    """Manages hooks for session tracing."""
    
    def __init__(self):
        self.hooks: Dict[str, List[Hook]] = {
            "session_start": [],
            "session_end": [],
            "timestep_start": [],
            "timestep_end": [],
            "event_recorded": [],
            "message_recorded": [],
            "before_save": [],
            "after_save": [],
        }
    
    def register(self, 
                 event: str, 
                 callback: Callable,
                 name: str = None,
                 priority: int = 0,
                 event_types: List[str] = None) -> Hook:
        """Register a new hook."""
        if event not in self.hooks:
            raise ValueError(f"Unknown hook event: {event}")
            
        hook = Hook(
            name=name or callback.__name__,
            callback=callback,
            event_types=event_types,
            priority=priority,
        )
        
        self.hooks[event].append(hook)
        self.hooks[event].sort(key=lambda h: h.priority, reverse=True)
        
        return hook
    
    def unregister(self, event: str, name: str):
        """Unregister a hook by name."""
        if event not in self.hooks:
            return
            
        self.hooks[event] = [h for h in self.hooks[event] if h.name != name]
    
    async def trigger(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger all hooks for an event."""
        if event not in self.hooks:
            return []
            
        results = []
        for hook in self.hooks[event]:
            if not hook.enabled:
                continue
                
            # Check event type filter
            if hook.event_types and "event_obj" in kwargs:
                event_obj = kwargs["event_obj"]
                if hasattr(event_obj, "event_type") and event_obj.event_type not in hook.event_types:
                    continue
                    
            try:
                if asyncio.iscoroutinefunction(hook.callback):
                    result = await hook.callback(*args, **kwargs)
                else:
                    result = hook.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                # Log error but don't stop processing
                print(f"Hook {hook.name} failed: {e}")
                
        return results


# Default hooks for common use cases
def create_default_hooks() -> HookManager:
    """Create hook manager with default hooks."""
    manager = HookManager()
    
    # Example: Log session starts
    async def log_session_start(session_id: str, metadata: Dict[str, Any]):
        print(f"Session started: {session_id}")
    
    # Example: Validate events before recording
    def validate_event(event_obj: BaseEvent) -> bool:
        if not event_obj.system_instance_id:
            raise ValueError("Event must have system_instance_id")
        return True
    
    # Example: Add metadata to all events
    def enrich_event(event_obj: BaseEvent):
        if "enriched" not in event_obj.metadata:
            event_obj.metadata["enriched"] = True
            event_obj.metadata["hook_version"] = "1.0"
    
    # Register default hooks
    manager.register("session_start", log_session_start, priority=10)
    manager.register("event_recorded", validate_event, name="validate_event", priority=20)
    manager.register("event_recorded", enrich_event, name="enrich_event", priority=15)
    
    return manager

# Global hooks instance (can be customized per application)
GLOBAL_HOOKS = create_default_hooks()