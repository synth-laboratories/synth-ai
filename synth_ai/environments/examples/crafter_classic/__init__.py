from .config_logging import configure_logging
from .environment import CrafterClassicEnvironment
from .engine import CrafterEngine

# Configure logging when crafter_classic module is imported
configure_logging()

__all__ = ["CrafterClassicEnvironment", "CrafterEngine"]
