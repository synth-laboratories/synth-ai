from .config_logging import configure_logging
from .engine import CrafterEngine
from .environment import CrafterClassicEnvironment

# Configure logging when crafter_classic module is imported
configure_logging()

__all__ = ["CrafterClassicEnvironment", "CrafterEngine"]
