from .config_logging import configure_logging
from .environment import PokemonRedEnvironment

# Configure logging when red module is imported
configure_logging()

__all__ = ["PokemonRedEnvironment"]
