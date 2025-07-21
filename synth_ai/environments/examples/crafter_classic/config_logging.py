"""
Logging configuration for Craftax environment.
Suppresses obnoxious JAX debug messages and sets appropriate log levels.
"""

import logging
import os
import warnings


def configure_logging():
    """Configure logging to suppress noisy debug messages."""

    # Suppress JAX debug logging by setting appropriate log levels
    jax_loggers = [
        "jax._src.cache_key",
        "jax._src.compilation_cache",
        "jax._src.compiler",
        "jax._src.dispatch",
        "jax",
        "jaxlib",
    ]

    for logger_name in jax_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    # Set JAX platform to CPU to avoid GPU-related logging
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Suppress JAX warnings and compilation messages
    os.environ.setdefault("JAX_ENABLE_X64", "False")
    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")

    # Configure root logger to INFO level (but don't override if already configured)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Suppress other noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Filter out specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="jax")
    warnings.filterwarnings("ignore", category=FutureWarning, module="jax")


def safe_compare(left, right, operation="<"):
    """
    Safely compare two values, handling string vs int comparison errors.

    Args:
        left: Left operand
        right: Right operand
        operation: Comparison operation ('>', '<', '>=', '<=', '==', '!=')

    Returns:
        bool: Result of comparison, or False if types are incompatible
    """
    try:
        # If both are strings, try to convert to numbers
        if isinstance(left, str) and isinstance(right, str):
            try:
                left = float(left)
                right = float(right)
            except ValueError:
                # If conversion fails, compare as strings
                pass
        # If one is string and one is number, try to convert string to number
        elif isinstance(left, str) and isinstance(right, (int, float)):
            try:
                left = type(right)(left)
            except ValueError:
                logging.warning(f"Cannot compare string '{left}' with number {right}")
                return False
        elif isinstance(left, (int, float)) and isinstance(right, str):
            try:
                right = type(left)(right)
            except ValueError:
                logging.warning(f"Cannot compare number {left} with string '{right}'")
                return False

        # Perform the comparison
        if operation == "<":
            return left < right
        elif operation == ">":
            return left > right
        elif operation == "<=":
            return left <= right
        elif operation == ">=":
            return left >= right
        elif operation == "==":
            return left == right
        elif operation == "!=":
            return left != right
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    except TypeError as e:
        logging.error(f"Type error in comparison: {left} {operation} {right} - {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in comparison: {left} {operation} {right} - {e}")
        return False


# Configure logging when module is imported
configure_logging()
