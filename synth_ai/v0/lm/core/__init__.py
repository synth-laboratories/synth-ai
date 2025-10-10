"""
Core LM module with v3 tracing support.
"""

# Import v3 as the default LM implementation
from .main_v3 import LM

__all__ = ["LM"]
