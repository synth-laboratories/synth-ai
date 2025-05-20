"""
Synth AI - Software for aiding the best and multiplying the will.
"""

from importlib.metadata import version

from synth_ai.zyk import LM  # Assuming LM is in zyk.py in the same directory

__version__ = version("synth-ai")  # Gets version from installed package metadata
__all__ = ["LM"]  # Explicitly define public API
