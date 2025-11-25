"""Configuration base classes and utilities.

This module provides base classes for job configs and common
configuration patterns used across the SDK.
"""

from __future__ import annotations

from .base import BaseJobConfig, ConfigValidator

__all__ = [
    "BaseJobConfig",
    "ConfigValidator",
]


