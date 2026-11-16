"""Bounded acceptance reads and transfers. See SYN-5692."""

from __future__ import annotations

__all__ = ["COMMAND_SECONDS", "OUTPUT_BYTES_MAX", "OUTPUT_PAGES_MAX", "PAGE_ITEMS", "WAIT_SECONDS"]
COMMAND_SECONDS = 120
WAIT_SECONDS = 540
PAGE_ITEMS = 100
OUTPUT_PAGES_MAX = 50
OUTPUT_BYTES_MAX = 64 << 20
