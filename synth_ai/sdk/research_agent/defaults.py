"""Defaults and shared constants for the research agent (library side)."""

from __future__ import annotations

DEFAULT_INSTRUCTIONS = "Run baseline, then optimize prompt with configured optimizer."
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_REASONING_EFFORT = "medium"

DEFAULT_BACKEND = "docker"
DEFAULT_BASE_IMAGE = "ubuntu:24.04"
DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_PACKAGES = (
    "git",
    "curl",
    "build-essential",
    "cmake",
    "ninja-build",
    "pkg-config",
    "python3",
    "python3-venv",
    "python3-pip",
    "ca-certificates",
    "jq",
)

DEFAULT_RESULT_PATTERNS = (
    "*.json",
    "*.log",
    "*.md",
    "*.toml",
    "diff.patch",
)

DEFAULT_SYNTH_PIP_SPEC = "synth-ai==0.2.26.dev1"
