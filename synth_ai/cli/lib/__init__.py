"""
Shared utility helpers for Synth AI CLI commands.

This package centralizes reusable logic (environment handling, tracing helpers,
sqld management, etc.) so individual command modules can stay focused on Click
CLI wiring.
"""

from __future__ import annotations

from .files import write_text
from .http import http_request
from .modal import (
    ensure_modal_installed,
    ensure_task_app_ready,
    find_asgi_apps,
    is_local_demo_url,
    is_modal_public_url,
    normalize_endpoint_url,
)
from .process import ensure_local_port_available, popen_capture, popen_stream, popen_stream_capture
from .secrets import key_preview
from .task_app_config import (
    create_new_config,
    find_vllm_tomls,
    fmt_float,
    prompt_value,
    select_or_create_config,
)
from .user_config import USER_CONFIG_PATH, load_user_config, save_user_config, update_user_config

__all__ = [
    "create_new_config",
    "ensure_local_port_available",
    "ensure_modal_installed",
    "ensure_task_app_ready",
    "find_asgi_apps",
    "find_vllm_tomls",
    "fmt_float",
    "http_request",
    "is_local_demo_url",
    "is_modal_public_url",
    "key_preview",
    "normalize_endpoint_url",
    "popen_capture",
    "popen_stream",
    "popen_stream_capture",
    "prompt_value",
    "select_or_create_config",
    "write_text",
    "load_user_config",
    "save_user_config",
    "update_user_config",
    "USER_CONFIG_PATH",
]
