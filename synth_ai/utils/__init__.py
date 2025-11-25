"""
Backward compatibility re-exports for synth_ai.utils.

All utilities have been reorganized:
- Core utilities → synth_ai.core.*
- CLI utilities → synth_ai.cli.lib.*

This module provides lazy imports to avoid circular import issues.
"""

from __future__ import annotations

from typing import Any

# Direct imports from core (no circular dependency risk)
from synth_ai.core import task_app_state
from synth_ai.core.env import PROD_BASE_URL_DEFAULT, get_backend_from_env
from synth_ai.core.http import AsyncHttpClient, HTTPError, http_request
from synth_ai.core.json import create_and_write_json, load_json_to_dict, strip_json_comments
from synth_ai.core.paths import (
    REPO_ROOT,
    cleanup_paths,
    configure_import_paths,
    find_config_path,
    get_bin_path,
    get_env_file_paths,
    get_home_config_file_paths,
    is_hidden_path,
)
from synth_ai.core.process import (
    ensure_local_port_available,
    popen_capture,
    popen_stream,
    popen_stream_capture,
)
from synth_ai.core.task_app_state import (
    DEFAULT_TASK_APP_SECRET_NAME,
    current_task_app_id,
    load_demo_dir,
    load_template_id,
    now_iso,
    persist_api_key,
    persist_demo_dir,
    persist_env_api_key,
    persist_task_url,
    persist_template_id,
    read_task_app_config,
    record_task_app,
    resolve_task_app_entry,
    task_app_config_path,
    task_app_id_from_path,
    update_task_app_entry,
    write_task_app_config,
)
from synth_ai.core.telemetry import (
    flush_logger,
    log_batch,
    log_error,
    log_event,
    log_info,
    log_warning,
)
from synth_ai.core.user_config import (
    USER_CONFIG_PATH,
    load_user_config,
    load_user_env,
    save_user_config,
    update_user_config,
)

# Lazy imports for CLI utilities to avoid circular imports
_CLI_IMPORTS = {
    # From cli.lib.agents
    "write_agents_md": ("synth_ai.cli.lib.agents", "write_agents_md"),
    # From core.apps.common
    "extract_routes_from_app": ("synth_ai.core.apps.common", "extract_routes_from_app"),
    "get_asgi_app": ("synth_ai.core.apps.common", "get_asgi_app"),
    "load_module": ("synth_ai.core.apps.common", "load_module"),
    # From cli.lib.bin
    "install_bin": ("synth_ai.cli.lib.bin", "install_bin"),
    "verify_bin": ("synth_ai.cli.lib.bin", "verify_bin"),
    # From cli.lib.env
    "mask_str": ("synth_ai.cli.lib.env", "mask_str"),
    "read_env_var_from_file": ("synth_ai.cli.lib.env", "read_env_var_from_file"),
    "resolve_env_var": ("synth_ai.cli.lib.env", "resolve_env_var"),
    "write_env_var_to_dotenv": ("synth_ai.cli.lib.env", "write_env_var_to_dotenv"),
    "write_env_var_to_json": ("synth_ai.cli.lib.env", "write_env_var_to_json"),
    # From cli.lib.modal
    "ensure_modal_installed": ("synth_ai.cli.lib.modal", "ensure_modal_installed"),
    "ensure_task_app_ready": ("synth_ai.cli.lib.modal", "ensure_task_app_ready"),
    "find_asgi_apps": ("synth_ai.cli.lib.modal", "find_asgi_apps"),
    "is_local_demo_url": ("synth_ai.cli.lib.modal", "is_local_demo_url"),
    "is_modal_public_url": ("synth_ai.cli.lib.modal", "is_modal_public_url"),
    "normalize_endpoint_url": ("synth_ai.cli.lib.modal", "normalize_endpoint_url"),
    # From cli.lib.prompts
    "PromptedChoiceOption": ("synth_ai.cli.lib.prompts", "PromptedChoiceOption"),
    "PromptedChoiceType": ("synth_ai.cli.lib.prompts", "PromptedChoiceType"),
    "PromptedPathOption": ("synth_ai.cli.lib.prompts", "PromptedPathOption"),
    "ctx_print": ("synth_ai.cli.lib.prompts", "ctx_print"),
    "prompt_choice": ("synth_ai.cli.lib.prompts", "prompt_choice"),
    "prompt_for_path": ("synth_ai.cli.lib.prompts", "prompt_for_path"),
    # From cli.lib.sqld
    "SQLD_VERSION": ("synth_ai.cli.lib.sqld", "SQLD_VERSION"),
    "find_sqld_binary": ("synth_ai.cli.lib.sqld", "find_sqld_binary"),
    "install_sqld": ("synth_ai.cli.lib.sqld", "install_sqld"),
    # From cli.lib.task_app_discovery
    "AppChoice": ("synth_ai.cli.lib.task_app_discovery", "AppChoice"),
    "discover_eval_config_paths": ("synth_ai.cli.lib.task_app_discovery", "discover_eval_config_paths"),
    "select_app_choice": ("synth_ai.cli.lib.task_app_discovery", "select_app_choice"),
    # From cli.lib.task_app_env
    "ensure_env_credentials": ("synth_ai.cli.lib.task_app_env", "ensure_env_credentials"),
    "ensure_port_free": ("synth_ai.cli.lib.task_app_env", "ensure_port_free"),
    "preflight_env_key": ("synth_ai.cli.lib.task_app_env", "preflight_env_key"),
    # From core.integrations.mcp.claude
    "ClaudeConfig": ("synth_ai.core.integrations.mcp.claude", "ClaudeConfig"),
}


def __getattr__(name: str) -> Any:
    if name in _CLI_IMPORTS:
        module_name, attr_name = _CLI_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'synth_ai.utils' has no attribute '{name}'")


__all__ = [
    # Core
    "AsyncHttpClient",
    "ClaudeConfig",
    "DEFAULT_TASK_APP_SECRET_NAME",
    "HTTPError",
    "PROD_BASE_URL_DEFAULT",
    "REPO_ROOT",
    "USER_CONFIG_PATH",
    "cleanup_paths",
    "configure_import_paths",
    "create_and_write_json",
    "current_task_app_id",
    "ensure_local_port_available",
    "find_config_path",
    "flush_logger",
    "get_backend_from_env",
    "get_bin_path",
    "get_env_file_paths",
    "get_home_config_file_paths",
    "http_request",
    "is_hidden_path",
    "load_demo_dir",
    "load_json_to_dict",
    "load_template_id",
    "load_user_config",
    "load_user_env",
    "log_batch",
    "log_error",
    "log_event",
    "log_info",
    "log_warning",
    "now_iso",
    "persist_api_key",
    "persist_demo_dir",
    "persist_env_api_key",
    "persist_task_url",
    "persist_template_id",
    "popen_capture",
    "popen_stream",
    "popen_stream_capture",
    "read_task_app_config",
    "record_task_app",
    "resolve_task_app_entry",
    "save_user_config",
    "strip_json_comments",
    "task_app_config_path",
    "task_app_id_from_path",
    "task_app_state",
    "update_task_app_entry",
    "update_user_config",
    "write_task_app_config",
    # CLI (lazy)
    "AppChoice",
    "PromptedChoiceOption",
    "PromptedChoiceType",
    "PromptedPathOption",
    "SQLD_VERSION",
    "ctx_print",
    "discover_eval_config_paths",
    "ensure_env_credentials",
    "ensure_modal_installed",
    "ensure_port_free",
    "ensure_task_app_ready",
    "extract_routes_from_app",
    "find_asgi_apps",
    "find_sqld_binary",
    "get_asgi_app",
    "install_bin",
    "install_sqld",
    "is_local_demo_url",
    "is_modal_public_url",
    "load_module",
    "mask_str",
    "normalize_endpoint_url",
    "preflight_env_key",
    "prompt_choice",
    "prompt_for_path",
    "read_env_var_from_file",
    "resolve_env_var",
    "select_app_choice",
    "verify_bin",
    "write_agents_md",
    "write_env_var_to_dotenv",
    "write_env_var_to_json",
]
