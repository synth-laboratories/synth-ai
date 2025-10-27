from __future__ import annotations

import importlib

from .base_url import PROD_BASE_URL_DEFAULT, get_backend_from_env, get_learning_v2_base_url
from .cli import PromptedChoiceOption, PromptedChoiceType, print_next_step
from .env import mask_str, resolve_env_var, write_env_var_to_dotenv, write_env_var_to_json
from .http import AsyncHttpClient, HTTPError, http_request
from .modal import (
    ensure_modal_installed,
    ensure_task_app_ready,
    find_asgi_apps,
    is_local_demo_url,
    is_modal_public_url,
    normalize_endpoint_url,
)
from .process import ensure_local_port_available, popen_capture, popen_stream, popen_stream_capture
from .sqld import SQLD_VERSION, find_sqld_binary, install_sqld
from .task_app_discovery import AppChoice, discover_eval_config_paths, select_app_choice
from .task_app_env import (
    ensure_env_credentials,
    ensure_port_free,
    preflight_env_key,
)
from .user_config import USER_CONFIG_PATH, load_user_config, load_user_env, save_user_config, update_user_config

_BASE_EXPORTS = (
    "AppChoice",
    "AsyncHttpClient",
    "HTTPError",
    "PROD_BASE_URL_DEFAULT",
    "PromptedChoiceOption",
    "PromptedChoiceType",
    "SQLD_VERSION",
    "USER_CONFIG_PATH",
    "discover_eval_config_paths",
    "ensure_env_credentials",
    "ensure_local_port_available",
    "ensure_modal_installed",
    "ensure_port_free",
    "ensure_task_app_ready",
    "find_asgi_apps",
    "find_sqld_binary",
    "get_backend_from_env",
    "get_learning_v2_base_url",
    "http_request",
    "install_sqld",
    "is_local_demo_url",
    "is_modal_public_url",
    "load_user_config",
    "load_user_env",
    "mask_str",
    "normalize_endpoint_url",
    "popen_capture",
    "popen_stream",
    "popen_stream_capture",
    "preflight_env_key",
    "print_next_step",
    "resolve_env_var",
    "save_user_config",
    "select_app_choice",
    "task_app_state",
    "update_user_config",
    "write_env_var_to_dotenv",
    "write_env_var_to_json",
)

task_app_state = importlib.import_module("synth_ai.utils.task_app_state")

_TASK_APP_STATE_EXPORTS = tuple(getattr(task_app_state, "__all__", ()))
for _name in _TASK_APP_STATE_EXPORTS:
    globals()[_name] = getattr(task_app_state, _name)

__all__ = (*_BASE_EXPORTS, *_TASK_APP_STATE_EXPORTS)
