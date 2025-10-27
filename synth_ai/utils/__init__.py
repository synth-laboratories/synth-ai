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
    hydrate_user_environment,
    preflight_env_key,
)
from .user_config import (
    USER_CONFIG_PATH,
    load_user_config,
    load_user_env,
    save_user_config,
    update_user_config,
)

__all__ = [
    "AppChoice",
    "AsyncHttpClient",
    "HTTPError",
    "http_request",
    "PromptedChoiceOption",
    "PromptedChoiceType",
    "PROD_BASE_URL_DEFAULT",
    "discover_eval_config_paths",
    "ensure_env_credentials",
    "ensure_local_port_available",
    "ensure_modal_installed",
    "ensure_task_app_ready",
    "ensure_port_free",
    "find_asgi_apps",
    "is_local_demo_url",
    "is_modal_public_url",
    "get_backend_from_env",
    "get_learning_v2_base_url",
    "hydrate_user_environment",
    "mask_str",
    "normalize_endpoint_url",
    "popen_capture",
    "popen_stream",
    "popen_stream_capture",
    "preflight_env_key",
    "print_next_step",
    "resolve_env_var",
    "SQLD_VERSION",
    "find_sqld_binary",
    "install_sqld",
    "USER_CONFIG_PATH",
    "load_user_config",
    "load_user_env",
    "save_user_config",
    "update_user_config",
    "select_app_choice",
    "write_env_var_to_dotenv",
    "write_env_var_to_json",
]

task_app_state = importlib.import_module("synth_ai.utils.task_app_state")

__all__.append("task_app_state")

for _name in getattr(task_app_state, "__all__", []):
    globals()[_name] = getattr(task_app_state, _name)
    if _name not in __all__:
        __all__.append(_name)
