"""Core utilities module."""

from synth_ai.core.utils.dict import deep_update
from synth_ai.core.utils.env import get_api_key, mask_value, resolve_env_var
from synth_ai.core.utils.json import (
    create_and_write_json,
    load_json_to_dict,
    strip_json_comments,
)
from synth_ai.core.utils.paths import (
    REPO_ROOT,
    SYNTH_BIN_DIR,
    SYNTH_HOME_DIR,
    SYNTH_LOCALAPI_CONFIG_PATH,
    SYNTH_USER_CONFIG_PATH,
    configure_import_paths,
    temporary_import_paths,
)
from synth_ai.core.utils.secure_files import (
    ensure_private_dir,
    write_private_json,
    write_private_text,
)
from synth_ai.core.utils.urls import (
    BACKEND_URL_API,
    BACKEND_URL_BASE,
    FRONTEND_URL_BASE,
    normalize_backend_base,
    normalize_base_url,
    normalize_inference_base,
)

__all__ = [
    # dict
    "deep_update",
    # env
    "get_api_key",
    "mask_value",
    "resolve_env_var",
    # json
    "create_and_write_json",
    "load_json_to_dict",
    "strip_json_comments",
    # paths
    "REPO_ROOT",
    "SYNTH_BIN_DIR",
    "SYNTH_HOME_DIR",
    "SYNTH_LOCALAPI_CONFIG_PATH",
    "SYNTH_USER_CONFIG_PATH",
    "configure_import_paths",
    "temporary_import_paths",
    # secure_files
    "ensure_private_dir",
    "write_private_json",
    "write_private_text",
    # urls
    "BACKEND_URL_API",
    "BACKEND_URL_BASE",
    "FRONTEND_URL_BASE",
    "normalize_base_url",
    "normalize_backend_base",
    "normalize_inference_base",
]
