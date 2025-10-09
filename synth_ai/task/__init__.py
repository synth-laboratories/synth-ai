from .auth import (
    is_api_key_header_authorized,
    normalize_environment_api_key,
    require_api_key_dependency,
)
from .client import TaskAppClient
from .contracts import (
    RolloutEnvSpec,
    RolloutMetrics,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutSafetyConfig,
    RolloutStep,
    RolloutTrajectory,
    TaskAppContract,
    TaskAppEndpoints,
    TaskInfo,
)
from .datasets import TaskDatasetRegistry, TaskDatasetSpec
from .errors import error_payload, http_exception, json_error_response
from .health import task_app_health
from .json import to_jsonable
from .proxy import (
    INTERACT_TOOL_SCHEMA,
    extract_message_text,
    inject_system_hint,
    parse_tool_call_from_text,
    prepare_for_groq,
    prepare_for_openai,
    synthesize_tool_call_if_missing,
)
from .rubrics import (
    Criterion,
    Rubric,
    blend_rubrics,
    load_rubric,
    score_events_against_rubric,
    score_outcome_against_rubric,
)
from .server import (
    ProxyConfig,
    RubricBundle,
    TaskAppConfig,
    create_task_app,
    run_task_app,
)
from .validators import validate_task_app_url
from .vendors import (
    get_groq_key_or_503,
    get_openai_key_or_503,
    normalize_vendor_keys,
)

__all__ = [
    "validate_task_app_url",
    "task_app_health",
    "TaskAppContract",
    "TaskAppEndpoints",
    "RolloutEnvSpec",
    "RolloutPolicySpec",
    "RolloutRecordConfig",
    "RolloutSafetyConfig",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutTrajectory",
    "RolloutStep",
    "RolloutMetrics",
    "TaskInfo",
    "to_jsonable",
    "normalize_environment_api_key",
    "is_api_key_header_authorized",
    "require_api_key_dependency",
    "normalize_vendor_keys",
    "get_openai_key_or_503",
    "get_groq_key_or_503",
    "INTERACT_TOOL_SCHEMA",
    "prepare_for_openai",
    "prepare_for_groq",
    "inject_system_hint",
    "extract_message_text",
    "parse_tool_call_from_text",
    "synthesize_tool_call_if_missing",
    "TaskDatasetSpec",
    "TaskDatasetRegistry",
    "Criterion",
    "Rubric",
    "load_rubric",
    "blend_rubrics",
    "score_events_against_rubric",
    "score_outcome_against_rubric",
    "TaskAppClient",
    "error_payload",
    "http_exception",
    "json_error_response",
    "run_task_app",
    "create_task_app",
    "RubricBundle",
    "ProxyConfig",
    "TaskAppConfig",
]
