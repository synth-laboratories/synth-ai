from .validators import validate_task_app_url
from .health import task_app_health
from .contracts import (
    TaskAppContract,
    TaskAppEndpoints,
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutSafetyConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutTrajectory,
    RolloutStep,
    RolloutMetrics,
    TaskInfo,
)
from .json import to_jsonable
from .auth import (
    normalize_environment_api_key,
    is_api_key_header_authorized,
    require_api_key_dependency,
)
from .vendors import (
    normalize_vendor_keys,
    get_openai_key_or_503,
    get_groq_key_or_503,
)
from .proxy import (
    INTERACT_TOOL_SCHEMA,
    prepare_for_openai,
    prepare_for_groq,
    inject_system_hint,
    extract_message_text,
    parse_tool_call_from_text,
    synthesize_tool_call_if_missing,
)
from .datasets import TaskDatasetSpec, TaskDatasetRegistry
from .rubrics import (
    Criterion,
    Rubric,
    load_rubric,
    blend_rubrics,
    score_events_against_rubric,
    score_outcome_against_rubric,
)
from .client import TaskAppClient
from .errors import error_payload, http_exception, json_error_response

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
]
