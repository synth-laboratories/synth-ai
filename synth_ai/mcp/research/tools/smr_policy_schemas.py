"""Shared MCP input-schema fragments for Managed Research policy surfaces."""

from __future__ import annotations

from synth_ai.core.research._legacy.models.smr_credential_providers import (
    SMR_CREDENTIAL_PROVIDER_VALUES,
)
from synth_ai.core.research._legacy.models.smr_funding_sources import SMR_FUNDING_SOURCE_VALUES
from synth_ai.core.research._legacy.models.smr_inference_providers import (
    SMR_INFERENCE_PROVIDER_VALUES,
)
from synth_ai.core.research._legacy.models.smr_tool_providers import SMR_TOOL_PROVIDER_VALUES


def run_policy_input_schema() -> dict[str, object]:
    return {
        "type": "object",
        "description": "Optional run-scoped policy overlay for this launch.",
        "properties": {
            "funding_source": {
                "type": "string",
                "enum": list(SMR_FUNDING_SOURCE_VALUES),
                "description": "Optional funding-source hint for this run.",
            },
            "access": {
                "type": "object",
                "description": "Optional provider allowlists for this run.",
                "properties": {
                    "credential_providers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SMR_CREDENTIAL_PROVIDER_VALUES),
                        },
                        "description": "Optional credential-provider allowlist.",
                    },
                    "inference_providers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SMR_INFERENCE_PROVIDER_VALUES),
                        },
                        "description": "Optional metered-inference provider allowlist.",
                    },
                    "tool_providers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(SMR_TOOL_PROVIDER_VALUES),
                        },
                        "description": "Optional tool-provider allowlist.",
                    },
                },
            },
            "limits": {
                "type": "object",
                "description": "Optional run-scoped spend limits.",
                "properties": {
                    "total_cost_cents": {
                        "type": "integer",
                        "description": "Optional total-cost cap for this run in cents.",
                    }
                },
            },
        },
    }


__all__ = ["run_policy_input_schema"]
