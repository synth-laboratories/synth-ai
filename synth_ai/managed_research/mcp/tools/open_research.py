"""MCP tool definitions for the Open Research v1 surface.

Thin wrappers over ``OpenResearchClient``. Tool descriptions are
public-safe (no leak of internal scoring/profanity rules); typed error
envelopes from the backend pass through untouched so callers can branch
on ``class``.

These tools wrap the locked v1 HTTP contract used by the browser
composer: mandatory sign-in is supplied by the MCP transport, live
message moderation is deferred, and submissions go through the same
Spark reviewer gate with no MCP bypass.
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.mcp.registry import ToolDefinition, tool_schema


def _metric_target_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Metric and target the experiment is expected to satisfy.",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Metric identifier, e.g. ``craftax.reward.mean``."
                ),
            },
            "operator": {
                "type": "string",
                "enum": [">=", "<=", "=="],
            },
            "value": {
                "type": "number",
                "description": "Threshold value the metric is compared against.",
            },
        },
        "required": ["name", "operator", "value"],
    }


def build_open_research_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="open_research_list_projects",
            description=(
                "List public Open Research themes (slug, name, tagline, "
                "headline scores, default queue, supported queues)."
            ),
            input_schema=tool_schema(
                {
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=[],
            ),
            handler=server._tool_open_research_list_projects,
        ),
        ToolDefinition(
            name="open_research_get_project",
            description=(
                "Fetch one Open Research project's challenge statement, "
                "rubric, resources, current best, and supported queues."
            ),
            input_schema=tool_schema(
                {
                    "slug": {
                        "type": "string",
                        "description": "Project slug, e.g. ``craftax``.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["slug"],
            ),
            handler=server._tool_open_research_get_project,
        ),
        ToolDefinition(
            name="open_research_list_queues",
            description=(
                "List Open Research submission queues with admission status. "
                "Filter by project slug when provided."
            ),
            input_schema=tool_schema(
                {
                    "project_slug": {
                        "type": "string",
                        "description": "Optional project slug filter.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=[],
            ),
            handler=server._tool_open_research_list_queues,
        ),
        ToolDefinition(
            name="open_research_submit_question",
            description=(
                "Submit a question to an Open Research queue for review and, "
                "if approved, launch. Goes through the same backend review "
                "gate as the public composer (no MCP bypass). When unsigned-"
                "in, pair this tool with the auto-managed anonymous "
                "fingerprint or supply ``submitter_fingerprint`` explicitly."
            ),
            input_schema=tool_schema(
                {
                    "project_slug": {
                        "type": "string",
                        "description": "Project slug, e.g. ``craftax``.",
                    },
                    "queue_id": {
                        "type": "string",
                        "description": (
                            "Target queue id, e.g. ``q_oed_1h_craftax``."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": "Submission prompt (1..2000 chars).",
                    },
                    "hypothesis": {
                        "type": "string",
                        "maxLength": 1000,
                        "description": "Optional hypothesis (0..1000 chars).",
                    },
                    "metric_target": _metric_target_schema(),
                    "deo_kind": {
                        "type": "string",
                        "enum": ["open_ended_discovery"],
                        "description": "Directed-effort-outcome kind for this queue.",
                    },
                    "rubric_acknowledged": {
                        "type": "boolean",
                        "description": (
                            "Must be true: confirms the caller read the "
                            "project rubric before submitting."
                        ),
                    },
                    "submitter_handle": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                        "description": "Public submitter handle shown on the experiment row.",
                    },
                    "submitter_fingerprint": {
                        "type": "string",
                        "description": (
                            "Optional anonymous-submitter fingerprint. If "
                            "omitted and no api_key is supplied, the tool "
                            "loads or mints a stable per-machine value."
                        ),
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=[
                    "project_slug",
                    "queue_id",
                    "prompt",
                    "metric_target",
                    "deo_kind",
                    "rubric_acknowledged",
                    "submitter_handle",
                ],
            ),
            handler=server._tool_open_research_submit_question,
        ),
        ToolDefinition(
            name="open_research_get_submission",
            description=(
                "Poll one submission for review verdict and the launched "
                "experiment id once approved."
            ),
            input_schema=tool_schema(
                {
                    "submission_id": {
                        "type": "string",
                        "description": "Submission id returned by submit_question.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "submitter_fingerprint": {
                        "type": "string",
                        "description": (
                            "Anonymous fingerprint used at submit time. "
                            "Required to read your own pending submission "
                            "without a signed-in api key."
                        ),
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["submission_id"],
            ),
            handler=server._tool_open_research_get_submission,
        ),
        ToolDefinition(
            name="open_research_list_experiments",
            description=(
                "Public live experiment table for an Open Research project. "
                "Filter by project slug, status, and cursor-paginate."
            ),
            input_schema=tool_schema(
                {
                    "project_slug": {
                        "type": "string",
                        "description": "Optional project slug filter.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "done", "failed", "all"],
                        "description": "Optional status filter (default ``all``).",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Page size (1..100, default 25).",
                    },
                    "cursor": {
                        "type": "string",
                        "description": "Pagination cursor from a prior call.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=[],
            ),
            handler=server._tool_open_research_list_experiments,
        ),
        ToolDefinition(
            name="open_research_get_experiment",
            description=(
                "Full viewer payload for one experiment — status, reward "
                "series, achievements, score table, rollouts, artifact "
                "links."
            ),
            input_schema=tool_schema(
                {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment id (≡ SMR run id).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["experiment_id"],
            ),
            handler=server._tool_open_research_get_experiment,
        ),
        ToolDefinition(
            name="open_research_get_receipt",
            description=(
                "Public receipt payload for one finished experiment "
                "(markdown + structured metadata). Returns a typed "
                "``not_found`` envelope until the run is done and the "
                "WorkProduct is viewable."
            ),
            input_schema=tool_schema(
                {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment id.",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["experiment_id"],
            ),
            handler=server._tool_open_research_get_receipt,
        ),
        ToolDefinition(
            name="open_research_download_bundle",
            description=(
                "Stream the experiment bundle (tar.gz) to a local path. "
                "Returns the destination, byte count, and a sha256 of the "
                "downloaded stream."
            ),
            input_schema=tool_schema(
                {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment id.",
                    },
                    "dest_path": {
                        "type": "string",
                        "description": (
                            "Absolute or home-relative path for the .tar.gz "
                            "file. Parent directories are created as needed."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "HTTP timeout in seconds (default 600).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Optional Synth API key override.",
                    },
                    "backend_base": {
                        "type": "string",
                        "description": "Optional backend base override.",
                    },
                },
                required=["experiment_id", "dest_path"],
            ),
            handler=server._tool_open_research_download_bundle,
        ),
    ]


__all__ = ["build_open_research_tools"]
