"""Graph completions client for graph inference (policies, verifiers, RLM).

**Status:** Alpha

This module provides the client for running inference on trained graphs,
including policy graphs, verifier graphs, and Reasoning Language Models (RLM).

Provides both sync and async clients:
- GraphCompletionsSyncClient: Synchronous client using Rust core bindings
- GraphCompletionsAsyncClient: Asynchronous client using Rust core bindings
- GraphCompletionsClient: Alias for GraphCompletionsAsyncClient (backward compat)
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Mapping, TypedDict

from synth_ai.core.errors import HTTPError
from synth_ai.core.tracing_v3.serialization import normalize_for_json
from synth_ai.sdk.graphs.trace_upload import (
    AUTO_UPLOAD_THRESHOLD_BYTES,
    TraceUploaderAsync,
)
from synth_ai.sdk.graphs.verifier_schemas import (
    CalibrationExampleInput,
    EvidenceItem,
    GoldExampleInput,
)

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.graphs.completions.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for graph completions. Install rust bindings.")
    return synth_ai_py


GraphKind = Literal["zero_shot", "graphgen", "registered"]

# Default evidence output directory
DEFAULT_EVIDENCE_DIR = Path(".synth_ai_evidence")


def save_evidence_locally(
    output: dict[str, Any],
    *,
    evidence_dir: Path | str | None = None,
    prefix: str = "verifier",
) -> Path | None:
    """Save evidence from verifier output to a local JSON file.

    Args:
        output: Verifier output dict containing 'evidence' field
        evidence_dir: Directory to save evidence (default: .synth_ai_evidence)
        prefix: Filename prefix (default: "verifier")

    Returns:
        Path to saved evidence file, or None if no evidence found
    """
    evidence = output.get("evidence", [])
    if not evidence:
        return None

    # Use default directory if not specified
    save_dir = Path(evidence_dir) if evidence_dir else DEFAULT_EVIDENCE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_evidence_{timestamp}.json"
    filepath = save_dir / filename

    # Parse evidence items if they're dicts
    evidence_items = []
    for item in evidence:
        if isinstance(item, dict):
            evidence_items.append(EvidenceItem.model_validate(item).model_dump())
        elif isinstance(item, EvidenceItem):
            evidence_items.append(item.model_dump())
        else:
            evidence_items.append(item)

    # Save with metadata
    evidence_data = {
        "timestamp": datetime.now().isoformat(),
        "evidence_count": len(evidence_items),
        "evidence": evidence_items,
        # Include outcome if present
        "outcome_review": output.get("outcome_review"),
        "event_totals": output.get("event_totals"),
    }

    with open(filepath, "w") as f:
        json.dump(evidence_data, f, indent=2, default=str)

    return filepath


class GraphTarget(TypedDict, total=False):
    kind: GraphKind
    job_id: str
    graph_name: str
    graphgen_job_id: str
    verifier_shape: str


class GraphInfo(TypedDict, total=False):
    """Metadata for a registered graph."""

    graph_id: str
    name: str
    version: int
    kind: str  # "policy", "verifier"
    best_score: float | None
    job_id: str | None  # Source job that created this graph
    created_at: str


class ListGraphsResponse(TypedDict):
    """Response from list_graphs."""

    graphs: List[GraphInfo]
    total: int


@dataclass
class GraphCompletionResponse:
    """Response from graph completion endpoint."""

    output: dict[str, Any]
    """The graph output data."""

    usage: dict[str, Any] | None = None
    """Token usage statistics."""

    cache_status: str | None = None
    """Cache hit status: 'warm', 'cold', or None."""

    latency_ms: float | None = None
    """Request latency in milliseconds."""

    raw: dict[str, Any] | None = None
    """Raw response dict for accessing additional fields."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphCompletionResponse:
        """Create from API response dict."""
        return cls(
            output=data.get("output", {}),
            usage=data.get("usage"),
            cache_status=data.get("cache_status"),
            latency_ms=data.get("latency_ms"),
            raw=data,
        )


class GraphCompletionsSyncClient:
    """Synchronous client for graph completions using Rust core bindings.

    Example:
        ```python
        client = GraphCompletionsSyncClient(base_url, api_key)

        # Run inference on a GraphGen job
        response = client.run(job_id="graphgen_xxx", input_data={"query": "hello"})
        print(response.output)

        # Just get the output
        output = client.run_output(job_id="graphgen_xxx", input_data={"query": "hello"})
        ```
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout
        self._rust = _require_rust()

    def _resolve_job_id(self, *, job_id: str | None, graph: GraphTarget | None) -> str:
        return self._rust.resolve_graph_job_id(job_id, graph)

    def run(
        self,
        *,
        input_data: Mapping[str, Any],
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
        timeout: float | None = None,
    ) -> GraphCompletionResponse:
        """Run graph completion and return typed response.

        Args:
            input_data: Input data for the graph
            job_id: GraphGen job ID or graph name
            graph: Alternative graph target specification
            model: Optional model override
            prompt_snapshot_id: Specific snapshot to use
            timeout: Request timeout (overrides client default)

        Returns:
            GraphCompletionResponse with output, usage, cache_status, etc.
        """
        payload: dict[str, Any] = {
            "job_id": self._resolve_job_id(job_id=job_id, graph=graph),
            "input": normalize_for_json(dict(input_data)),
        }
        if model:
            payload["model"] = model
        if prompt_snapshot_id:
            payload["prompt_snapshot_id"] = prompt_snapshot_id

        client = self._rust.SynthClient(self._key, self._base)
        result = client.graph_complete(payload)
        if not isinstance(result, dict):
            raise ValueError("graph_completions_invalid_response_shape")
        return GraphCompletionResponse.from_dict(result)

    def run_output(
        self,
        *,
        input_data: Mapping[str, Any],
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run graph completion and return just the output dict.

        Convenience method that returns only the output field.
        """
        result = self.run(
            input_data=input_data,
            job_id=job_id,
            graph=graph,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
            timeout=timeout,
        )
        return result.output

    def complete(
        self,
        graph_id: str,
        input_data: Mapping[str, Any],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> GraphCompletionResponse:
        """Execute any graph with arbitrary input.

        Args:
            graph_id: Built-in graph name, GraphGen job_id, or snapshot UUID
            input_data: Graph-specific input data
            model: Optional model override
            timeout: Request timeout

        Returns:
            GraphCompletionResponse
        """
        return self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
            timeout=timeout,
        )


class GraphCompletionsAsyncClient:
    """Asynchronous client for graph completions.

    Example:
        ```python
        client = GraphCompletionsAsyncClient(base_url, api_key)

        # Run inference on a GraphGen job
        result = await client.run(job_id="graphgen_xxx", input_data={"query": "hello"})
        print(result["output"])

        # With auto-upload for large traces (recommended for verifier calls)
        client = GraphCompletionsAsyncClient(base_url, api_key, auto_upload_traces=True)
        result = await client.verify_with_rubric(session_trace=large_trace, rubric=rubric)
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        auto_upload_traces: bool = False,
        auto_upload_threshold: int = AUTO_UPLOAD_THRESHOLD_BYTES,
    ) -> None:
        """Initialize the graph completions client.

        Args:
            base_url: Graph service base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 60s)
            auto_upload_traces: If True, automatically upload large traces via
                presigned URLs to avoid timeout issues. Recommended for verifier calls.
            auto_upload_threshold: Size threshold in bytes for auto-upload (default: 100KB)
        """
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout
        self._auto_upload_traces = auto_upload_traces
        self._auto_upload_threshold = auto_upload_threshold
        self._trace_uploader: TraceUploaderAsync | None = None
        self._rust = _require_rust()

    def _get_trace_uploader(self) -> TraceUploaderAsync:
        """Get or create the trace uploader instance."""
        if self._trace_uploader is None:
            self._trace_uploader = TraceUploaderAsync(
                self._base,
                self._key,
                timeout=self._timeout,
                auto_upload_threshold=self._auto_upload_threshold,
            )
        return self._trace_uploader

    async def _maybe_upload_trace(
        self, trace: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any] | None, str | None]:
        """Upload trace if auto-upload is enabled and trace is large.

        Returns:
            Tuple of (trace_content, trace_ref) - one will be None
        """
        if not self._auto_upload_traces:
            return trace, None

        uploader = self._get_trace_uploader()
        if not uploader.should_upload(trace):
            return trace, None

        # Upload and return ref
        trace_ref = await uploader.upload_trace(trace)
        return None, trace_ref

    async def list_graphs(
        self,
        *,
        kind: str | None = None,
        limit: int = 50,
    ) -> ListGraphsResponse:
        """List graphs registered to your organization.

        Returns graphs that have been created via GraphGen optimization jobs
        or manually registered. Only returns graphs belonging to your organization
        (determined by API key).

        Args:
            kind: Optional filter by graph kind ("policy", "verifier")
            limit: Maximum number of graphs to return (default: 50)

        Returns:
            ListGraphsResponse with graphs list and total count

        Example:
            ```python
            client = GraphCompletionsClient(base_url, api_key)

            # List all graphs
            result = await client.list_graphs()
            for graph in result["graphs"]:
                print(f"{graph['name']} (v{graph['version']}): {graph['kind']}")

            # List only verifier graphs
            verifiers = await client.list_graphs(kind="verifier")
            ```
        """
        params: dict[str, Any] = {"limit": limit}
        if kind:
            params["kind"] = kind

        client = self._rust.SynthClient(self._key, self._base)
        js = await asyncio.to_thread(client.list_graphs, kind, limit)
        if not isinstance(js, dict):
            return {"graphs": [], "total": 0}
        return {
            "graphs": js.get("graphs", []),
            "total": js.get("total", 0),
        }

    def _resolve_job_id(self, *, job_id: str | None, graph: GraphTarget | None) -> str:
        return self._rust.resolve_graph_job_id(job_id, graph)

    async def run(
        self,
        *,
        input_data: Mapping[str, Any],
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self._resolve_job_id(job_id=job_id, graph=graph),
            "input": normalize_for_json(dict(input_data)),
        }
        if model:
            payload["model"] = model
        if prompt_snapshot_id:
            payload["prompt_snapshot_id"] = prompt_snapshot_id

        client = self._rust.SynthClient(self._key, self._base)
        result = await asyncio.to_thread(client.graph_complete, payload)
        if not isinstance(result, dict):
            raise ValueError("graph_completions_invalid_response_shape")
        return result

    async def run_stream(
        self,
        *,
        input_data: Mapping[str, Any],
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run graph completion with SSE streaming.

        Yields events as they arrive from the backend. Terminal events have
        event_type in {"run_succeeded", "run_failed", "run_cancelled"}.

        Args:
            input_data: Input data for the graph
            job_id: GraphGen job ID or graph name
            graph: Alternative graph target specification
            model: Optional model override
            prompt_snapshot_id: Specific snapshot to use

        Yields:
            dict: SSE event payloads with event data

        Example:
            ```python
            async for event in client.run_stream(job_id="...", input_data={...}):
                event_type = event.get("event", {}).get("event_type")
                print(f"Event: {event_type}")
                if event_type == "run_succeeded":
                    print(f"Output: {event.get('event', {}).get('output')}")
            ```
        """
        payload: dict[str, Any] = {
            "job_id": self._resolve_job_id(job_id=job_id, graph=graph),
            "input": normalize_for_json(dict(input_data)),
        }
        if model:
            payload["model"] = model
        if prompt_snapshot_id:
            payload["prompt_snapshot_id"] = prompt_snapshot_id

        url = f"{self._base}/api/graphs/completions?stream=true"
        headers = {
            "X-API-Key": self._key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        from synth_ai.core.rust_core.sse import stream_sse_events

        try:
            async for event in stream_sse_events(
                url,
                headers=headers,
                method="POST",
                json_payload=payload,
                timeout=None,
            ):
                yield event
                event_type = (
                    event.get("event", {}).get("event_type") if isinstance(event, dict) else None
                )
                if event_type in {
                    "run_succeeded",
                    "run_failed",
                    "run_cancelled",
                    "output_validation_failed",
                }:
                    return
        except HTTPError as err:
            status = int(getattr(err, "status", 0) or 0)
            detail = getattr(err, "detail", None) or ""
            text = str(detail)[:500]
            if status in (400, 422):
                raise ValueError(f"graph_completions_validation_error: {text}") from err
            if status in (401, 403):
                raise PermissionError(f"graph_completions_auth_error: {text}") from err
            if status == 404:
                raise FileNotFoundError(f"graph_completions_not_found: {text}") from err
            if status == 429:
                raise Exception("graph_completions_rate_limited") from err
            raise Exception(f"graph_completions_error: {status} {text}") from err

    async def run_output(
        self,
        *,
        input_data: Mapping[str, Any],
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
    ) -> Any:
        result = await self.run(
            input_data=input_data,
            job_id=job_id,
            graph=graph,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
        )
        return result.get("output") if isinstance(result, dict) else None

    def _select_graph_shape(self, session_trace: Mapping[str, Any]) -> str:
        """Auto-select graph shape based on trace size.

        Returns one of: "single", "rlm"
        """
        # Estimate token count
        trace_str = json.dumps(normalize_for_json(session_trace))
        estimated_tokens = len(trace_str) // 4

        if estimated_tokens < 50_000:
            return "single"
        return "rlm"

    async def complete(
        self,
        graph_id: str,
        input_data: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Execute any graph with arbitrary input.

        Args:
            graph_id: Built-in graph name, GraphGen job_id, or snapshot UUID
            input_data: Graph-specific input data
            model: Optional model override

        Returns:
            Graph output dictionary
        """
        return await self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
        )

    async def verify_with_rubric(
        self,
        *,
        session_trace: Mapping[str, Any] | str,
        rubric: Mapping[str, Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        verifier_shape: str | None = None,
        rlm_impl: Literal["v1", "v2"] | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
        save_evidence: bool | Path | str = False,
    ) -> dict[str, Any]:
        """Verify trace using rubric criteria.

        Args:
            session_trace: V3/V4 trace format, or a trace_ref string (e.g., "trace:trace_abc123")
                If auto_upload_traces=True and the trace is large, it will be automatically
                uploaded via presigned URL.
            rubric: Rubric with event/outcome criteria
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt
            verifier_shape: "single" or "rlm" (auto-detects if None)
            rlm_impl: Optional RLM implementation ("v1" or "v2") when verifier_shape="rlm" (defaults to v1)
            options: Optional execution options (event, outcome, etc.)
            model: Optional model override
            save_evidence: If True, save evidence to default dir; if Path/str, save to that dir

        Returns:
            Verification result with event_reviews, outcome_review, evidence, etc.

        Raises:
            ValueError: If verifier_shape is not supported or rlm_impl is invalid for the shape.
        """
        # Handle trace_ref string (already uploaded trace)
        is_trace_ref = isinstance(session_trace, str) and (
            session_trace.startswith("trace:") or session_trace.startswith("trace_")
        )

        trace_ref = None
        trace_content: Mapping[str, Any] | None = None

        if is_trace_ref:
            trace_ref = session_trace
        else:
            trace_content, trace_ref = await self._maybe_upload_trace(session_trace)
            if trace_ref is None:
                trace_content = normalize_for_json(trace_content)

        request = self._rust.build_verifier_request(
            rubric=normalize_for_json(rubric),
            trace_content=trace_content,
            trace_ref=trace_ref,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            options=dict(options or {}) if options is not None else None,
            model=model,
            verifier_shape=verifier_shape,
            rlm_impl=rlm_impl,
        )
        client = self._rust.SynthClient(self._key, self._base)
        result = await asyncio.to_thread(client.graph_complete, request)
        output = result.get("output", result) if isinstance(result, dict) else result

        if save_evidence:
            evidence_dir = save_evidence if isinstance(save_evidence, (Path, str)) else None
            evidence_path = save_evidence_locally(
                output, evidence_dir=evidence_dir, prefix="rubric"
            )
            if evidence_path:
                output["_evidence_saved_to"] = str(evidence_path)

        return output

    async def verify_fewshot(
        self,
        *,
        session_trace: Mapping[str, Any],
        calibration_examples: List[Mapping[str, Any]],
        expected_score: float | None = None,
        expected_rubric: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        verifier_shape: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
        save_evidence: bool | Path | str = False,
    ) -> dict[str, Any]:
        """Deprecated. Use verify_with_rubric instead.

        Raises:
            ValueError: Always raised because this method is deprecated.
        """
        warnings.warn(
            "verify_fewshot is deprecated and no longer supported. Use verify_with_rubric instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise ValueError(
            "verify_fewshot is deprecated and no longer supported. Use verify_with_rubric instead."
        )

        # Validate calibration_examples client-side before sending to server
        validated_examples = []
        for idx, example in enumerate(calibration_examples):
            try:
                validated_examples.append(CalibrationExampleInput.model_validate(example))
            except Exception as e:
                raise ValueError(
                    f"Invalid calibration_example at index {idx}: {e}. "
                    f"Each example must have session_trace (V3/V4 format), event_rewards (list[float] 0.0-1.0), "
                    f"and outcome_reward (float 0.0-1.0). event_rewards length must match trace events."
                ) from e

        if verifier_shape is None:
            verifier_shape = self._select_graph_shape(session_trace)

        graph_id = f"zero_shot_verifier_fewshot_{verifier_shape}"

        # Convert validated examples back to dict for serialization
        input_data: dict[str, Any] = {
            "trace": normalize_for_json(session_trace),
            "calibration_examples": [ex.model_dump() for ex in validated_examples],
            "options": dict(options or {}),
        }
        if expected_score is not None:
            input_data["expected_score"] = expected_score
        if expected_rubric:
            input_data["expected_rubric"] = expected_rubric
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        if user_prompt:
            input_data["user_prompt"] = user_prompt

        result = await self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
        )
        output = result.get("output", result)

        # Save evidence locally if requested
        if save_evidence:
            evidence_dir = save_evidence if isinstance(save_evidence, (Path, str)) else None
            evidence_path = save_evidence_locally(
                output, evidence_dir=evidence_dir, prefix="fewshot"
            )
            if evidence_path:
                output["_evidence_saved_to"] = str(evidence_path)

        return output

    async def verify_contrastive(
        self,
        *,
        session_trace: Mapping[str, Any],
        gold_examples: List[Mapping[str, Any]],
        candidate_score: float,
        candidate_reasoning: str,
        expected_rubric: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        verifier_shape: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
        save_evidence: bool | Path | str = False,
    ) -> dict[str, Any]:
        """Deprecated. Use verify_with_rubric instead.

        Raises:
            ValueError: Always raised because this method is deprecated.
        """
        warnings.warn(
            "verify_contrastive is deprecated and no longer supported. Use verify_with_rubric instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise ValueError(
            "verify_contrastive is deprecated and no longer supported. Use verify_with_rubric instead."
        )

        # Validate gold_examples client-side before sending to server
        validated_gold_examples = []
        for idx, example in enumerate(gold_examples):
            try:
                validated_gold_examples.append(GoldExampleInput.model_validate(example))
            except Exception as e:
                raise ValueError(
                    f"Invalid gold_example at index {idx}: {e}. "
                    f"Each example must have summary (str, non-empty), gold_score (float 0.0-1.0), "
                    f"and gold_reasoning (str, non-empty)."
                ) from e

        # Validate candidate_score
        if (
            not isinstance(candidate_score, (int, float))
            or candidate_score < 0.0
            or candidate_score > 1.0
        ):
            raise ValueError(
                f"candidate_score must be float 0.0-1.0, got {candidate_score} (type: {type(candidate_score).__name__})"
            )

        # Validate candidate_reasoning
        if not isinstance(candidate_reasoning, str) or not candidate_reasoning.strip():
            raise ValueError(
                f"candidate_reasoning must be a non-empty string, got {type(candidate_reasoning).__name__}"
            )

        if verifier_shape is None:
            verifier_shape = self._select_graph_shape(session_trace)

        graph_id = f"zero_shot_verifier_contrastive_{verifier_shape}"

        # Convert validated examples back to dict for serialization
        input_data: dict[str, Any] = {
            "trace": normalize_for_json(session_trace),
            "gold_examples": [ex.model_dump() for ex in validated_gold_examples],
            "candidate_score": float(candidate_score),
            "candidate_reasoning": candidate_reasoning.strip(),
            "options": dict(options or {}),
        }
        if expected_rubric:
            input_data["expected_rubric"] = expected_rubric
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        if user_prompt:
            input_data["user_prompt"] = user_prompt

        result = await self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
        )
        output = result.get("output", result)

        # Save evidence locally if requested
        if save_evidence:
            evidence_dir = save_evidence if isinstance(save_evidence, (Path, str)) else None
            evidence_path = save_evidence_locally(
                output, evidence_dir=evidence_dir, prefix="contrastive"
            )
            if evidence_path:
                output["_evidence_saved_to"] = str(evidence_path)

        return output

    async def verify_with_prompts(
        self,
        *,
        session_trace: Mapping[str, Any],
        system_prompt: str,
        user_prompt: str,
        verifier_shape: str | None = None,
        rlm_impl: Literal["v1", "v2"] | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Verify trace using custom prompts (no rubric/examples).

        Args:
            session_trace: V3/V4 trace format
            system_prompt: Custom system prompt (required)
            user_prompt: Custom user prompt (required)
            verifier_shape: "single" or "rlm" (auto-detects if None)
            rlm_impl: Optional RLM implementation ("v1" or "v2") when verifier_shape="rlm" (defaults to v1)
            options: Optional execution options
            model: Optional model override

        Returns:
            Verification result

        Raises:
            ValueError: If verifier_shape is not supported or rlm_impl is invalid for the shape.
        """
        if verifier_shape is None:
            verifier_shape = self._select_graph_shape(session_trace)

        if verifier_shape not in {"single", "rlm"}:
            raise ValueError(
                "Unsupported verifier_shape. Use 'single' or 'rlm' with verify_with_prompts."
            )

        # For custom prompts, use rubric graphs but with custom prompts
        # The graph will use the prompts instead of rubric
        if verifier_shape == "single":
            graph_id = "zero_shot_verifier_rubric_single"
            if rlm_impl is not None:
                raise ValueError("rlm_impl is only valid when verifier_shape='rlm'.")
        else:
            if rlm_impl == "v2":
                graph_id = "zero_shot_verifier_rubric_rlm_v2"
            else:
                graph_id = "zero_shot_verifier_rubric_rlm"

        input_data: dict[str, Any] = {
            "trace": normalize_for_json(session_trace),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "options": dict(options or {}),
            # Empty rubric - prompts will be used instead
            "rubric": {"event": [], "outcome": []},
        }

        result = await self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
        )
        return result.get("output", result)

    async def rlm_inference(
        self,
        *,
        query: str,
        context: str | Mapping[str, Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        rlm_impl: Literal["v1", "v2"] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Zero-shot RLM inference for large-context tasks.

        Args:
            query: The query/question to answer
            context: Large context (can be string or dict, 1M+ tokens)
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt
            model: Model to use (must be RLM-capable, default: gpt-4o-mini)
            provider: Provider name (default: openai)
            rlm_impl: RLM implementation version ("v1" or "v2"). v1 is single-agent,
                v2 adds multi-agent coordination and AgentFS. Defaults to v1.
            options: Optional execution options (max_iterations, max_cost_usd, etc.)

        Returns:
            RLM inference result with output, usage, metadata
        """
        # Select graph based on rlm_impl
        if rlm_impl == "v2":
            graph_id = "zero_shot_rlm_single_v2"
        elif rlm_impl == "v1":
            graph_id = "zero_shot_rlm_single_v1"
        else:
            # Default to the original graph (which uses v1)
            graph_id = "zero_shot_rlm_single"

        input_data: dict[str, Any] = {
            "query": query,
            "context": context if isinstance(context, str) else normalize_for_json(context),
            "options": {
                "model": model,
                "provider": provider,
                **(dict(options or {})),
            },
        }
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        if user_prompt:
            input_data["user_prompt"] = user_prompt

        result = await self.run(
            input_data=input_data,
            job_id=graph_id,
        )
        return result


class VerifierAsyncClient(GraphCompletionsAsyncClient):
    """Verifier graph client that builds standard verifier inputs."""

    async def evaluate(
        self,
        *,
        session_trace: Mapping[str, Any],
        rubric: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        artifact: list[dict[str, Any]] | None = None,
        policy_name: str = "policy",
        task_app_id: str = "task",
        task_app_base_url: str | None = None,
        job_id: str | None = None,
        graph: GraphTarget | None = None,
        model: str | None = None,
        prompt_snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        task_app_payload: dict[str, Any] = {"id": task_app_id}
        if task_app_base_url:
            task_app_payload["base_url"] = task_app_base_url

        trace_payload = normalize_for_json(session_trace)
        input_data: dict[str, Any] = {
            "policy_name": policy_name,
            "task_app": task_app_payload,
            "trace": trace_payload,
            "options": dict(options or {}),
        }
        if artifact is not None:
            input_data["artifact"] = normalize_for_json(artifact)
        if rubric is not None:
            input_data["rubric"] = normalize_for_json(rubric)

        return await self.run(
            input_data=input_data,
            job_id=job_id,
            graph=graph,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
        )


GraphCompletionsClient = GraphCompletionsAsyncClient
"""Alias for GraphCompletionsAsyncClient."""

VerifierClient = VerifierAsyncClient
"""Alias for VerifierAsyncClient."""
