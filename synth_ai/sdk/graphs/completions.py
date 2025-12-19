"""Graph completions client for graph inference (policies, verifiers, RLM)."""

from __future__ import annotations

import json
from typing import Any, Literal, List, Mapping, Optional, TypedDict, Union

from synth_ai.core.http import AsyncHttpClient, HTTPError
from synth_ai.core.tracing_v3.serialization import normalize_for_json
from synth_ai.sdk.judging.schemas import (
    CalibrationExampleInput,
    GoldExampleInput,
)

GraphKind = Literal["zero_shot", "graphgen", "registered"]


class GraphTarget(TypedDict, total=False):
    kind: GraphKind
    job_id: str
    graph_name: str
    graphgen_job_id: str
    verifier_type: str


class GraphInfo(TypedDict, total=False):
    """Metadata for a registered graph."""

    graph_id: str
    name: str
    version: int
    kind: str  # "policy", "verifier", "judge"
    best_score: float | None
    job_id: str | None  # Source job that created this graph
    created_at: str


class ListGraphsResponse(TypedDict):
    """Response from list_graphs."""

    graphs: List[GraphInfo]
    total: int


class GraphCompletionsClient:
    """Client for /api/graphs/completions with flexible graph targeting."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout

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
            kind: Optional filter by graph kind ("policy", "verifier", "judge")
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

        try:
            async with AsyncHttpClient(self._base, self._key, timeout=self._timeout) as http:
                js = await http.get_json("/graph-evolve/graphs", params=params)
                if not isinstance(js, dict):
                    return {"graphs": [], "total": 0}
                return {
                    "graphs": js.get("graphs", []),
                    "total": js.get("total", 0),
                }
        except HTTPError as err:
            status = int(getattr(err, "status", 0) or 0)
            if status in (401, 403):
                raise PermissionError(f"list_graphs_auth_error: {err.detail}") from err
            if status >= 500:
                raise Exception("list_graphs_transient_error") from err
            raise

    def _resolve_job_id(self, *, job_id: str | None, graph: GraphTarget | None) -> str:
        if job_id:
            return job_id
        if not graph:
            raise ValueError("graph_completions_missing_job_id")
        if graph.get("job_id"):
            return str(graph["job_id"])
        kind = graph.get("kind")
        if kind == "zero_shot":
            verifier_type = graph.get("verifier_type") or graph.get("graph_name")
            if not verifier_type:
                raise ValueError("graph_completions_missing_verifier_type")
            return str(verifier_type)
        if kind == "graphgen":
            graphgen_job_id = graph.get("graphgen_job_id")
            if not graphgen_job_id:
                raise ValueError("graph_completions_missing_graphgen_job_id")
            return str(graphgen_job_id)
        graph_name = graph.get("graph_name")
        if graph_name:
            return str(graph_name)
        raise ValueError("graph_completions_missing_graph_target")

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

        try:
            async with AsyncHttpClient(self._base, self._key, timeout=self._timeout) as http:
                js = await http.post_json("/api/graphs/completions", json=payload)
                if not isinstance(js, dict):
                    raise ValueError("graph_completions_invalid_response_shape")
                return js
        except HTTPError as err:
            status = int(getattr(err, "status", 0) or 0)
            if status in (400, 422):
                raise ValueError(f"graph_completions_validation_error: {err.detail}") from err
            if status in (401, 403):
                raise PermissionError(f"graph_completions_auth_error: {err.detail}") from err
            if status == 404:
                raise FileNotFoundError(f"graph_completions_not_found: {err.detail}") from err
            if status == 429:
                raise Exception("graph_completions_rate_limited") from err
            if status >= 500:
                raise Exception("graph_completions_transient_error") from err
            raise

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
        
        Returns one of: "single", "mapreduce", "rlm"
        """
        # Estimate token count
        trace_str = json.dumps(normalize_for_json(session_trace))
        estimated_tokens = len(trace_str) // 4
        
        if estimated_tokens < 50_000:
            return "single"
        elif estimated_tokens < 500_000:
            return "mapreduce"
        else:
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
        session_trace: Mapping[str, Any],
        rubric: Mapping[str, Any],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        verifier_type: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Verify trace using rubric criteria.
        
        Args:
            session_trace: V3 trace format
            rubric: Rubric with event/outcome criteria
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt
            verifier_type: "single", "mapreduce", or "rlm" (auto-detects if None)
            options: Optional execution options (event, outcome, etc.)
            model: Optional model override
            
        Returns:
            Verification result with event_reviews, outcome_review, etc.
        """
        # Auto-select graph shape based on trace size
        if verifier_type is None:
            verifier_type = self._select_graph_shape(session_trace)
        
        # Use composable naming: zero_shot_verifier_{gold_output_format}_{graph_shape}
        graph_id = f"zero_shot_verifier_rubric_{verifier_type}"
        
        input_data: dict[str, Any] = {
            "session_trace": normalize_for_json(session_trace),
            "rubric": normalize_for_json(rubric),
            "options": dict(options or {}),
        }
        if system_prompt:
            input_data["system_prompt"] = system_prompt
        if user_prompt:
            input_data["user_prompt"] = user_prompt
        
        result = await self.run(
            input_data=input_data,
            job_id=graph_id,
            model=model,
        )
        return result.get("output", result)

    async def verify_fewshot(
        self,
        *,
        session_trace: Mapping[str, Any],
        calibration_examples: List[Mapping[str, Any]],
        expected_score: float | None = None,
        expected_rubric: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        verifier_type: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Verify trace using few-shot calibration examples.
        
        Args:
            session_trace: V3 trace format (validated using SessionTraceInput)
            calibration_examples: List of calibration examples with:
                - session_trace: V3 trace format
                - event_rewards: List[float] (0.0-1.0), one per event
                - outcome_reward: float (0.0-1.0)
            expected_score: Optional expected score for the trace being evaluated
            expected_rubric: Optional rubric/ground truth for the trace being evaluated
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt
            verifier_type: "single", "mapreduce", or "rlm" (auto-detects if None)
            options: Optional execution options
            model: Optional model override
            
        Returns:
            Verification result with event_reviews, outcome_review, etc.
            
        Raises:
            ValueError: If calibration_examples are invalid (validated client-side)
        """
        # Validate calibration_examples client-side before sending to server
        validated_examples = []
        for idx, example in enumerate(calibration_examples):
            try:
                validated_examples.append(CalibrationExampleInput.model_validate(example))
            except Exception as e:
                raise ValueError(
                    f"Invalid calibration_example at index {idx}: {e}. "
                    f"Each example must have session_trace (V3 format), event_rewards (list[float] 0.0-1.0), "
                    f"and outcome_reward (float 0.0-1.0). event_rewards length must match trace events."
                ) from e
        
        if verifier_type is None:
            verifier_type = self._select_graph_shape(session_trace)
        
        graph_id = f"zero_shot_verifier_fewshot_{verifier_type}"
        
        # Convert validated examples back to dict for serialization
        input_data: dict[str, Any] = {
            "session_trace": normalize_for_json(session_trace),
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
        return result.get("output", result)

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
        verifier_type: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Verify verifier judgment by comparing to gold-standard examples.
        
        NOTE: Contrastive mode evaluates a VERIFIER's judgment, not a trace directly.
        It asks: "Is this verifier's judgment consistent with how gold examples were scored?"
        
        Args:
            session_trace: V3 trace format (the trace being evaluated)
            gold_examples: List of gold examples with:
                - summary: str (required, non-empty)
                - gold_score: float (0.0-1.0, required)
                - gold_reasoning: str (required, non-empty)
            candidate_score: Verifier's predicted score for this trace (0.0-1.0, what we're evaluating)
            candidate_reasoning: Verifier's reasoning for this score (what we're evaluating)
            expected_rubric: Optional rubric/ground truth for this trace
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt
            verifier_type: "single", "mapreduce", or "rlm" (auto-detects if None)
            options: Optional execution options
            model: Optional model override
            
        Returns:
            Verification result with event_reviews, outcome_review, etc.
            
        Raises:
            ValueError: If gold_examples or candidate_score/reasoning are invalid (validated client-side)
        """
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
        if not isinstance(candidate_score, (int, float)) or candidate_score < 0.0 or candidate_score > 1.0:
            raise ValueError(
                f"candidate_score must be float 0.0-1.0, got {candidate_score} (type: {type(candidate_score).__name__})"
            )
        
        # Validate candidate_reasoning
        if not isinstance(candidate_reasoning, str) or not candidate_reasoning.strip():
            raise ValueError(
                f"candidate_reasoning must be a non-empty string, got {type(candidate_reasoning).__name__}"
            )
        
        if verifier_type is None:
            verifier_type = self._select_graph_shape(session_trace)
        
        graph_id = f"zero_shot_verifier_contrastive_{verifier_type}"
        
        # Convert validated examples back to dict for serialization
        input_data: dict[str, Any] = {
            "session_trace": normalize_for_json(session_trace),
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
        return result.get("output", result)

    async def verify_with_prompts(
        self,
        *,
        session_trace: Mapping[str, Any],
        system_prompt: str,
        user_prompt: str,
        verifier_type: str | None = None,
        options: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Verify trace using custom prompts (no rubric/examples).
        
        Args:
            session_trace: V3 trace format
            system_prompt: Custom system prompt (required)
            user_prompt: Custom user prompt (required)
            verifier_type: "single", "mapreduce", or "rlm" (auto-detects if None)
            options: Optional execution options
            model: Optional model override
            
        Returns:
            Verification result
        """
        if verifier_type is None:
            verifier_type = self._select_graph_shape(session_trace)
        
        # For custom prompts, use rubric single graph but with custom prompts
        # The graph will use the prompts instead of rubric
        graph_id = f"zero_shot_verifier_rubric_{verifier_type}"
        
        input_data: dict[str, Any] = {
            "session_trace": normalize_for_json(session_trace),
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
        context: Union[str, Mapping[str, Any]],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
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
            options: Optional execution options (max_iterations, max_cost_usd, etc.)
            
        Returns:
            RLM inference result with output, usage, metadata
        """
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


class VerifierClient(GraphCompletionsClient):
    """Verifier graph client that builds standard verifier inputs."""

    async def evaluate(
        self,
        *,
        session_trace: Mapping[str, Any],
        rubric: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
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
            "session_trace": trace_payload,
            "trace": trace_payload,
            "options": dict(options or {}),
        }
        if rubric is not None:
            input_data["rubric"] = normalize_for_json(rubric)

        return await self.run(
            input_data=input_data,
            job_id=job_id,
            graph=graph,
            model=model,
            prompt_snapshot_id=prompt_snapshot_id,
        )
