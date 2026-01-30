"""Client utilities for querying prompt learning job results."""

from typing import Any, Dict, List, Optional

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.sdk.optimization.internal.utils import run_sync

from .prompt_learning_types import PromptResults


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_outcome_reward(payload: Dict[str, Any]) -> Optional[float]:
    outcome_objectives = payload.get("outcome_objectives")
    if isinstance(outcome_objectives, dict):
        reward_val = _coerce_float(outcome_objectives.get("reward"))
        if reward_val is not None:
            return reward_val
    return _coerce_float(payload.get("outcome_reward"))


def _validate_job_id(job_id: str) -> None:
    """Validate that job_id has the expected prompt learning format.

    Args:
        job_id: Job ID to validate

    Raises:
        ValueError: If job_id doesn't start with 'pl_'
    """
    if not job_id.startswith("pl_"):
        raise ValueError(
            f"Invalid prompt learning job ID format: {job_id!r}. "
            f"Expected format: 'pl_<identifier>' (e.g., 'pl_9c58b711c2644083')"
        )


def _extract_reward_value(
    payload: Any, fallback_keys: Optional[List[str]] = None
) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    reward_val = _extract_outcome_reward(payload)
    if reward_val is not None:
        return float(reward_val)
    if fallback_keys:
        for key in fallback_keys:
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _convert_template_to_pattern(template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sections = template.get("sections", [])
    if not sections:
        sections = template.get("prompt_sections", [])
    if not isinstance(sections, list) or not sections:
        return None
    messages: List[Dict[str, Any]] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        content = sec.get("content")
        if not content:
            continue
        messages.append(
            {
                "role": sec.get("role", sec.get("name", "system")),
                "name": sec.get("name"),
                "content": content,
            }
        )
    if not messages:
        return None
    return {"messages": messages}


def _extract_event_type(event: Dict[str, Any]) -> str:
    event_type = event.get("type") or event.get("event_type")
    if not event_type and isinstance(event.get("data"), dict):
        event_type = event["data"].get("type") or event["data"].get("event_type")
    return str(event_type or "")


def _extract_event_data(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("data")
    if data is None:
        data = event.get("payload") or event.get("data_json")
    if not isinstance(data, dict):
        return {}

    # Unwrap nested envelope if data still looks like an event wrapper.
    while (
        isinstance(data, dict)
        and "data" in data
        and any(
            key in data for key in ("type", "event_type", "message", "seq", "timestamp_ms", "ts")
        )
    ):
        inner = data.get("data")
        if not isinstance(inner, dict):
            break
        data = inner

    return data


def _append_event_bucket(
    bucket: Dict[str, Any],
    key: str,
    event_type: str,
    event_data: Dict[str, Any],
) -> None:
    if not event_data:
        return
    bucket.setdefault(key, []).append({"_event_type": event_type, **event_data})


def _merge_candidate_payload(event_data: Dict[str, Any]) -> Dict[str, Any]:
    program_candidate = event_data.get("program_candidate")
    if isinstance(program_candidate, dict):
        return {**event_data, **program_candidate}
    return event_data


class PromptLearningClient:
    """Client for interacting with prompt learning jobs and retrieving results."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 30.0
    ) -> None:
        """Initialize the prompt learning client.

        Args:
            base_url: Base URL of the backend API (defaults to BACKEND_URL_BASE from urls.py)
            api_key: API key for authentication (defaults to SYNTH_API_KEY env var)
            timeout: Request timeout in seconds
        """
        import os

        from synth_ai.core.utils.urls import BACKEND_URL_BASE

        if not base_url:
            base_url = BACKEND_URL_BASE
        self._base_url = base_url

        # Resolve API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        self._api_key = api_key
        self._timeout = timeout

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job metadata and status.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            Job metadata including status, best_score, created_at, etc.

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/prompt-learning/online/jobs/{job_id}")

    async def get_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """Get events for a prompt learning job.

        Args:
            job_id: Job ID
            since_seq: Return events after this sequence number
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries with type, message, data, etc.

        Raises:
            ValueError: If job_id format is invalid or response structure is unexpected
        """
        _validate_job_id(job_id)
        import asyncio
        import os

        if os.getenv("SYNTH_USE_RUST_CORE", "").lower() in {"1", "true", "yes"}:
            try:
                import synth_ai_py

                resp = await asyncio.to_thread(
                    synth_ai_py.poll_events,
                    "prompt_learning",
                    job_id,
                    self._base_url,
                    self._api_key,
                    since_seq,
                    limit,
                    int(self._timeout * 1000),
                )
                raw_events = resp.get("events") if isinstance(resp, dict) else []
                events = []
                for ev in raw_events or []:
                    if not isinstance(ev, dict):
                        continue
                    events.append(
                        {
                            "seq": ev.get("seq", -1),
                            "type": ev.get("type") or ev.get("event_type") or "unknown",
                            "message": ev.get("message"),
                            "data": ev.get("data_json") or ev.get("data") or ev,
                            "ts": ev.get("ts"),
                        }
                    )
                return events
            except Exception:
                pass

        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get(
                f"/api/prompt-learning/online/jobs/{job_id}/events",
                params=params,
            )
        if isinstance(js, dict) and isinstance(js.get("events"), list):
            return js["events"]
        # Handle case where response is directly a list
        if isinstance(js, list):
            return js
        # Unexpected response structure - raise instead of silently returning empty list
        raise ValueError(
            f"Unexpected response structure from events endpoint. "
            f"Expected dict with 'events' list or list directly, got: {type(js).__name__}"
        )

    def _extract_full_text_from_template(self, template: Dict[str, Any]) -> str:
        """Extract full text from a serialized template dict (legacy)."""
        sections = template.get("sections", [])
        if not sections:
            sections = template.get("prompt_sections", [])
        full_text_parts = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_name = sec.get("name", "")
            sec_role = sec.get("role", "")
            sec_content = str(sec.get("content", ""))
            full_text_parts.append(f"[{sec_role} | {sec_name}]\n{sec_content}")
        return "\n\n".join(full_text_parts)

    def _extract_full_text_from_pattern(self, pattern: Dict[str, Any]) -> str:
        """Extract full text from a serialized pattern dict."""
        messages = pattern.get("messages", [])
        full_text_parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_name = msg.get("name", "")
            msg_role = msg.get("role", "")
            msg_content = str(msg.get("pattern") or msg.get("content") or "")
            full_text_parts.append(f"[{msg_role} | {msg_name}]\n{msg_content}")
        return "\n\n".join(full_text_parts)

    def _extract_full_text_from_object(self, obj: Dict[str, Any]) -> Optional[str]:
        """Extract full text from a candidate's object field.

        Args:
            obj: Candidate object dict (may have 'data' with 'sections' or 'text_replacements')

        Returns:
            Formatted full text string or None if extraction fails
        """
        # Try to get messages from object.data.messages (pattern format)
        data = obj.get("data", {})
        if isinstance(data, dict):
            messages = data.get("messages", [])
            if messages:
                return self._extract_full_text_from_pattern({"messages": messages})
            sections = data.get("sections", [])
            if sections:
                full_text_parts = []
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    sec_name = sec.get("name", "")
                    sec_role = sec.get("role", "")
                    sec_content = str(sec.get("content", ""))
                    full_text_parts.append(f"[{sec_role} | {sec_name}]\n{sec_content}")
                return "\n\n".join(full_text_parts)

            # Try text_replacements format (transformation format)
            text_replacements = data.get("text_replacements", [])
            if text_replacements and isinstance(text_replacements, list):
                full_text_parts = []
                for replacement in text_replacements:
                    if not isinstance(replacement, dict):
                        continue
                    new_text = replacement.get("new_text", "")
                    role = replacement.get("apply_to_role", "system")
                    if new_text:
                        full_text_parts.append(f"[{role}]\n{new_text}")
                if full_text_parts:
                    return "\n\n".join(full_text_parts)

        # Try direct messages on object
        messages = obj.get("messages", [])
        if messages:
            return self._extract_full_text_from_pattern({"messages": messages})

        # Try direct sections on object
        sections = obj.get("sections", [])
        if sections:
            return self._extract_full_text_from_template({"sections": sections})

        return None

    async def get_prompts(self, job_id: str) -> PromptResults:
        """Get the best prompts and scoring metadata from a completed job.

        Args:
            job_id: Job ID

        Returns:
            PromptResults dataclass containing:
                - best_prompt: The top-performing prompt with sections and metadata
                - best_score: The best accuracy score achieved
                - top_prompts: List of top-K prompts with train/val scores
                - optimized_candidates: All frontier/Pareto-optimal candidates
                - attempted_candidates: All candidates tried during optimization

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        events = await self.get_events(job_id, limit=10000)

        result = PromptResults()

        # Build validation score map by rank for later use
        validation_by_rank: Dict[int, float] = {}

        # Extract results from events
        for event in events:
            event_type = _extract_event_type(event)
            event_data = _extract_event_data(event)
            if not event_type and isinstance(event_data, dict):
                event_type = str(event_data.get("type") or event_data.get("event_type") or "")

            if event_type:
                result.event_counts[event_type] = result.event_counts.get(event_type, 0) + 1
                result.event_history.append(
                    {
                        "type": event_type,
                        "seq": event.get("seq"),
                        "timestamp_ms": event.get("timestamp_ms"),
                        "ts": event.get("ts"),
                        "message": event.get("message"),
                        "data": event_data,
                    }
                )

            # Best prompt event (canonical)
            if event_type == "learning.policy.gepa.candidate.new_best":
                result.best_prompt = event_data.get("best_prompt")
                best_score = _extract_reward_value(event_data, fallback_keys=["best_score"])
                if best_score is not None:
                    result.best_reward = best_score
                best_candidate = (
                    event_data.get("best_candidate")
                    or event_data.get("candidate")
                    or event_data.get("program_candidate")
                )
                if isinstance(best_candidate, dict):
                    result.best_candidate = best_candidate
                _append_event_bucket(result.gepa, "best_candidates", event_type, event_data)

            # Candidate evaluated events may contain top-K prompt content
            elif event_type == "learning.policy.gepa.candidate.evaluated":
                candidate_view = _merge_candidate_payload(event_data)
                # Check if this is a top prompt content event (has rank)
                if event_data.get("rank") is not None:
                    pattern_payload = event_data.get("pattern")
                    if not pattern_payload:
                        template_payload = event_data.get("template")
                        if isinstance(template_payload, dict):
                            pattern_payload = _convert_template_to_pattern(template_payload)
                    prompt_entry: Dict[str, Any] = {
                        "rank": event_data.get("rank"),
                        "train_accuracy": candidate_view.get("train_accuracy"),
                        "val_accuracy": candidate_view.get("val_accuracy"),
                        "pattern": pattern_payload,
                        "full_text": event_data.get("full_text"),
                    }
                    candidate_id = candidate_view.get("candidate_id") or candidate_view.get(
                        "version_id"
                    )
                    if candidate_id is not None:
                        prompt_entry["candidate_id"] = candidate_id
                    for key in (
                        "parent_id",
                        "generation",
                        "accepted",
                        "is_pareto",
                        "mutation_type",
                        "mutation_params",
                        "prompt_summary",
                        "prompt_text",
                        "objectives",
                        "instance_scores",
                        "instance_objectives",
                        "seed_scores",
                        "seed_info",
                        "rollout_sample",
                        "token_usage",
                        "cost_usd",
                        "evaluation_duration_ms",
                        "skip_reason",
                        "stages",
                        "seeds_evaluated",
                        "full_score",
                    ):
                        value = candidate_view.get(key)
                        if value is not None:
                            prompt_entry[key] = value
                    mutation_type = candidate_view.get("mutation_type") or candidate_view.get(
                        "operator"
                    )
                    if mutation_type is not None:
                        prompt_entry["mutation_type"] = mutation_type
                    if candidate_view.get("minibatch_scores") is not None:
                        prompt_entry["minibatch_scores"] = candidate_view.get("minibatch_scores")
                    elif candidate_view.get("minibatch_score") is not None:
                        prompt_entry["minibatch_scores"] = [candidate_view.get("minibatch_score")]
                    result.top_prompts.append(prompt_entry)
                _append_event_bucket(result.gepa, "candidates", event_type, candidate_view)

            # Job completed event (contains all candidates) - canonical
            elif event_type == "learning.policy.gepa.job.completed":
                result.optimized_candidates = event_data.get("optimized_candidates", [])
                result.attempted_candidates = event_data.get("attempted_candidates", [])
                result.version_tree = event_data.get("version_tree")
                # Also extract best_prompt from final.results if not already set
                if result.best_prompt is None:
                    result.best_prompt = event_data.get("best_prompt")
                if result.best_reward is None:
                    best_score = _extract_reward_value(event_data, fallback_keys=["best_score"])
                    if best_score is not None:
                        result.best_reward = best_score

                # Extract rollout and proposal metrics
                # These may come from event_data directly or from nested state dict
                # Only update if we find non-zero values (preserve from earlier events)
                new_rollouts = event_data.get("total_rollouts") or event_data.get("state", {}).get(
                    "total_rollouts"
                )
                if new_rollouts:
                    result.total_rollouts = new_rollouts
                # trials_tried is the number of proposal/mutation calls
                new_proposals = event_data.get("trials_tried") or event_data.get("state", {}).get(
                    "total_trials"
                )
                if new_proposals:
                    result.total_proposal_calls = new_proposals

                # Extract validation results from validation field if present
                validation_data = event_data.get("validation")
                if isinstance(validation_data, list):
                    for val_item in validation_data:
                        if isinstance(val_item, dict):
                            rank = val_item.get("rank")
                            accuracy = _extract_reward_value(val_item)
                            if rank is not None and accuracy is not None:
                                validation_by_rank[rank] = accuracy
                if isinstance(event_data.get("baseline"), dict):
                    result.gepa["baseline"] = event_data.get("baseline")
                _append_event_bucket(result.gepa, "job_completed", event_type, event_data)

            # Validation results - build map by rank (canonical)
            elif event_type == "learning.policy.gepa.validation.completed":
                result.validation_results.append(event_data)
                # Try to extract rank and accuracy for mapping
                rank = event_data.get("rank")
                accuracy = _extract_reward_value(event_data)
                if rank is not None and accuracy is not None:
                    validation_by_rank[rank] = accuracy
                _append_event_bucket(result.gepa, "validation", event_type, event_data)

            elif event_type.startswith("learning.policy.gepa."):
                if event_type.endswith(".baseline") or "baseline" in event_type:
                    result.gepa["baseline"] = event_data
                elif "frontier_updated" in event_type or "frontier.updated" in event_type:
                    _append_event_bucket(result.gepa, "frontier_updates", event_type, event_data)
                elif "generation.complete" in event_type or "generation.completed" in event_type:
                    _append_event_bucket(result.gepa, "generations", event_type, event_data)
                elif "progress" in event_type or "rollouts.progress" in event_type:
                    _append_event_bucket(result.gepa, "progress_updates", event_type, event_data)

            # MIPRO completion event - extract best_score (canonical)
            elif event_type == "learning.policy.mipro.job.completed":
                if result.best_reward is None:
                    # Prefer unified best_score field, fallback to best_full_score or best_minibatch_score
                    result.best_reward = _extract_reward_value(
                        event_data,
                        fallback_keys=["best_score", "best_full_score", "best_minibatch_score"],
                    )
                if event_data.get("attempted_candidates") is not None:
                    result.attempted_candidates = event_data.get("attempted_candidates", [])
                if event_data.get("optimized_candidates") is not None:
                    result.optimized_candidates = event_data.get("optimized_candidates", [])
                result.mipro["job"] = {"_event_type": event_type, **event_data}

            elif event_type.startswith("learning.policy.mipro."):
                # Capture MIPRO iteration/trial/incumbent/budget events for inspection
                if "iteration.started" in event_type or "iteration.completed" in event_type:
                    _append_event_bucket(result.mipro, "iterations", event_type, event_data)
                elif "trial.started" in event_type or "trial.completed" in event_type:
                    _append_event_bucket(result.mipro, "trials", event_type, event_data)
                elif "candidate.new_best" in event_type:
                    _append_event_bucket(result.mipro, "incumbents", event_type, event_data)
                    best_score = _extract_reward_value(
                        event_data,
                        fallback_keys=["best_score", "score", "full_score", "minibatch_score"],
                    )
                    if best_score is not None and (
                        result.best_reward is None or best_score > result.best_reward
                    ):
                        result.best_reward = best_score
                    best_candidate = (
                        event_data.get("best_candidate")
                        or event_data.get("candidate")
                        or event_data.get("program_candidate")
                    )
                    if isinstance(best_candidate, dict):
                        result.best_candidate = best_candidate
                elif "candidate.evaluated" in event_type:
                    _append_event_bucket(result.mipro, "candidates", event_type, event_data)
                elif "budget" in event_type:
                    _append_event_bucket(result.mipro, "budget_updates", event_type, event_data)

        # If top_prompts is empty but we have optimized_candidates, extract from them
        if not result.top_prompts and result.optimized_candidates:
            for idx, cand in enumerate(result.optimized_candidates):
                if not isinstance(cand, dict):
                    continue

                # Extract rank (use index+1 if rank not present)
                rank = cand.get("rank")
                if rank is None:
                    rank = idx + 1

                # Extract train accuracy from score
                score = cand.get("score", {})
                if not isinstance(score, dict):
                    score = {}
                train_accuracy = _extract_reward_value(score)
                if train_accuracy is None:
                    train_accuracy = _extract_reward_value(cand)

                # Extract val accuracy from validation map
                val_accuracy = validation_by_rank.get(rank)

                # Try to extract pattern/template and full_text
                pattern = None
                full_text = None

                # First try: pattern field (preferred)
                cand_pattern = cand.get("pattern")
                if cand_pattern and isinstance(cand_pattern, dict):
                    pattern = cand_pattern
                    full_text = self._extract_full_text_from_pattern(cand_pattern)

                # Second try: template field (legacy)
                cand_template = cand.get("template")
                if cand_template and isinstance(cand_template, dict):
                    pattern = _convert_template_to_pattern(cand_template)
                    full_text = self._extract_full_text_from_template(cand_template)
                # If it's not a dict, skip (might be a backend object that wasn't serialized)

                # Third try: object field
                if not full_text:
                    obj = cand.get("object", {})
                    if isinstance(obj, dict):
                        full_text = self._extract_full_text_from_object(obj)
                        # If we got full_text but no pattern/template, try to build structure
                        if full_text and not pattern:
                            # Try to extract pattern from object.data
                            obj_data = obj.get("data", {})
                            if isinstance(obj_data, dict):
                                if obj_data.get("messages"):
                                    pattern = {"messages": obj_data["messages"]}
                                elif obj_data.get("sections"):
                                    pattern = _convert_template_to_pattern(
                                        {"sections": obj_data["sections"]}
                                    )

                # Build prompt entry
                prompt_entry: Dict[str, Any] = {
                    "rank": rank,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                }
                for key in (
                    "candidate_id",
                    "version_id",
                    "parent_id",
                    "generation",
                    "accepted",
                    "is_pareto",
                    "mutation_type",
                    "mutation_params",
                    "prompt_summary",
                    "prompt_text",
                    "objectives",
                    "instance_scores",
                    "instance_objectives",
                    "seed_scores",
                    "seed_info",
                    "rollout_sample",
                    "token_usage",
                    "cost_usd",
                    "evaluation_duration_ms",
                    "skip_reason",
                    "stages",
                    "seeds_evaluated",
                    "full_score",
                ):
                    value = cand.get(key)
                    if value is not None:
                        prompt_entry[key] = value
                if pattern:
                    prompt_entry["pattern"] = pattern
                if full_text:
                    prompt_entry["full_text"] = full_text

                result.top_prompts.append(prompt_entry)

            # Sort by rank to ensure correct order
            result.top_prompts.sort(key=lambda p: p.get("rank", 999))

        # If we have validation results, prefer validation score for best_score
        # Rank 0 is the best prompt
        if validation_by_rank and 0 in validation_by_rank:
            # Use validation score for best_score when available
            result.best_reward = validation_by_rank[0]

        return result

    async def get_prompt_text(self, job_id: str, rank: int = 1) -> Optional[str]:
        """Get the full text of a specific prompt by rank.

        Args:
            job_id: Job ID
            rank: Prompt rank (1 = best, 2 = second best, etc.)

        Returns:
            Full prompt text or None if not found

        Raises:
            ValueError: If job_id format is invalid or rank < 1
        """
        _validate_job_id(job_id)
        if rank < 1:
            raise ValueError(f"Rank must be >= 1, got: {rank}")
        prompts_data = await self.get_prompts(job_id)
        top_prompts = prompts_data.top_prompts

        for prompt_info in top_prompts:
            if prompt_info.get("rank") == rank:
                return prompt_info.get("full_text")

        return None

    async def get_scoring_summary(self, job_id: str) -> Dict[str, Any]:
        """Get a summary of scoring metrics for all candidates.

        Args:
            job_id: Job ID

        Returns:
            Dictionary with scoring statistics:
                - best_train_accuracy: Best training accuracy
                - best_val_accuracy: Best validation accuracy (if available)
                - num_candidates_tried: Total candidates evaluated
                - num_frontier_candidates: Number in Pareto frontier
                - score_distribution: Histogram of accuracy scores

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        prompts_data = await self.get_prompts(job_id)

        attempted = prompts_data.attempted_candidates
        optimized = prompts_data.optimized_candidates
        validation = prompts_data.validation_results

        # Extract train accuracies (prefer objectives when available)
        train_accuracies: List[float] = []
        for candidate in attempted:
            if not isinstance(candidate, dict):
                continue
            reward_val = _extract_reward_value(candidate)
            if reward_val is None:
                score = candidate.get("score")
                reward_val = _extract_reward_value(score)
            if reward_val is not None:
                train_accuracies.append(reward_val)

        # Extract val accuracies (prefer objectives, exclude baseline)
        # IMPORTANT: Exclude baseline from "best" calculation - baseline is for comparison only
        val_accuracies: List[float] = []
        for val_item in validation:
            if not isinstance(val_item, dict) or val_item.get("is_baseline", False):
                continue
            reward_val = _extract_reward_value(val_item)
            if reward_val is not None:
                val_accuracies.append(reward_val)

        # Score distribution (bins)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = {f"{bins[i]:.1f}-{bins[i + 1]:.1f}": 0 for i in range(len(bins) - 1)}
        for acc in train_accuracies:
            for i in range(len(bins) - 1):
                if bins[i] <= acc < bins[i + 1] or (i == len(bins) - 2 and acc == bins[i + 1]):
                    distribution[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] += 1
                    break

        return {
            "best_train_accuracy": max(train_accuracies) if train_accuracies else None,
            "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
            "num_candidates_tried": len(attempted),
            "num_frontier_candidates": len(optimized),
            "score_distribution": distribution,
            "mean_train_accuracy": sum(train_accuracies) / len(train_accuracies)
            if train_accuracies
            else None,
        }


# Synchronous wrapper for convenience
def get_prompts(job_id: str, base_url: str, api_key: str) -> PromptResults:
    """Synchronous wrapper to get prompts from a job.

    Args:
        job_id: Job ID (e.g., "pl_9c58b711c2644083")
        base_url: Backend API base URL
        api_key: API key for authentication

    Returns:
        PromptResults dataclass with prompt results
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_prompts(job_id),
        label="get_prompts()",
    )


def get_prompt_text(job_id: str, base_url: str, api_key: str, rank: int = 1) -> Optional[str]:
    """Synchronous wrapper to get prompt text by rank.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        api_key: API key for authentication
        rank: Prompt rank (1 = best, 2 = second best, etc.)

    Returns:
        Full prompt text or None if not found
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_prompt_text(job_id, rank),
        label="get_prompt_text()",
    )


def get_scoring_summary(job_id: str, base_url: str, api_key: str) -> Dict[str, Any]:
    """Synchronous wrapper to get scoring summary.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        api_key: API key for authentication

    Returns:
        Dictionary with scoring statistics
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_scoring_summary(job_id),
        label="get_scoring_summary()",
    )
