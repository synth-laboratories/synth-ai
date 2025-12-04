"""Client utilities for querying prompt learning job results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from synth_ai.core._utils.http import AsyncHttpClient

from .prompt_learning_types import PromptResults


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


class PromptLearningClient:
    """Client for interacting with prompt learning jobs and retrieving results."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        """Initialize the prompt learning client.
        
        Args:
            base_url: Base URL of the backend API (e.g., "http://localhost:8000" or "http://localhost:8000/api")
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        base_url = base_url.rstrip("/")
        # Validate base_url format - warn if it already ends with /api (will be handled by AsyncHttpClient)
        if base_url.endswith("/api"):
            # This is OK - AsyncHttpClient._abs() will handle double /api/api paths
            pass
        self._base_url = base_url
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
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
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
        params = {"since_seq": since_seq, "limit": limit}
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            js = await http.get(
                f"/api/prompt-learning/online/jobs/{job_id}/events",
                params=params
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
        """Extract full text from a serialized template dict.
        
        Args:
            template: Serialized template dict with 'sections' field
            
        Returns:
            Formatted full text string matching backend format
        """
        sections = template.get("sections", [])
        if not sections:
            # Try alternative structure: prompt_sections (from to_dict format)
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
    
    def _extract_full_text_from_object(self, obj: Dict[str, Any]) -> Optional[str]:
        """Extract full text from a candidate's object field.
        
        Args:
            obj: Candidate object dict (may have 'data' with 'sections' or 'text_replacements')
            
        Returns:
            Formatted full text string or None if extraction fails
        """
        # Try to get sections from object.data.sections (template format)
        data = obj.get("data", {})
        if isinstance(data, dict):
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
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            
            # Best prompt event
            if event_type == "prompt.learning.best.prompt":
                result.best_prompt = event_data.get("best_prompt")
                result.best_score = event_data.get("best_score")
            
            # Top-K prompt content events
            elif event_type == "prompt.learning.top.prompt.content":
                result.top_prompts.append({
                    "rank": event_data.get("rank"),
                    "train_accuracy": event_data.get("train_accuracy"),
                    "val_accuracy": event_data.get("val_accuracy"),
                    "template": event_data.get("template"),
                    "full_text": event_data.get("full_text"),
                })
            
            # Final results event (contains all candidates)
            elif event_type == "prompt.learning.final.results":
                result.optimized_candidates = event_data.get("optimized_candidates", [])
                result.attempted_candidates = event_data.get("attempted_candidates", [])
                # Also extract best_prompt from final.results if not already set
                if result.best_prompt is None:
                    result.best_prompt = event_data.get("best_prompt")
                if result.best_score is None:
                    result.best_score = event_data.get("best_score")
                
                # Extract validation results from validation field if present
                validation_data = event_data.get("validation")
                if isinstance(validation_data, list):
                    for val_item in validation_data:
                        if isinstance(val_item, dict):
                            rank = val_item.get("rank")
                            accuracy = val_item.get("accuracy")
                            if rank is not None and accuracy is not None:
                                validation_by_rank[rank] = accuracy
            
            # Validation results - build map by rank
            elif event_type == "prompt.learning.validation.scored":
                result.validation_results.append(event_data)
                # Try to extract rank and accuracy for mapping
                rank = event_data.get("rank")
                accuracy = event_data.get("accuracy")
                if rank is not None and accuracy is not None:
                    validation_by_rank[rank] = accuracy
            
            # Completion event (fallback for best_score)
            elif event_type == "prompt.learning.gepa.complete":
                if result.best_score is None:
                    result.best_score = event_data.get("best_score")
            
            # MIPRO completion event - extract best_score
            elif event_type == "mipro.job.completed":
                if result.best_score is None:
                    # Prefer unified best_score field, fallback to best_full_score or best_minibatch_score
                    result.best_score = (
                        event_data.get("best_score") 
                        or event_data.get("best_full_score") 
                        or event_data.get("best_minibatch_score")
                    )
        
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
                train_accuracy = score.get("accuracy")
                
                # Extract val accuracy from validation map
                val_accuracy = validation_by_rank.get(rank)
                
                # Try to extract template and full_text
                template = None
                full_text = None
                
                # First try: template field (may be serialized dict)
                cand_template = cand.get("template")
                if cand_template and isinstance(cand_template, dict):
                        template = cand_template
                        full_text = self._extract_full_text_from_template(cand_template)
                    # If it's not a dict, skip (might be a backend object that wasn't serialized)
                
                # Second try: object field
                if not full_text:
                    obj = cand.get("object", {})
                    if isinstance(obj, dict):
                        full_text = self._extract_full_text_from_object(obj)
                        # If we got full_text but no template, try to build template structure
                        if full_text and not template:
                            # Try to extract template from object.data
                            obj_data = obj.get("data", {})
                            if isinstance(obj_data, dict) and obj_data.get("sections"):
                                template = {"sections": obj_data["sections"]}
                
                # Build prompt entry
                prompt_entry: Dict[str, Any] = {
                    "rank": rank,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                }
                if template:
                    prompt_entry["template"] = template
                if full_text:
                    prompt_entry["full_text"] = full_text
                
                result.top_prompts.append(prompt_entry)
            
            # Sort by rank to ensure correct order
            result.top_prompts.sort(key=lambda p: p.get("rank", 999))
        
        # If we have validation results, prefer validation score for best_score
        # Rank 0 is the best prompt
        if validation_by_rank and 0 in validation_by_rank:
            # Use validation score for best_score when available
            result.best_score = validation_by_rank[0]
        
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
        
        # Extract train accuracies (only from candidates that have accuracy field)
        train_accuracies = [
            c["accuracy"] for c in attempted if "accuracy" in c
        ]
        
        # Extract val accuracies (only from validations that have accuracy field)
        # IMPORTANT: Exclude baseline from "best" calculation - baseline is for comparison only
        val_accuracies = [
            v["accuracy"] for v in validation 
            if "accuracy" in v and not v.get("is_baseline", False)
        ]
        
        # Score distribution (bins)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": 0 for i in range(len(bins) - 1)}
        for acc in train_accuracies:
            for i in range(len(bins) - 1):
                if bins[i] <= acc < bins[i+1] or (i == len(bins) - 2 and acc == bins[i+1]):
                    distribution[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] += 1
                    break
        
        return {
            "best_train_accuracy": max(train_accuracies) if train_accuracies else None,
            "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
            "num_candidates_tried": len(attempted),
            "num_frontier_candidates": len(optimized),
            "score_distribution": distribution,
            "mean_train_accuracy": sum(train_accuracies) / len(train_accuracies) if train_accuracies else None,
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
    import asyncio
    
    client = PromptLearningClient(base_url, api_key)
    return asyncio.run(client.get_prompts(job_id))


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
    import asyncio
    
    client = PromptLearningClient(base_url, api_key)
    return asyncio.run(client.get_prompt_text(job_id, rank))


def get_scoring_summary(job_id: str, base_url: str, api_key: str) -> Dict[str, Any]:
    """Synchronous wrapper to get scoring summary.
    
    Args:
        job_id: Job ID
        base_url: Backend API base URL
        api_key: API key for authentication
        
    Returns:
        Dictionary with scoring statistics
    """
    import asyncio
    
    client = PromptLearningClient(base_url, api_key)
    return asyncio.run(client.get_scoring_summary(job_id))

