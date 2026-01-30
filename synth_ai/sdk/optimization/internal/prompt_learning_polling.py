from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from synth_ai.sdk.optimization.models import PolicyJobStatus as JobStatus
from synth_ai.sdk.optimization.models import PromptLearningResult

from .utils import ensure_api_base, http_get, parse_json_response


def poll_prompt_learning_until_complete(
    *,
    backend_url: str,
    api_key: str,
    job_id: str,
    timeout: float,
    interval: float,
    progress: bool,
    on_status: Optional[Callable[[Dict[str, Any]], None]],
    request_timeout: float,
) -> PromptLearningResult:
    base_url = ensure_api_base(backend_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start_time = time.time()
    elapsed = 0.0
    last_data: Dict[str, Any] = {}
    error_count = 0
    max_errors = 5
    logger = logging.getLogger(__name__)

    while elapsed <= timeout:
        try:
            url = f"{base_url}/prompt-learning/online/jobs/{job_id}"
            resp = http_get(url, headers=headers, timeout=request_timeout)
            last_data = parse_json_response(resp, context="Prompt learning status")
            error_count = 0

            status = JobStatus.from_string(last_data.get("status", "pending"))
            best_score = (
                last_data.get("best_score")
                or last_data.get("best_reward")
                or last_data.get("best_train_score")
                or last_data.get("best_train_reward")
            )

            if progress:
                mins, secs = divmod(int(elapsed), 60)
                score_str = f"score: {best_score:.2f}" if best_score is not None else "score: --"
                iteration = last_data.get("iteration") or last_data.get("current_iteration")
                iter_str = f" | iter: {iteration}" if iteration is not None else ""
                line = f"[{mins:02d}:{secs:02d}] {status.value} | {score_str}{iter_str}"
                logger.info(line)

            if on_status:
                on_status(last_data)

            if status.is_terminal:
                if status == JobStatus.FAILED:
                    error_msg = (
                        last_data.get("error")
                        or last_data.get("error_message")
                        or last_data.get("failure_reason")
                        or last_data.get("message")
                        or "unknown"
                    )
                    logger.error(
                        "Job %s reached terminal state: %s — error: %s",
                        job_id,
                        status.value,
                        error_msg,
                    )
                elif status == JobStatus.CANCELLED:
                    logger.warning("Job %s was cancelled", job_id)
                else:
                    logger.info("Job %s completed with status: %s", job_id, status.value)
                return PromptLearningResult.from_response(job_id, last_data)

        except Exception as exc:
            error_count += 1
            logger.warning(
                "Polling error %s/%s for job %s: %s",
                error_count,
                max_errors,
                job_id,
                exc,
            )
            if error_count >= max_errors:
                raise RuntimeError(
                    f"Polling failed after {error_count} consecutive errors."
                ) from exc

        time.sleep(interval)
        elapsed = time.time() - start_time

    if progress:
        logger.warning("Polling timeout after %.0fs for job %s", timeout, job_id)

    return PromptLearningResult.from_response(job_id, last_data)
