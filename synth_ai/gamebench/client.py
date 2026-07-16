"""Typed direct client for jobs on a declared GameBench scorer service."""

from __future__ import annotations

import math
import time
from typing import Any

from synth_ai.gamebench.models import (
    GameBenchCandidateScoreCancellation,
    GameBenchCandidateScoreCleanupReceipt,
    GameBenchCandidateScoreIdentity,
    GameBenchCandidateScoreRequest,
    GameBenchCandidateScoreResult,
    GameBenchCandidateScoreSubmission,
)
from synth_ai.sdk.base import SynthBaseClient


_SCORER_JOBS_PATH = "/v1/gamebench/scorers/craftax/jobs"


class GameBenchScorersClient(SynthBaseClient):
    """Create, inspect, cancel, and clean exact isolated score jobs."""

    def __init__(
        self,
        *,
        scorer_token: str,
        scorer_base: str,
        timeout_seconds: float,
    ) -> None:
        if not isinstance(scorer_token, str) or not scorer_token.strip():
            raise ValueError("scorer_token must be explicit for GameBench scoring")
        if not isinstance(scorer_base, str) or not scorer_base.strip():
            raise ValueError("scorer_base must be explicit for GameBench scoring")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be explicit, finite, and positive")
        super().__init__(
            api_key=scorer_token,
            backend_base=scorer_base,
            timeout_seconds=timeout_seconds,
        )
        self._expected_by_job: dict[str, GameBenchCandidateScoreRequest] = {}

    def submit_candidate(
        self,
        request: GameBenchCandidateScoreRequest,
    ) -> GameBenchCandidateScoreSubmission:
        """Submit exact candidate bytes and bind the returned job to that request."""
        if not isinstance(request, GameBenchCandidateScoreRequest):
            raise TypeError("request must be GameBenchCandidateScoreRequest")
        payload = self._request(
            "POST",
            _SCORER_JOBS_PATH,
            json_body=request.to_wire(),
            timeout_seconds=request.timeout_seconds,
        )
        submission = GameBenchCandidateScoreSubmission.model_validate(payload)
        self._assert_identity(request, submission)
        if submission.input_authority.request_body_sha256 != request.request_body_sha256():
            raise ValueError("score submission request-body authority mismatch")
        if submission.input_authority.candidate_size_bytes != len(request.candidate_bytes):
            raise ValueError("score submission candidate-size authority mismatch")
        existing = self._expected_by_job.get(submission.job_id)
        if existing is not None and existing != request:
            raise ValueError("score job id was rebound to a different request")
        self._expected_by_job[submission.job_id] = request
        return submission

    def get(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> GameBenchCandidateScoreSubmission | GameBenchCandidateScoreResult:
        """Read a job only when this client has its exact submitted identity."""
        expected = self._expected_request(job_id)
        payload = self._request(
            "GET",
            f"{_SCORER_JOBS_PATH}/{job_id}",
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            ),
        )
        if not isinstance(payload, dict) or "status" not in payload:
            raise ValueError("score job response must be a typed object with status")
        if payload["status"] == "accepted":
            parsed: GameBenchCandidateScoreSubmission | GameBenchCandidateScoreResult = (
                GameBenchCandidateScoreSubmission.model_validate(payload)
            )
        else:
            parsed = GameBenchCandidateScoreResult.model_validate(payload)
        self._assert_identity(expected, parsed)
        if parsed.job_id != job_id:
            raise ValueError("score job response job_id mismatch")
        return parsed

    def wait(
        self,
        job_id: str,
        timeout: float,
        poll_interval: float,
    ) -> GameBenchCandidateScoreResult:
        """Wait for terminal state using only explicit polling and deadline values."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be explicit, finite, and positive")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be explicit, finite, and positive")
        if poll_interval > timeout:
            raise ValueError("poll_interval must not exceed timeout")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"GameBench score job {job_id} did not reach terminal state "
                    f"within {timeout} seconds"
                )
            state = self.get(
                job_id,
                timeout_seconds=min(self.timeout_seconds, remaining),
            )
            if isinstance(state, GameBenchCandidateScoreResult):
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"GameBench score job {job_id} did not reach terminal state "
                    f"within {timeout} seconds"
                )
            time.sleep(min(poll_interval, remaining))

    def cancel(self, job_id: str) -> GameBenchCandidateScoreCancellation:
        """Request cancellation and require an identity-bound typed receipt."""
        expected = self._expected_request(job_id)
        payload = self._request(
            "POST",
            f"{_SCORER_JOBS_PATH}/{job_id}/cancel",
            json_body={"job_id": job_id},
            timeout_seconds=self.timeout_seconds,
        )
        cancellation = GameBenchCandidateScoreCancellation.model_validate(payload)
        self._assert_identity(expected, cancellation)
        if cancellation.job_id != job_id:
            raise ValueError("score cancellation job_id mismatch")
        return cancellation

    def cleanup(self, job_id: str) -> GameBenchCandidateScoreCleanupReceipt:
        """Remove job-owned resources and require exact zero-residue proof."""
        expected = self._expected_request(job_id)
        payload = self._request(
            "POST",
            f"{_SCORER_JOBS_PATH}/{job_id}/cleanup",
            json_body={"job_id": job_id},
            timeout_seconds=self.timeout_seconds,
        )
        receipt = GameBenchCandidateScoreCleanupReceipt.model_validate(payload)
        self._assert_identity(expected, receipt)
        if receipt.job_id != job_id:
            raise ValueError("score cleanup job_id mismatch")
        return receipt

    def _expected_request(self, job_id: str) -> GameBenchCandidateScoreRequest:
        if not isinstance(job_id, str) or not job_id or job_id != job_id.strip():
            raise ValueError("job_id must be an explicit canonical string")
        expected = self._expected_by_job.get(job_id)
        if expected is None:
            raise ValueError(
                "job_id has no request identity in this client; resubmit the same "
                "idempotent GameBenchCandidateScoreRequest before get/cancel/cleanup"
            )
        return expected

    @staticmethod
    def _assert_identity(
        expected: GameBenchCandidateScoreRequest,
        actual: GameBenchCandidateScoreIdentity,
    ) -> None:
        expected_fields = expected.authority_fields()
        actual_fields = actual.authority_fields()
        for key, value in expected_fields.items():
            if actual_fields.get(key) != value:
                raise ValueError(f"score response authority mismatch for {key}")


class GameBenchClient:
    """Direct scorer-service namespace exposed as ``SynthClient.gamebench``."""

    def __init__(
        self,
        *,
        scorer_token: str,
        scorer_base: str,
        timeout_seconds: float,
    ) -> None:
        self.scorers = GameBenchScorersClient(
            scorer_token=scorer_token,
            scorer_base=scorer_base,
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        """Close scorer transport resources."""
        self.scorers.close()


__all__ = ["GameBenchClient", "GameBenchScorersClient"]
