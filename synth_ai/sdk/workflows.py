"""Hosted workflow SDK helpers.

Access via ``SynthClient().workflows``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from synth_ai.sdk.base import SynthBaseClient

__all__ = [
    "AsyncWorkflowsClient",
    "JesterkyReplayResult",
    "JesterkyRunResult",
    "JesterkyStreamEvent",
    "JesterkyValidationResult",
    "WorkflowsClient",
]


JesterkyActor = Literal["fake", "codex"]
_DEFAULT_WORKFLOW_HTTP_TIMEOUT_SECONDS = 3600.0


class JesterkyValidationResult(BaseModel):
    """Validation result returned by the hosted workflow API."""

    valid: bool
    exit_status: int | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    spec_path: str | None = None
    run_dir: str | None = None


class JesterkyRunResult(BaseModel):
    """Hosted jesterky run result with the canonical manifest payload."""

    run_id: str
    process_exit_status: int | None = None
    timed_out: bool = False
    manifest: dict[str, Any]
    manifest_path: str
    events_path: str
    spec_path: str | None = None
    run_dir: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


class JesterkyReplayResult(BaseModel):
    """Replay result returned by the hosted workflow API."""

    ok: bool
    exit_status: int | None = None
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    manifest_path: str | None = None
    spec_path: str | None = None
    run_dir: str | None = None


class JesterkyStreamEvent(BaseModel):
    """One SSE frame from the hosted jesterky run stream."""

    event: str
    raw_data: str
    data: Any = Field(default=None)


class WorkflowsClient(SynthBaseClient):
    """Submit and stream hosted workflow runs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            backend_base=backend_base,
            timeout_seconds=timeout_seconds,
        )
        self._prefix = "/api/v1/workflows"

    def validate_jesterky(
        self,
        spec: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> JesterkyValidationResult:
        """Validate a jesterky workflow spec through the backend runner."""
        body: dict[str, Any] = {"spec": dict(spec)}
        if timeout_seconds is not None:
            body["timeout_seconds"] = timeout_seconds
        payload = self._request(
            "POST",
            f"{self._prefix}/jesterky/validate",
            json_body=body,
            timeout_seconds=timeout_seconds,
        )
        return self.cast_to(JesterkyValidationResult, payload)

    def run_jesterky(
        self,
        spec: Mapping[str, Any],
        *,
        args: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        actor: JesterkyActor = "fake",
        model: str | None = None,
        codex_home: str | None = None,
        cd: str | None = None,
        follow: bool = False,
        width: int | None = None,
        timeout_seconds: float | None = None,
    ) -> JesterkyRunResult:
        """Run a jesterky workflow and return the manifest produced by jesterky."""
        body = _jesterky_run_body(
            spec,
            args=args,
            run_id=run_id,
            actor=actor,
            model=model,
            codex_home=codex_home,
            cd=cd,
            follow=follow,
            width=width,
            timeout_seconds=timeout_seconds,
        )
        payload = self._request(
            "POST",
            f"{self._prefix}/jesterky/runs",
            json_body=body,
            timeout_seconds=_workflow_http_timeout(self.timeout_seconds, timeout_seconds),
        )
        return self.cast_to(JesterkyRunResult, payload)

    def stream_jesterky(
        self,
        spec: Mapping[str, Any],
        *,
        args: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        actor: JesterkyActor = "fake",
        model: str | None = None,
        codex_home: str | None = None,
        cd: str | None = None,
        follow: bool = False,
        width: int | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[JesterkyStreamEvent]:
        """Stream a hosted jesterky run as SSE frames."""
        body = _jesterky_run_body(
            spec,
            args=args,
            run_id=run_id,
            actor=actor,
            model=model,
            codex_home=codex_home,
            cd=cd,
            follow=follow,
            width=width,
            timeout_seconds=timeout_seconds,
        )
        with self._client().stream(
            "POST",
            f"{self._prefix}/jesterky/runs/stream",
            headers=self._headers(),
            json=body,
            timeout=_workflow_http_timeout(self.timeout_seconds, timeout_seconds),
        ) as response:
            response.raise_for_status()
            yield from _iter_sse_events(response.iter_lines())

    def replay_jesterky(
        self,
        manifest: Mapping[str, Any],
        *,
        spec: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> JesterkyReplayResult:
        """Replay a jesterky manifest through the backend runner."""
        body: dict[str, Any] = {"manifest": dict(manifest)}
        if spec is not None:
            body["spec"] = dict(spec)
        if timeout_seconds is not None:
            body["timeout_seconds"] = timeout_seconds
        payload = self._request(
            "POST",
            f"{self._prefix}/jesterky/replay",
            json_body=body,
            timeout_seconds=_workflow_http_timeout(self.timeout_seconds, timeout_seconds),
        )
        return self.cast_to(JesterkyReplayResult, payload)

    def validate(
        self,
        spec: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> JesterkyValidationResult:
        """Alias for :meth:`validate_jesterky`."""
        return self.validate_jesterky(spec, timeout_seconds=timeout_seconds)

    def run(self, spec: Mapping[str, Any], **kwargs: Any) -> JesterkyRunResult:
        """Alias for :meth:`run_jesterky`."""
        return self.run_jesterky(spec, **kwargs)

    def stream(
        self, spec: Mapping[str, Any], **kwargs: Any
    ) -> Iterator[JesterkyStreamEvent]:
        """Alias for :meth:`stream_jesterky`."""
        return self.stream_jesterky(spec, **kwargs)

    def replay(
        self,
        manifest: Mapping[str, Any],
        *,
        spec: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> JesterkyReplayResult:
        """Alias for :meth:`replay_jesterky`."""
        return self.replay_jesterky(
            manifest,
            spec=spec,
            timeout_seconds=timeout_seconds,
        )


class AsyncWorkflowsClient:
    """Async adapter over :class:`WorkflowsClient`."""

    def __init__(self, sync_client: WorkflowsClient) -> None:
        self._sync_client = sync_client

    async def stream_jesterky(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[JesterkyStreamEvent]:
        """Thread-offloaded async stream adapter."""
        events = await asyncio.to_thread(
            lambda: list(self._sync_client.stream_jesterky(*args, **kwargs))
        )
        for event in events:
            yield event

    async def stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[JesterkyStreamEvent]:
        """Alias for :meth:`stream_jesterky`."""
        async for event in self.stream_jesterky(*args, **kwargs):
            yield event

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._sync_client, name)
        if callable(attr):

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            return _wrapped
        return attr


def _jesterky_run_body(
    spec: Mapping[str, Any],
    *,
    args: Mapping[str, Any] | None,
    run_id: str | None,
    actor: JesterkyActor,
    model: str | None,
    codex_home: str | None,
    cd: str | None,
    follow: bool,
    width: int | None,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "spec": dict(spec),
        "args": dict(args or {}),
        "actor": actor,
        "follow": follow,
    }
    optional = {
        "run_id": run_id,
        "model": model,
        "codex_home": codex_home,
        "cd": cd,
        "width": width,
        "timeout_seconds": timeout_seconds,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def _workflow_http_timeout(client_timeout: float, run_timeout: float | None) -> float:
    if run_timeout is not None:
        return run_timeout + 30.0
    return max(client_timeout, _DEFAULT_WORKFLOW_HTTP_TIMEOUT_SECONDS)


def _iter_sse_events(lines: Iterator[str | bytes]) -> Iterator[JesterkyStreamEvent]:
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
        if text == "":
            if data_lines:
                yield _build_sse_event(event, data_lines)
            event = "message"
            data_lines = []
            continue
        if text.startswith(":"):
            continue
        if text.startswith("event:"):
            event = text[6:].strip() or "message"
            continue
        if text.startswith("data:"):
            data_lines.append(text[5:].strip())
    if data_lines:
        yield _build_sse_event(event, data_lines)


def _build_sse_event(event: str, data_lines: list[str]) -> JesterkyStreamEvent:
    raw_data = "\n".join(data_lines)
    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError:
        data = raw_data
    return JesterkyStreamEvent(event=event, raw_data=raw_data, data=data)
