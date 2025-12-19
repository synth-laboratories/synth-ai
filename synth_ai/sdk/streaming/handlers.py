from __future__ import annotations

import contextlib
import json
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import click

from .types import StreamMessage, StreamType


def _mask_sensitive_urls(text: str) -> str:
    """Mask S3/Wasabi URLs and sensitive paths in log messages.
    
    Replaces full S3/Wasabi URLs with masked versions to prevent leaking
    bucket names, paths, and infrastructure details in public SDK logs.
    
    Examples:
        s3://synth-artifacts/models/... -> s3://***/***/[masked]
        Wasabi s3://bucket/path/file.tar.gz -> Wasabi s3://***/***/[masked]
    """
    if not text:
        return text
    
    # Pattern matches:
    # - Optional "Wasabi " prefix
    # - s3:// or http(s):// scheme
    # - Any bucket/host
    # - Any path
    # - Common model file extensions
    pattern = r'(Wasabi\s+)?((s3|https?)://[^\s]+\.(tar\.gz|zip|pt|pth|safetensors|ckpt|bin))'
    
    def replace_url(match: re.Match) -> str:
        prefix = match.group(1) or ""  # "Wasabi " or empty
        url = match.group(2)
        # Extract just the filename
        filename = url.split("/")[-1] if "/" in url else "file"
        return f'{prefix}s3://***/***/[{filename}]'
    
    return re.sub(pattern, replace_url, text, flags=re.IGNORECASE)


class StreamHandler(ABC):
    """Base class for log handlers that consume ``StreamMessage`` objects."""

    @abstractmethod
    def handle(self, message: StreamMessage) -> None:
        """Process a message produced by the streamer."""

    def should_handle(self, message: StreamMessage) -> bool:  # pragma: no cover - trivial
        """Predicate allowing handlers to filter messages before processing."""
        return True

    def flush(self) -> None:  # pragma: no cover - optional
        """Flush buffered output."""
        return None


class CLIHandler(StreamHandler):
    """Simple CLI output mirroring current poller behaviour."""

    def __init__(
        self,
        *,
        hidden_event_types: set[str] | None = None,
        hidden_event_substrings: set[str] | None = None,
    ) -> None:
        self._hidden_event_types = set(hidden_event_types or set())
        self._hidden_event_substrings = {s.lower() for s in (hidden_event_substrings or set())}

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        if message.stream_type is StreamType.STATUS:
            status = str(message.data.get("status") or message.data.get("state") or "unknown")
            click.echo(f"[{timestamp}] status={status}")
            return

        if message.stream_type is StreamType.EVENTS:
            event_type = message.data.get("type", "event")
            if event_type in self._hidden_event_types:
                return
            level = message.data.get("level")
            msg = message.data.get("message") or ""
            # Evaluate substring filters against lower-cased concatenated text
            if self._hidden_event_substrings:
                blob = " ".join(
                    [
                        event_type or "",
                        str(msg),
                        json.dumps(message.data.get("data", "")),
                    ]
                ).lower()
                if any(sub in blob for sub in self._hidden_event_substrings):
                    return
            prefix = f"[{timestamp}] [{message.seq}] {event_type}"
            if level:
                prefix += f" ({level})"
            # Mask sensitive URLs before displaying
            sanitized_msg = _mask_sensitive_urls(msg)
            
            # For error events, show full details including underlying errors
            if level == "error" or event_type.endswith(".failed"):
                click.echo(f"{prefix}: {sanitized_msg}")
                # Show error details from data field if available
                data = message.data.get("data", {})
                if isinstance(data, dict):
                    error_detail = data.get("detail") or data.get("error") or data.get("error_detail")
                    if error_detail and str(error_detail) != sanitized_msg:
                        # Show underlying error if different from main message
                        click.echo(f"    Error details: {error_detail}")
                    # Show traceback or stack if available
                    traceback_info = data.get("traceback") or data.get("stack")
                    if traceback_info:
                        lines = str(traceback_info).split("\n")
                        # Show last few lines of traceback (most relevant)
                        for line in lines[-5:]:
                            if line.strip():
                                click.echo(f"    {line}")
            else:
                click.echo(f"{prefix}: {sanitized_msg}".rstrip(": "))

            data = message.data.get("data") if isinstance(message.data.get("data"), dict) else {}
            if event_type == "prompt.learning.mipro.complete" and data:
                best_prompt = data.get("best_prompt")
                if isinstance(best_prompt, dict):
                    sections = best_prompt.get("sections")
                    if isinstance(sections, list) and sections:
                        click.echo("    --- BEST PROMPT ---")
                        for section in sections:
                            if not isinstance(section, dict):
                                continue
                            role = section.get("role", "unknown").upper()
                            name = section.get("name")
                            header = f"    [{role}]"
                            if name:
                                header += f" {name}"
                            click.echo(header)
                            content = section.get("content", "")
                            if isinstance(content, str) and content:
                                click.echo(f"        {content}")
                        click.echo("    -------------------")

            if event_type == "mipro.topk.evaluated" and data:
                rank = data.get("rank")
                train_score = data.get("train_score")
                test_score = data.get("test_score")
                instruction_text = data.get("instruction_text", "")
                demo_indices = data.get("demo_indices", [])
                lift_abs = data.get("lift_absolute")
                lift_pct = data.get("lift_percent")
                stage_payloads = data.get("stage_payloads", {})
                details: list[str] = []
                if rank is not None:
                    details.append(f"Rank {rank}")
                if isinstance(train_score, int | float):
                    train_score_float = float(train_score)
                    details.append(f"train={train_score_float:.3f} ({train_score_float*100:.1f}%)")
                if isinstance(test_score, int | float):
                    test_score_float = float(test_score)
                    details.append(f"test={test_score_float:.3f} ({test_score_float*100:.1f}%)")
                if isinstance(lift_abs, int | float) and isinstance(lift_pct, int | float):
                    details.append(f"lift={lift_abs:+.3f} ({lift_pct:+.1f}%)")
                if details:
                    click.echo("    --- TOP-K CANDIDATE ---")
                    click.echo(f"    {' | '.join(details)}")
                    if isinstance(instruction_text, str) and instruction_text.strip():
                        snippet = instruction_text.strip()
                        click.echo(f"        Instruction: {snippet}")
                    if isinstance(demo_indices, list) and demo_indices:
                        click.echo(f"        Demo indices: {demo_indices}")
                    
                    # Display per-stage information if available
                    if isinstance(stage_payloads, dict) and stage_payloads:
                        click.echo("        Per-stage breakdown:")
                        for stage_id, payload in stage_payloads.items():
                            if isinstance(payload, dict):
                                module_id = payload.get("module_id", stage_id)
                                instr_ids = payload.get("instruction_indices", [])
                                demo_ids = payload.get("demo_indices", [])
                                click.echo(f"          [{module_id}/{stage_id}] instr_ids={instr_ids} demo_ids={demo_ids}")
                    
                    seed_scores = data.get("test_seed_scores")
                    if isinstance(seed_scores, list) and seed_scores:
                        formatted_scores = ", ".join(
                            f"{item.get('seed')}: {item.get('score'):.2f}"
                            for item in seed_scores
                            if isinstance(item, dict) and isinstance(item.get("seed"), int) and isinstance(item.get("score"), int | float)
                        )
                        if formatted_scores:
                            click.echo(f"        Test per-seed: {formatted_scores}")
                    click.echo("    ----------------------")
            return

        if message.stream_type is StreamType.METRICS:
            name = message.data.get("name")
            value = message.data.get("value")
            step = message.data.get("step")
            data = message.data.get("data", {})
            
            # Format metric display
            metric_str = f"[{timestamp}] [metric] {name}={value:.4f}" if isinstance(value, int | float) else f"[{timestamp}] [metric] {name}={value}"
            if step is not None:
                metric_str += f" (step={step})"
            
            # Add any additional context from data field
            if isinstance(data, dict):
                n = data.get("n")
                if n is not None:
                    metric_str += f" n={n}"
            
            click.echo(metric_str)
            return

        if message.stream_type is StreamType.TIMELINE:
            phase = message.data.get("phase", "phase")
            click.echo(f"[{timestamp}] timeline={phase}")


class JSONHandler(StreamHandler):
    """Emit messages as JSON lines suitable for machine parsing."""

    def __init__(self, output_file: str | None = None, *, indent: int | None = None) -> None:
        self.output_file = Path(output_file).expanduser() if output_file else None
        self._indent = indent

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        payload: dict[str, Any] = {
            "stream_type": message.stream_type.name,
            "timestamp": message.timestamp,
            "job_id": message.job_id,
            "data": message.data,
        }
        if message.seq is not None:
            payload["seq"] = message.seq
        if message.step is not None:
            payload["step"] = message.step
        if message.phase is not None:
            payload["phase"] = message.phase

        line = json.dumps(payload, indent=self._indent)
        if self.output_file:
            with self.output_file.open("a", encoding="utf-8") as fh:
                fh.write(line)
                if self._indent is None:
                    fh.write("\n")
        else:
            click.echo(line)

    def flush(self) -> None:
        return None


class CallbackHandler(StreamHandler):
    """Invoke user-provided callbacks for specific stream types."""

    def __init__(
        self,
        *,
        on_status: Callable[[dict[str, Any]], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_metric: Callable[[dict[str, Any]], None] | None = None,
        on_timeline: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_status = on_status
        self._on_event = on_event
        self._on_metric = on_metric
        self._on_timeline = on_timeline

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        if message.stream_type is StreamType.STATUS and self._on_status:
            self._on_status(message.data)
        elif message.stream_type is StreamType.EVENTS and self._on_event:
            self._on_event(message.data)
        elif message.stream_type is StreamType.METRICS and self._on_metric:
            self._on_metric(message.data)
        elif message.stream_type is StreamType.TIMELINE and self._on_timeline:
            self._on_timeline(message.data)


class BufferedHandler(StreamHandler):
    """Collect messages and emit them in batches."""

    def __init__(self, *, flush_interval: float = 5.0, max_buffer_size: int = 100) -> None:
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self._buffer: list[StreamMessage] = []
        self._last_flush = time.time()

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        self._buffer.append(message)
        now = time.time()
        if len(self._buffer) >= self.max_buffer_size or now - self._last_flush >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.process_batch(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()

    def process_batch(self, messages: list[StreamMessage]) -> None:  # pragma: no cover - abstract
        """Override to define how buffered messages should be processed."""


class IntegrationTestHandler(StreamHandler):
    """Collect messages for integration tests or programmatic assertions."""

    def __init__(self) -> None:
        self.messages: list[StreamMessage] = []

    def handle(self, message: StreamMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()


class GraphGenHandler(StreamHandler):
    """Handler for ADAS jobs that delegate child job streams to an underlying handler.
    
    ADAS jobs emit events from child jobs (GEPA, MIPRO, RL, SFT, etc.). This handler
    provides light ADAS-aware filtering and routing while keeping child job output
    intact via a delegate handler. The delegate can be supplied directly or created
    via a factory; by default we choose a prompt-learning handler for GEPA/MIPRO and
    a basic CLI handler for other job types.
    """

    def __init__(
        self,
        *,
        child_handler: StreamHandler | None = None,
        child_handler_factory: Callable[[str | None], StreamHandler | None] | None = None,
        show_trial_results: bool = True,
        show_transformations: bool = False,
        show_validation: bool = True,
        filter_verbose_events: bool = True,
        wrap_child_events: bool = True,
    ) -> None:
        # User-supplied delegate or factory; both are optional.
        self.child_handler = child_handler
        self._child_handler_factory = child_handler_factory

        # Options for the default prompt-learning delegate
        self._pl_show_trial_results = show_trial_results
        self._pl_show_transformations = show_transformations
        self._pl_show_validation = show_validation

        self.filter_verbose_events = filter_verbose_events
        # If False, skip ADAS-specific filtering/transformations and just pass through.
        self.wrap_child_events = wrap_child_events

        # Detected child job type (gepa/mipro/rl/sft/etc.)
        self.child_job_type: str | None = None
        # Track whether we created the delegate automatically (so we can swap if needed)
        self._delegate_auto_created = False

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        if message.stream_type is StreamType.EVENTS:
            self._detect_child_job_type(message)
            self._maybe_reset_delegate_for_child_type()

            if self.wrap_child_events and self.filter_verbose_events:
                if self._should_filter_event(message):
                    return

            if self.wrap_child_events:
                message = self._transform_event_message(message)

        delegate = self._get_child_handler()
        if delegate:
            delegate.handle(message)

    def _get_child_handler(self) -> StreamHandler:
        """Return or create the delegate handler used for child job events."""
        if self.child_handler:
            return self.child_handler

        handler: StreamHandler | None = None
        if self._child_handler_factory:
            handler = self._child_handler_factory(self.child_job_type)

        if handler is None:
            # Choose a sensible default based on detected child job type
            if self._is_prompt_learning_type(self.child_job_type):
                handler = PromptLearningHandler(
                    show_trial_results=self._pl_show_trial_results,
                    show_transformations=self._pl_show_transformations,
                    show_validation=self._pl_show_validation,
                )
            else:
                handler = CLIHandler()

        self.child_handler = handler
        self._delegate_auto_created = self._child_handler_factory is None and self.child_handler is not None
        return handler

    def _detect_child_job_type(self, message: StreamMessage) -> None:
        """Infer the child job type from event types."""
        if self.child_job_type:
            return

        event_type = str(message.data.get("type") or "").lower()
        if not event_type:
            return

        if event_type.startswith("graph_evolve."):
            self.child_job_type = "graph_evolve"
        elif "mipro" in event_type:
            self.child_job_type = "mipro"
        elif "gepa" in event_type or event_type.startswith("prompt.learning"):
            self.child_job_type = "prompt_learning"
        elif event_type.startswith("rl.") or ".rl." in event_type:
            self.child_job_type = "rl"
        elif event_type.startswith("sft.") or ".sft." in event_type:
            self.child_job_type = "sft"
        else:
            # Fall back to the first segment as a hint (e.g., "adas.child_type")
            parts = event_type.split(".")
            if parts:
                self.child_job_type = parts[0]

    def _maybe_reset_delegate_for_child_type(self) -> None:
        """Swap out auto-created delegates when we later detect a different child type."""
        if not self.child_handler or not self._delegate_auto_created:
            return

        # If the detected type does not match the current delegate choice, rebuild.
        wants_prompt_learning = self._is_prompt_learning_type(self.child_job_type)
        has_prompt_learning_handler = isinstance(self.child_handler, PromptLearningHandler)

        if wants_prompt_learning and not has_prompt_learning_handler:
            self.child_handler = None
            self._delegate_auto_created = False
        elif not wants_prompt_learning and has_prompt_learning_handler:
            self.child_handler = None
            self._delegate_auto_created = False

    def _should_filter_event(self, message: StreamMessage) -> bool:
        """Determine if an event should be hidden from output."""
        event_type = message.data.get("type", "") or ""
        event_type_lower = event_type.lower()

        # Never filter graph_evolve events - they're important for GraphGen jobs
        if event_type.startswith("graph_evolve."):
            return False

        # Only filter prompt-learning style events; leave other job types untouched.
        if not any(key in event_type_lower for key in ("prompt.learning", "gepa", "mipro")):
            return False

        important_events = {
            "prompt.learning.created",
            "prompt.learning.gepa.start",
            "prompt.learning.gepa.complete",
            "prompt.learning.mipro.job.started",
            "prompt.learning.mipro.optimization.exhausted",
            "prompt.learning.trial.results",
            "prompt.learning.progress",
            "prompt.learning.gepa.new_best",
            "prompt.learning.validation.summary",
            "prompt.learning.candidate.evaluated",
            "prompt.learning.candidate.evaluation.started",
            # GraphGen/graph_evolve important events
            "graph_evolve.job_started",
            "graph_evolve.generation_started",
            "graph_evolve.generation_completed",
            "graph_evolve.candidate_evaluated",
            "graph_evolve.archive_updated",
            "graph_evolve.job_completed",
            "graph_evolve.job_failed",
        }
        if event_type in important_events:
            return False

        verbose_patterns = [
            "gepa.transformation.proposed",
            "gepa.proposal.scored",
            "prompt.learning.proposal.scored",
            "mipro.tpe.update",
            "prompt.learning.stream.connected",
        ]
        return any(pattern in event_type_lower for pattern in verbose_patterns)

    def _transform_event_message(self, message: StreamMessage) -> StreamMessage:
        """Transform event messages for ADAS context (currently passthrough)."""
        return message

    def flush(self) -> None:
        # Ensure delegate flushes buffered output if needed.
        if self.child_handler and hasattr(self.child_handler, "flush"):
            with contextlib.suppress(Exception):
                self.child_handler.flush()

    @staticmethod
    def _is_prompt_learning_type(job_type: str | None) -> bool:
        """Return True if the child job type should use prompt-learning formatting."""
        return job_type in {"gepa", "mipro", "prompt_learning", "prompt-learning", None}


class LossCurveHandler(StreamHandler):
    """Render a live-updating loss chart inside a fixed Rich panel."""

    def __init__(
        self,
        *,
        metric_name: str = "train.loss",
        max_points: int = 200,
        width: int = 60,
        console: Any | None = None,
        live: Any | None = None,
    ) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.text import Text
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "LossCurveHandler requires the 'rich' package. Install synth-ai[all] or rich>=13."
            ) from exc

        self.metric_name = metric_name
        self.max_points = max_points
        self.width = width

        self._console_class = Console
        self._panel_class = Panel
        self._text_class = Text

        self._console = console or Console()
        self._live = live or Live(console=self._console, transient=False, refresh_per_second=8)
        self._started = False

        self._steps: list[int] = []
        self._values: list[float] = []
        self._status = "waiting"
        self._last_event: str | None = None

    def handle(self, message: StreamMessage) -> None:
        updated = False

        if message.stream_type is StreamType.STATUS:
            status = str(message.data.get("status") or message.data.get("state") or "unknown")
            if status != self._status:
                self._status = status
                updated = True

        elif message.stream_type is StreamType.EVENTS:
            event_type = message.data.get("type", "")
            msg = message.data.get("message") or ""
            level = message.data.get("level")
            summary = f"{event_type}".strip()
            if level:
                summary += f" ({level})"
            if msg:
                summary += f": {msg}"
            if summary != self._last_event:
                self._last_event = summary
                updated = True

        elif message.stream_type is StreamType.METRICS:
            if message.data.get("name") != self.metric_name:
                return
            value = message.data.get("value")
            step = message.data.get("step")
            if not isinstance(value, int | float) or not isinstance(step, int):
                return
            self._values.append(float(value))
            self._steps.append(step)
            if len(self._values) > self.max_points:
                self._values = self._values[-self.max_points :]
                self._steps = self._steps[-self.max_points :]
            updated = True

        elif message.stream_type is StreamType.TIMELINE:
            phase = message.data.get("phase")
            if phase:
                self._status = str(phase)
                updated = True

        if updated:
            self._refresh()

    def flush(self) -> None:
        if self._started:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._started = False

    def _ensure_live(self) -> None:
        if not self._started:
            with contextlib.suppress(Exception):
                self._live.start()
            self._started = True

    def _refresh(self) -> None:
        self._ensure_live()
        body = self._build_body()
        title = f"{self.metric_name} | status={self._status}"
        self._live.update(self._panel_class(body, title=title, border_style="cyan"))

    def _build_body(self) -> Any:
        if not self._values:
            return self._text_class("Waiting for metrics…", style="yellow")

        chart = self._render_sparkline()
        last_value = self._values[-1]
        lines = [
            chart,
            f"latest: {last_value:.4f} (step {self._steps[-1]})",
        ]
        if self._last_event:
            lines.append(f"event: {self._last_event}")
        return "\n".join(lines)

    def _render_sparkline(self) -> str:
        blocks = "▁▂▃▄▅▆▇█"
        tail_len = min(self.width, len(self._values))
        tail = self._values[-tail_len:]
        minimum = min(tail)
        maximum = max(tail)
        if maximum == minimum:
            level = blocks[0]
            return f"{minimum:.2f} {level * tail_len} {maximum:.2f}"
        scale = (len(blocks) - 1) / (maximum - minimum)
        chars = "".join(blocks[int((v - minimum) * scale + 0.5)] for v in tail)
        return f"{minimum:.2f} {chars} {maximum:.2f}"

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.flush()

class RichHandler(StreamHandler):
    """Rich powered handler with live progress and metrics table."""

    def __init__(
        self,
        *,
        event_log_size: int = 20,
        console: Any | None = None,
    ) -> None:
        try:
            from rich.console import Console
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
            from rich.table import Table
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "RichHandler requires the 'rich' package. Install synth-ai[all] or rich>=13."
            ) from exc

        self._console_class = Console
        self._progress_class = Progress
        self._spinner_column = SpinnerColumn
        self._text_column = TextColumn
        self._bar_column = BarColumn
        self._table_class = Table

        self._console = console or Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}" if console else ""),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=False,
            console=self._console,
        )
        self._task_id: int | None = None
        self._current_status = "unknown"
        self._latest_metrics: dict[str, Any] = {}
        self._event_log: deque[str] = deque(maxlen=event_log_size)
        self._progress_started = False

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        if message.stream_type is StreamType.STATUS:
            self._current_status = str(message.data.get("status") or message.data.get("state"))
            self._ensure_progress_started()
            if self._task_id is not None:
                description = f"Status: {self._current_status}"
                self._progress.update(self._task_id, description=description)  # type: ignore[arg-type]
            self._render_summary()
            return

        if message.stream_type is StreamType.EVENTS:
            event_type = message.data.get("type", "event")
            summary = message.data.get("message") or ""
            level = message.data.get("level")
            # Mask sensitive URLs before displaying
            sanitized_summary = _mask_sensitive_urls(summary)
            formatted = f"[{event_type}] {sanitized_summary}".strip()
            if level:
                formatted = f"{formatted} ({level})"
            self._event_log.append(formatted)
            data = message.data.get("data") or {}
            step = data.get("step") or data.get("current_step")
            total_steps = data.get("total_steps") or data.get("max_steps")
            if step and total_steps:
                self._ensure_progress_started(total_steps)
                if self._task_id is not None:
                    self._progress.update(self._task_id, completed=int(step), total=int(total_steps))  # type: ignore[arg-type]
            self._render_summary()
            return

        if message.stream_type is StreamType.METRICS:
            name = message.data.get("name", "")
            value = message.data.get("value")
            if name:
                self._latest_metrics[name] = value
            self._render_summary()
            return

        if message.stream_type is StreamType.TIMELINE:
            phase = message.data.get("phase", "")
            if phase and phase.lower() not in {"training", "running"}:
                self._event_log.append(f"[timeline] {phase}")
            self._render_summary()

    def flush(self) -> None:
        if self._progress_started:
            self._progress.stop()
            self._progress_started = False
        self._render_summary(force=True)

    def _ensure_progress_started(self, total: int | float | None = None) -> None:
        if not self._progress_started:
            self._progress.start()
            self._progress_started = True
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                f"Status: {self._current_status}", total=total or 100
            )
        elif total is not None and self._task_id is not None:
            self._progress.update(self._task_id, total=total)  # type: ignore[arg-type]

    def _render_summary(self, force: bool = False) -> None:
        if force and self._progress_started:
            self._progress.refresh()

        table = self._table_class(title="Latest Metrics")
        table.add_column("Metric")
        table.add_column("Value")

        if not self._latest_metrics:
            table.add_row("—", "—")
        else:
            for name, value in sorted(self._latest_metrics.items()):
                table.add_row(str(name), str(value))

        if self._progress_started:
            self._progress.console.print(table)
        else:
            self._console.print(table)

        if self._event_log:
            self._console.print("\nRecent events:")
            for entry in list(self._event_log):
                self._console.print(f"  • {entry}")

class ContextLearningHandler(StreamHandler):
    """CLI-friendly handler for Context Learning jobs.

    Emits high-signal progress similar to other infra job handlers,
    specialized for generation-based bash context optimization.
    """

    def __init__(self) -> None:
        self.best_score_so_far = 0.0
        self.current_generation = 0

    def handle(self, message: StreamMessage) -> None:
        if not self.should_handle(message):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        if message.stream_type is StreamType.STATUS:
            status = str(message.data.get("status") or message.data.get("state") or "unknown")
            click.echo(f"[{timestamp}] status={status}")
            return

        if message.stream_type is StreamType.METRICS:
            name = message.data.get("name")
            value = message.data.get("value")
            step = message.data.get("step")
            if isinstance(value, int | float):
                try:
                    val_f = float(value)
                    if val_f > self.best_score_so_far:
                        self.best_score_so_far = val_f
                    if isinstance(step, int):
                        self.current_generation = max(self.current_generation, step)
                    click.echo(f"[{timestamp}] gen={step} best={val_f:.3f}")
                    return
                except Exception:
                    pass
            click.echo(f"[{timestamp}] metric {name}={value}")
            return

        if message.stream_type is StreamType.EVENTS:
            event_type = str(message.data.get("type") or "")
            msg = message.data.get("message") or ""
            data = message.data.get("data") or {}

            if event_type == "context.learning.generation.completed":
                gen = data.get("generation") or data.get("gen") or self.current_generation
                score = data.get("best_score") or data.get("score") or self.best_score_so_far
                try:
                    score_f = float(score)
                    if score_f > self.best_score_so_far:
                        self.best_score_so_far = score_f
                    click.echo(f"[{timestamp}] generation {gen} best={score_f:.3f}")
                except Exception:
                    click.echo(f"[{timestamp}] generation {gen} completed")
                return

            if event_type.endswith(".failed"):
                click.echo(f"[{timestamp}] {event_type}: {msg}")
                return

            if msg:
                click.echo(f"[{timestamp}] {event_type}: {msg}")
            else:
                click.echo(f"[{timestamp}] {event_type}")


class PromptLearningHandler(StreamHandler):
    """Enhanced handler for GEPA/MIPRO prompt optimization jobs with rich formatting and metrics tracking.
    
    This handler processes streaming events from both GEPA (Genetic Evolutionary Prompt
    Algorithm) and MIPRO (Meta-Instruction PROposer) optimization jobs. It provides:
    
    - **Real-time progress tracking**: Shows trial results, rollouts, iterations, and budget usage
    - **Optimization curve tracking**: Maintains a history of best scores over time
    - **GEPA-specific features**: Tracks transformations, rollouts, and validation results
    - **MIPRO-specific features**: Tracks iterations, trials, minibatch/full evaluations, and budget
    - **Dual output**: Writes to both console (via click.echo) and optional log file
    
    The handler filters verbose events (like TPE updates, proposed instructions) to keep
    output readable while preserving important progress information. It formats output
    consistently between GEPA and MIPRO for easier comparison.
    
    Example:
        >>> handler = PromptLearningHandler(
        ...     show_trial_results=True,
        ...     max_tokens=1_000_000,
        ...     log_file=Path("optimization.log")
        ... )
        >>> # Handler is used by JobStreamer to process events
    """
    
    def __init__(
        self,
        *,
        show_trial_results: bool = True,
        show_transformations: bool = False,
        show_validation: bool = True,
        max_tokens: int | None = None,
        max_time_seconds: float | None = None,
        max_rollouts: int | None = None,
        log_file: Path | None = None,
    ):
        """Initialize the prompt learning handler.
        
        Args:
            show_trial_results: Whether to display individual trial scores (default: True).
                When True, shows each trial's score and best score so far.
            show_transformations: Whether to display transformation/proposal details
                (default: False). When True, shows verbose transformation events.
            show_validation: Whether to display validation summaries (default: True).
                Shows validation results comparing candidates against baseline.
            max_tokens: Maximum token budget for MIPRO (from TOML termination_config).
                Used to track progress and enforce limits.
            max_time_seconds: Maximum time budget in seconds (from TOML termination_config).
                Used to track elapsed time and ETA.
            max_rollouts: Maximum rollouts budget (from TOML termination_config).
                Used to track rollout progress for both GEPA and MIPRO.
            log_file: Optional path to log file for persistent logging. If provided,
                all output is written to both console and file. File is opened in
                append mode and remains open for streaming.
        """
        self.show_trial_results = show_trial_results
        self.show_transformations = show_transformations
        self.show_validation = show_validation
        self.optimization_curve: list[tuple[int, float]] = []
        self.trial_counter = 0
        self.best_score_so_far = 0.0
        
        # MIPRO progress tracking
        self.mipro_start_time: float | None = None
        self.mipro_total_trials: int | None = None
        self.mipro_completed_trials: int = 0
        self.mipro_total_tokens: int = 0
        self.mipro_policy_tokens: int = 0  # Rollout tokens (policy only)
        self.mipro_max_tokens: int | None = max_tokens  # From TOML termination_config
        self.mipro_total_cost: float = 0.0
        self.mipro_max_cost: float | None = None
        self.mipro_current_iteration: int = 0
        self.mipro_num_iterations: int | None = None
        self.mipro_trials_per_iteration: int | None = None
        self.mipro_best_score: float = 0.0  # Track best full eval score
        self.mipro_baseline_score: float | None = None  # Track baseline for comparison
        self.mipro_batch_size: int | None = None  # Track minibatch size (N for minibatch scores)
        self.mipro_rollouts_completed: int = 0  # Total rollouts completed
        self.mipro_max_rollouts: int | None = max_rollouts  # From TOML termination_config
        self.mipro_max_time_seconds: float | None = max_time_seconds  # From TOML termination_config
        self._last_progress_emit_time: float | None = None  # Throttle progress updates
        self._progress_emit_interval: float = 5.0  # Emit progress at most every 5 seconds
        
        # Log file for real-time streaming
        self.log_file: Path | None = log_file
        self._log_file_handle = None
        if self.log_file:
            try:
                # Create parent directory if needed
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                # Open file in append mode for live streaming
                # Note: File must remain open for streaming, so we can't use context manager
                from datetime import datetime
                self._log_file_handle = open(self.log_file, "a", encoding="utf-8")  # noqa: SIM115
                # Write header
                self._log_file_handle.write("=" * 80 + "\n")
                self._log_file_handle.write("PROMPT LEARNING VERBOSE LOG\n")
                self._log_file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._log_file_handle.write("=" * 80 + "\n\n")
                self._log_file_handle.flush()
            except Exception as e:
                # If we can't open the log file, continue without it
                click.echo(f"⚠️  Could not open log file {log_file}: {e}", err=True)
                self.log_file = None
                self._log_file_handle = None
    
    def _write_log(self, text: str) -> None:
        """Write text to both console and log file."""
        click.echo(text)
        if self._log_file_handle:
            try:
                self._log_file_handle.write(text + "\n")
                self._log_file_handle.flush()
            except Exception:
                # If write fails, close handle and continue without logging
                from contextlib import suppress
                with suppress(Exception):
                    self._log_file_handle.close()
                self._log_file_handle = None
    
    def handle(self, message: StreamMessage) -> None:
        """Handle a stream message from the prompt learning job.
        
        Routes messages to appropriate handlers based on stream type:
        - STATUS: Job status updates (queued, running, completed, etc.)
        - EVENTS: Algorithm-specific events (trials, iterations, transformations)
        - METRICS: Performance metrics (scores, accuracies, costs)
        - TIMELINE: Phase transitions
        
        Filters verbose events (TPE updates, proposed instructions) to keep output
        readable. MIPRO and GEPA events are handled by specialized methods.
        
        Args:
            message: StreamMessage containing event data from the backend
        """
        if not self.should_handle(message):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message.stream_type is StreamType.STATUS:
            status = str(message.data.get("status") or message.data.get("state") or "unknown")
            self._write_log(f"[{timestamp}] status={status}")
            return
        
        if message.stream_type is StreamType.EVENTS:
            event_type = message.data.get("type", "event")
            level = message.data.get("level")
            msg = message.data.get("message") or ""
            
            # Handle MIPRO-specific events for progress tracking (before skipping hidden events)
            if event_type == "mipro.job.started":
                self._handle_mipro_job_started(message.data)
                # Continue to default display
            
            if event_type == "mipro.budget.update":
                self._handle_mipro_budget_update(message.data)
                # Continue to default display
            
            if event_type == "mipro.trial.complete":
                self._handle_mipro_trial_complete(message.data)
                # Continue to default display
            
            # Show more MIPRO events - only hide the most verbose ones
            _hidden_mipro_events = {
                # Keep only the most verbose TPE updates hidden
                "mipro.tpe.update",  # Very frequent, low value
            }
            if event_type in _hidden_mipro_events:
                return
            
            # Show GEPA transformation proposals - they're useful for debugging
            # if event_type == "gepa.transformation.proposed":
            #     return
            
            # Handle trial results for optimization curve tracking
            if event_type == "prompt.learning.trial.results":
                self._handle_trial_results(message.data)
                # Continue to default display
            
            # Handle validation summary
            if event_type == "prompt.learning.validation.summary":
                if self.show_validation:
                    self._handle_validation_summary(message.data)
                # Continue to default display
            
            # Handle progress events
            if event_type == "prompt.learning.progress":
                self._handle_progress(message.data)
                # Continue to default display
            
            # Handle MIPRO-specific events for progress tracking
            if event_type == "mipro.iteration.start":
                self._handle_mipro_iteration_start(message.data)
                # Continue to default display
            
            if event_type == "mipro.iteration.complete":
                self._handle_mipro_iteration_complete(message.data)
                # Continue to default display
            
            if event_type == "mipro.fulleval.complete":
                self._handle_mipro_fulleval_complete(message.data)
                # Continue to default display
            
            if event_type == "mipro.optimization.exhausted":
                # Graceful conclusion - show final progress
                self._emit_mipro_progress()
                # Continue to default display
            
            if event_type == "mipro.new_incumbent":
                self._handle_mipro_new_incumbent(message.data)
                # Continue to default display
            
            # Handle rollouts start event
            if event_type == "prompt.learning.rollouts.start":
                self._handle_rollouts_start(message.data)
                # Continue to default display

            # Handle GEPA new best event
            if event_type == "prompt.learning.gepa.new_best":
                self._handle_gepa_new_best(message.data)
                # Continue to default display

            # Handle phase changed event
            if event_type == "prompt.learning.phase.changed":
                self._handle_phase_changed(message.data)
                # Continue to default display

            # Handle stream connected event (connection lifecycle)
            if event_type == "prompt.learning.stream.connected":
                self._handle_stream_connected(message.data)
                # Continue to default display

            # Handle proposal scored events (transformations) - show by default
            if event_type == "prompt.learning.proposal.scored":
                self._handle_proposal_scored(message.data)
                # Continue to default display
            
            # Show verbose transformation events by default - they're useful
            # Only skip if explicitly disabled via show_transformations=False
            # verbose_event_types = [
            #     "prompt.learning.proposal.scored",
            #     "prompt.learning.eval.summary",
            #     "prompt.learning.validation.scored",
            #     "prompt.learning.final.results",
            # ]
            # if event_type in verbose_event_types and not self.show_transformations:
            #     return
            
            # Default event display - show more details
            prefix = f"[{timestamp}] {event_type}"
            if level:
                prefix += f" ({level})"
            sanitized_msg = _mask_sensitive_urls(msg)
            
            # Include key data fields if message is empty or short
            if not sanitized_msg or len(sanitized_msg) < 50:
                data = message.data.get("data", {})
                if isinstance(data, dict):
                    # Show useful fields
                    useful_fields = []
                    for key in ["score", "accuracy", "mean", "step", "iteration", "trial", "completed", "total", "version_id"]:
                        if key in data:
                            value = data[key]
                            if isinstance(value, (int, float)):
                                useful_fields.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
                            else:
                                useful_fields.append(f"{key}={value}")
                    if useful_fields:
                        sanitized_msg = sanitized_msg + (" " if sanitized_msg else "") + " ".join(useful_fields[:5])  # Limit to 5 fields
            
            self._write_log(f"{prefix}: {sanitized_msg}".rstrip(": "))
            return
        
        if message.stream_type is StreamType.METRICS:
            name = message.data.get("name")
            value = message.data.get("value")
            step = message.data.get("step")
            data = message.data.get("data", {})
            
            metric_str = f"[{timestamp}] [metric] {name}={value:.4f}" if isinstance(value, int | float) else f"[{timestamp}] [metric] {name}={value}"
            if step is not None:
                metric_str += f" (step={step})"
            
            if isinstance(data, dict):
                n = data.get("n")
                if n is not None:
                    metric_str += f" n={n}"
            
            self._write_log(metric_str)
            return
        
        if message.stream_type is StreamType.TIMELINE:
            phase = message.data.get("phase", "phase")
            self._write_log(f"[{timestamp}] timeline={phase}")
    
    def _handle_trial_results(self, event_data: dict[str, Any]) -> None:
        """Handle GEPA trial results events and track optimization curve.
        
        Processes trial completion events from GEPA optimization, tracking:
        - Mean score for the trial
        - Best score achieved so far
        - Number of rollouts completed (N)
        - Optimization curve data points
        
        Updates the optimization curve with (trial_number, best_score) tuples
        for visualization. Displays trial results if show_trial_results is True.
        
        Args:
            event_data: Event data dictionary containing:
                - data.mean: Mean score for this trial
                - data.completed: Number of rollouts completed
                - data.total: Total rollouts planned
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        mean_score = data.get("mean")
        if mean_score is not None:
            self.trial_counter += 1
            self.best_score_so_far = max(self.best_score_so_far, float(mean_score))
            self.optimization_curve.append((self.trial_counter, self.best_score_so_far))
            
            if self.show_trial_results:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Extract N (number of rollouts)
                completed = data.get("completed")
                total = data.get("total")
                
                n_str = f" N={completed}/{total}" if completed is not None and total is not None else (f" N={completed}" if completed is not None else "")
                
                self._write_log(f"[{timestamp}] [Trial {self.trial_counter}] Score: {mean_score:.4f} (Best: {self.best_score_so_far:.4f}){n_str}")
    
    def _handle_validation_summary(self, event_data: dict[str, Any]) -> None:
        """Handle validation summary events showing candidate performance.
        
        Displays validation results comparing optimized prompts against a baseline.
        Shows baseline score, number of candidates evaluated (N), and top candidate
        scores. Only displayed if show_validation is True.
        
        Args:
            event_data: Event data dictionary containing:
                - data.baseline: Baseline score (dict with accuracy/score or number)
                - data.results: List of candidate results with accuracy/score fields
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Extract baseline
        baseline = data.get("baseline")
        baseline_score = None
        if isinstance(baseline, dict):
            baseline_score = baseline.get("accuracy") or baseline.get("score")
        elif isinstance(baseline, int | float):
            baseline_score = baseline
        
        # Extract results
        results = data.get("results", [])
        if not isinstance(results, list):
            results = []
        
        # Display validation summary
        self._write_log(f"[{timestamp}] Validation Summary:")
        
        # Show baseline if available
        if baseline_score is not None:
            self._write_log(f"  Baseline: {baseline_score:.4f}")
        
        # Show N (number of candidates)
        n_candidates = len(results)
        if n_candidates > 0:
            self._write_log(f"  N={n_candidates}")
        
        # Display validation results
        if results:
            for i, result in enumerate(results[:10]):  # Show top 10
                if isinstance(result, dict):
                    accuracy = result.get("accuracy") or result.get("score")
                    if accuracy is not None:
                        self._write_log(f"  Candidate {i+1}: {accuracy:.4f}")
    
    def _handle_progress(self, event_data: dict[str, Any]) -> None:
        """Handle GEPA progress events with detailed rollout and transformation tracking.
        
        Displays comprehensive progress information including:
        - Overall completion percentage
        - Rollout progress (completed/total with percentage)
        - Transformation progress (tried/planned with percentage)
        - Token usage (used/budget in millions)
        - Elapsed time and ETA
        
        Formats progress in a human-readable format similar to CLI progress bars.
        
        Args:
            event_data: Event data dictionary containing:
                - data.rollouts_completed: Number of rollouts completed
                - data.rollouts_total: Total rollouts planned
                - data.transformations_tried: Number of transformations tried
                - data.transformations_planned: Total transformations planned
                - data.rollout_tokens_used: Tokens consumed
                - data.rollout_tokens_budget: Token budget
                - data.elapsed_seconds: Time elapsed
                - data.eta_seconds: Estimated time remaining
                - data.percent_overall: Overall completion percentage
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Extract rollout progress
        rollouts_completed = data.get("rollouts_completed")
        rollouts_total = data.get("rollouts_total")
        percent_rollouts = data.get("percent_rollouts")
        
        # Extract transformation progress
        transformations_tried = data.get("transformations_tried")
        transformations_planned = data.get("transformations_planned")
        percent_transformations = data.get("percent_transformations")
        
        # Extract overall progress
        percent_overall = data.get("percent_overall")
        
        # Extract timing
        elapsed_seconds = data.get("elapsed_seconds")
        eta_seconds = data.get("eta_seconds")
        
        # Extract token usage
        rollout_tokens_used = data.get("rollout_tokens_used")
        rollout_tokens_budget = data.get("rollout_tokens_budget")
        
        # Build progress message
        parts = []
        
        # Overall percentage
        if percent_overall is not None:
            parts.append(f"{int(percent_overall * 100)}% complete")
        
        # Rollout progress
        if rollouts_completed is not None and rollouts_total is not None:
            parts.append(f"rollouts={rollouts_completed}/{rollouts_total}")
            if percent_rollouts is not None:
                parts.append(f"({int(percent_rollouts * 100)}%)")
        elif rollouts_completed is not None:
            parts.append(f"rollouts={rollouts_completed}")
        
        # Transformation progress
        if transformations_tried is not None and transformations_planned is not None:
            parts.append(f"transformations={transformations_tried}/{transformations_planned}")
            if percent_transformations is not None:
                parts.append(f"({int(percent_transformations * 100)}%)")
        elif transformations_tried is not None:
            parts.append(f"transformations={transformations_tried}")
        
        # Token usage
        if rollout_tokens_used is not None:
            tokens_millions = rollout_tokens_used / 1_000_000.0
            if rollout_tokens_budget is not None:
                budget_millions = rollout_tokens_budget / 1_000_000.0
                parts.append(f"tokens={tokens_millions:.2f}M/{budget_millions:.2f}M")
            else:
                parts.append(f"tokens={tokens_millions:.2f}M")
        
        # Timing
        if elapsed_seconds is not None:
            if elapsed_seconds >= 60:
                elapsed_str = f"{elapsed_seconds / 60:.1f}min"
            else:
                elapsed_str = f"{int(elapsed_seconds)}s"
            parts.append(f"elapsed={elapsed_str}")
        
        if eta_seconds is not None:
            eta_str = f"{eta_seconds / 60:.1f}min" if eta_seconds >= 60 else f"{int(eta_seconds)}s"
            parts.append(f"eta={eta_str}")
        
        # Fallback to simple step/total_steps if no detailed info
        if not parts:
            step = data.get("step") or data.get("current_step")
            total_steps = data.get("total_steps") or data.get("max_steps")
            if step is not None and total_steps is not None:
                parts.append(f"{step}/{total_steps} ({100 * step / total_steps:.1f}%)")
        
        if parts:
            progress_msg = " ".join(parts)
            self._write_log(f"[{timestamp}] Progress: {progress_msg}")
    
    def _handle_rollouts_start(self, event_data: dict[str, Any]) -> None:
        """Handle GEPA rollouts start event.
        
        Displays when rollouts begin, showing the number of training seeds
        that will be evaluated. This marks the start of the main optimization
        phase for GEPA.
        
        Args:
            event_data: Event data dictionary containing:
                - data.train_seeds: List of training seed values
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        train_seeds = data.get("train_seeds", [])
        
        if isinstance(train_seeds, list) and train_seeds:
            num_seeds = len(train_seeds)
            self._write_log(f"[{timestamp}] Starting rollouts: {num_seeds} seeds")
        else:
            self._write_log(f"[{timestamp}] Starting rollouts")

    def _handle_gepa_new_best(self, event_data: dict[str, Any]) -> None:
        """Handle GEPA new best candidate event.

        Displays when a new best candidate is found during optimization,
        showing the improvement over the previous best.

        Args:
            event_data: Event data dictionary containing:
                - data.accuracy: New best accuracy score
                - data.previous_best_score: Previous best score
                - data.improvement: Absolute improvement
                - data.version_id: ID of the new best candidate
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        accuracy = data.get("accuracy")
        previous = data.get("previous_best_score")
        improvement = data.get("improvement")

        if accuracy is not None:
            msg = f"[{timestamp}] \u2728 New best: {accuracy:.4f}"
            if previous is not None and improvement is not None:
                msg += f" (+{improvement:.4f} from {previous:.4f})"
            elif previous is not None:
                msg += f" (was {previous:.4f})"
            self._write_log(msg)

    def _handle_phase_changed(self, event_data: dict[str, Any]) -> None:
        """Handle phase transition event.

        Displays when the optimization transitions between phases
        (e.g., bootstrap -> optimization -> validation -> complete).

        Args:
            event_data: Event data dictionary containing:
                - data.from_phase: Previous phase name
                - data.to_phase: New phase name
                - data.phase_summary: Optional summary of completed phase
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        from_phase = data.get("from_phase") or "start"
        to_phase = data.get("to_phase")

        if to_phase:
            self._write_log(f"[{timestamp}] Phase: {from_phase} \u2192 {to_phase}")

    def _handle_stream_connected(self, event_data: dict[str, Any]) -> None:
        """Handle SSE stream connection event.

        Displays connection confirmation with cursor position for debugging.

        Args:
            event_data: Event data dictionary containing:
                - data.cursor: Current sequence cursor position
                - data.heartbeat_interval_seconds: Heartbeat interval
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        cursor = data.get("cursor", 0)
        self._write_log(f"[{timestamp}] Stream connected (cursor={cursor})")

    def _handle_mipro_job_started(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO job start event and extract configuration.
        
        Captures initial MIPRO configuration from the job start event to enable
        progress tracking. Extracts num_iterations and num_trials_per_iteration
        to estimate total trials and rollouts.
        
        Args:
            event_data: Event data dictionary containing:
                - data.num_iterations: Total number of optimization iterations
                - data.num_trials_per_iteration: Trials per iteration
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        # Extract config values to estimate max rollouts
        num_iterations = data.get("num_iterations")
        num_trials_per_iteration = data.get("num_trials_per_iteration")
        
        if num_iterations is not None:
            self.mipro_num_iterations = num_iterations
        if num_trials_per_iteration is not None:
            self.mipro_trials_per_iteration = num_trials_per_iteration
    
    def _handle_mipro_iteration_start(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO iteration start event and initialize progress tracking.
        
        Called at the start of each MIPRO iteration. On the first iteration (0),
        initializes all progress tracking variables including:
        - Total iterations and trials per iteration
        - Batch size (for minibatch evaluations)
        - Max rollouts estimate (iterations * trials * batch_size)
        - Time and token budgets
        
        Sets the start time for elapsed time tracking.
        
        Args:
            event_data: Event data dictionary containing:
                - data.iteration: Current iteration number (0-indexed)
                - data.num_iterations: Total iterations
                - data.num_trials_per_iteration: Trials per iteration
                - data.batch_size: Minibatch size (N for minibatch scores)
                - data.max_trials: Maximum trials limit (optional)
                - data.max_rollouts: Maximum rollouts limit (optional)
                - data.max_time_seconds: Maximum time limit (optional)
        """
        import time
        
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        iteration = data.get("iteration")
        if iteration == 0 and self.mipro_start_time is None:
            self.mipro_start_time = time.time()
        
        # Extract total iterations and trials per iteration from first iteration
        if iteration == 0:
            self.mipro_num_iterations = data.get("num_iterations") or self.mipro_num_iterations
            self.mipro_trials_per_iteration = data.get("num_trials_per_iteration") or self.mipro_trials_per_iteration
            batch_size = data.get("batch_size")
            if batch_size is not None:
                self.mipro_batch_size = batch_size
            
            if self.mipro_num_iterations and self.mipro_trials_per_iteration:
                self.mipro_total_trials = self.mipro_num_iterations * self.mipro_trials_per_iteration
            
            # Extract max limits if available (from events, but TOML value takes precedence)
            # Only override if TOML value wasn't set
            max_trials = data.get("max_trials")
            max_rollouts_from_event = data.get("max_rollouts")
            if self.mipro_max_rollouts is None:
                if max_rollouts_from_event is not None:
                    # Use event value if TOML value wasn't set
                    self.mipro_max_rollouts = max_rollouts_from_event
                elif max_trials is not None:
                    # Fallback: If max_trials is set, use it as max rollouts (approximation)
                    self.mipro_max_rollouts = max_trials
                elif self.mipro_num_iterations and self.mipro_trials_per_iteration and self.mipro_batch_size:
                    # Estimate max rollouts: iterations * trials_per_iteration * batch_size
                    self.mipro_max_rollouts = self.mipro_num_iterations * self.mipro_trials_per_iteration * self.mipro_batch_size
            
            max_time_seconds = data.get("max_time_seconds") or data.get("max_wall_clock_seconds")
            if max_time_seconds is not None and self.mipro_max_time_seconds is None:
                # Use event value only if TOML value wasn't set
                self.mipro_max_time_seconds = float(max_time_seconds)
        
        self.mipro_current_iteration = iteration if iteration is not None else self.mipro_current_iteration
    
    def _handle_mipro_iteration_complete(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO iteration completion event.
        
        Updates progress tracking when an iteration completes, including:
        - Cumulative trial count
        - Current iteration number
        
        Emits a progress update showing overall progress, trials completed,
        iterations, rollouts, tokens, and time.
        
        Args:
            event_data: Event data dictionary containing:
                - data.iteration: Completed iteration number
                - data.cumulative: Cumulative trial count across all iterations
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        cumulative = data.get("cumulative")
        if cumulative is not None:
            self.mipro_completed_trials = cumulative
        
        # Update current iteration
        iteration = data.get("iteration")
        if iteration is not None:
            self.mipro_current_iteration = iteration
        
        # Emit progress update
        self._emit_mipro_progress()
    
    def _handle_mipro_trial_complete(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO trial completion event (minibatch evaluation).
        
        Processes minibatch trial completion events, which occur frequently during
        MIPRO optimization. Tracks:
        - Completed trial count
        - Rollouts completed (from num_seeds)
        - Minibatch scores (displayed if show_trial_results is True)
        
        Displays trial results in GEPA-like format: [Trial X] Score: Y (Best: Z) N=W
        where N is the minibatch size. Emits throttled progress updates.
        
        Args:
            event_data: Event data dictionary containing:
                - data.minibatch_score: Score from minibatch evaluation
                - data.iteration: Current iteration number
                - data.trial: Trial number within iteration
                - data.num_seeds: Number of seeds evaluated (minibatch size N)
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        # Increment completed trials counter
        self.mipro_completed_trials += 1
        
        # Count rollouts from trial events
        num_seeds = data.get("num_seeds") or data.get("num_instances", 0)
        if num_seeds:
            self.mipro_rollouts_completed += num_seeds
        
        # Show trial score (minibatch) - like GEPA trial format
        if self.show_trial_results:
            timestamp = datetime.now().strftime("%H:%M:%S")
            minibatch_score = data.get("minibatch_score")
            iteration = data.get("iteration")
            trial = data.get("trial")
            
            if minibatch_score is not None:
                try:
                    score_float = float(minibatch_score)
                    # Calculate trial number for display
                    if iteration is not None and trial is not None and self.mipro_trials_per_iteration:
                        trial_num_display = (iteration * self.mipro_trials_per_iteration) + (trial + 1)
                    else:
                        trial_num_display = self.mipro_completed_trials
                    
                    n_str = f" N={num_seeds}" if num_seeds else ""
                    best_str = f" (Best: {self.mipro_best_score:.4f})" if self.mipro_best_score > 0 else ""
                    
                    self._write_log(
                        f"[{timestamp}] [Trial {trial_num_display}] Score: {score_float:.4f}{best_str}{n_str}"
                    )
                except (ValueError, TypeError):
                    pass
        
        # Emit progress update after each trial (throttled internally)
        self._emit_mipro_progress()
    
    def _handle_mipro_fulleval_complete(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO full evaluation completion event.
        
        Processes full evaluation events, which occur less frequently than minibatch
        trials. Full evaluations use the full validation set and are more expensive.
        Only displays results if the score is "promising":
        - Better than current best score, OR
        - At least 5% improvement over baseline
        
        Tracks rollouts from full evaluations and updates best score. Displays
        results with baseline comparison and improvement percentage.
        
        Args:
            event_data: Event data dictionary containing:
                - data.score: Full evaluation score
                - data.iteration: Current iteration number
                - data.trial: Trial number within iteration
                - data.num_seeds: Number of seeds evaluated (full eval size)
                - data.seeds: List of seed values (alternative to num_seeds)
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        # Count rollouts from full eval
        num_seeds = data.get("num_seeds") or data.get("seeds", 0)
        if isinstance(num_seeds, list):
            num_seeds = len(num_seeds)
        if num_seeds:
            self.mipro_rollouts_completed += num_seeds
        
        score = data.get("score")
        if score is None:
            return
        
        try:
            score_float = float(score)
        except (ValueError, TypeError):
            return
        
        # Initialize baseline if not set (use first score as baseline)
        if self.mipro_baseline_score is None:
            self.mipro_baseline_score = score_float
        
        # Only show if score is promising:
        # - Better than current best, OR
        # - At least 5% improvement over baseline
        is_promising = False
        if score_float > self.mipro_best_score:
            self.mipro_best_score = score_float
            is_promising = True
        elif self.mipro_baseline_score is not None:
            improvement = score_float - self.mipro_baseline_score
            improvement_pct = (improvement / self.mipro_baseline_score * 100) if self.mipro_baseline_score > 0 else 0
            if improvement_pct >= 5.0:  # At least 5% improvement over baseline
                is_promising = True
        
        if is_promising:
            timestamp = datetime.now().strftime("%H:%M:%S")
            iteration = data.get("iteration")
            trial = data.get("trial")
            seeds = data.get("seeds") or data.get("num_seeds", 0)
            if isinstance(seeds, list):
                seeds = len(seeds)
            
            # Format similar to GEPA trial results with N displayed
            iter_str = f" iter={iteration}" if iteration is not None else ""
            trial_str = f" trial={trial}" if trial is not None else ""
            n_str = f" N={seeds}" if seeds else ""
            
            baseline_str = ""
            if self.mipro_baseline_score is not None:
                improvement = score_float - self.mipro_baseline_score
                improvement_pct = (improvement / self.mipro_baseline_score * 100) if self.mipro_baseline_score > 0 else 0
                baseline_str = f" (Baseline: {self.mipro_baseline_score:.4f}, +{improvement_pct:.1f}%)"
            
            self._write_log(
                f"[{timestamp}] Full eval: Score={score_float:.4f} (Best: {self.mipro_best_score:.4f}){n_str}{baseline_str}{iter_str}{trial_str}"
            )
    
    def _handle_mipro_new_incumbent(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO new incumbent event (best candidate found).
        
        Processes events when MIPRO finds a new best candidate (incumbent).
        Updates the optimization curve and displays the result in GEPA-like format
        for consistency. Tracks cumulative trial count for curve visualization.
        
        Args:
            event_data: Event data dictionary containing:
                - data.minibatch_score: Minibatch score of the new incumbent
                - data.best_score: Overall best score
                - data.iteration: Current iteration number
                - data.trial: Trial number within iteration
                - data.cumulative_trials: Cumulative trial count across iterations
                - data.num_seeds: Minibatch size (N)
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        minibatch_score = data.get("minibatch_score")
        best_score = data.get("best_score")
        iteration = data.get("iteration")
        trial = data.get("trial")
        num_seeds = data.get("num_seeds")  # N for minibatch
        
        if minibatch_score is None:
            return
        
        try:
            score_float = float(minibatch_score)
        except (ValueError, TypeError):
            return
        
        # Update best score if this is better
        if best_score is not None:
            best_float = float(best_score)
            if best_float > self.best_score_so_far:
                self.best_score_so_far = best_float
        elif score_float > self.best_score_so_far:
            self.best_score_so_far = score_float
        
        # Track optimization curve
        if trial is not None:
            # Use cumulative trial count for x-axis
            cumulative_trials = data.get("cumulative_trials")
            if cumulative_trials is not None:
                trial_num = cumulative_trials
            else:
                # Estimate: (iteration * trials_per_iteration) + trial
                if iteration is not None and self.mipro_trials_per_iteration:
                    trial_num = (iteration * self.mipro_trials_per_iteration) + (trial + 1)
                else:
                    trial_num = self.trial_counter + 1
            
            self.optimization_curve.append((trial_num, self.best_score_so_far))
            self.trial_counter = trial_num
        
        # Format like GEPA: [Trial X] Score: X (Best: Y) N=Z
        trial_num_display = self.trial_counter if self.trial_counter > 0 else (trial + 1 if trial is not None else 1)
        n_str = f" N={num_seeds}" if num_seeds is not None else ""
        
        click.echo(
            f"[{timestamp}] [Trial {trial_num_display}] Score: {score_float:.4f} (Best: {self.best_score_so_far:.4f}){n_str}"
        )
        
        # Emit progress update after each trial (throttled internally)
        self._emit_mipro_progress()
    
    def _handle_mipro_budget_update(self, event_data: dict[str, Any]) -> None:
        """Handle MIPRO budget update events.
        
        Tracks token usage and cost accumulation during optimization. Updates:
        - Total tokens consumed (all operations)
        - Policy tokens (rollout tokens only)
        - Total cost in USD
        - Max token and cost limits (if provided in event)
        
        Emits throttled progress updates to show budget consumption.
        
        Args:
            event_data: Event data dictionary containing:
                - data.total_tokens: Total tokens consumed
                - data.policy_tokens: Tokens used for rollouts (policy only)
                - data.total_cost_usd: Total cost in USD
                - data.max_token_limit: Maximum token budget (optional)
                - data.max_spend_usd: Maximum cost budget (optional)
        """
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        # Update token tracking
        total_tokens = data.get("total_tokens")
        if total_tokens is not None:
            self.mipro_total_tokens = total_tokens
        
        # Track policy tokens separately (rollout tokens)
        policy_tokens = data.get("policy_tokens")
        if policy_tokens is not None:
            self.mipro_policy_tokens = policy_tokens
        
        # Update cost tracking
        total_cost = data.get("total_cost_usd")
        if total_cost is not None:
            self.mipro_total_cost = total_cost
        
        # Extract max limits if available in event data
        max_token_limit = data.get("max_token_limit")
        if max_token_limit is not None:
            self.mipro_max_tokens = max_token_limit
        
        max_spend_usd = data.get("max_spend_usd")
        if max_spend_usd is not None:
            self.mipro_max_cost = max_spend_usd
        
        # Emit progress update periodically (throttled)
        self._emit_mipro_progress()
    
    def _emit_mipro_progress(self) -> None:
        """Emit a comprehensive progress update for MIPRO (throttled).
        
        Formats and displays MIPRO progress in a format similar to GEPA for consistency.
        Shows:
        - Overall completion percentage
        - Trial progress (completed/total with remaining)
        - Iteration progress (current/total)
        - Rollout progress (completed/max)
        - Token usage (used/budget in millions)
        - Cost (USD)
        - Elapsed time and ETA
        
        Progress updates are throttled to emit at most every 5 seconds to avoid
        overwhelming the console. This method is called after significant events
        (trial completion, iteration completion, budget updates).
        
        Note:
            Only emits if start_time is set (job has started) and sufficient time
            has passed since the last update.
        """
        import time
        
        if self.mipro_start_time is None:
            return
        
        # Throttle progress updates - only emit every N seconds
        now = time.time()
        if self._last_progress_emit_time is not None:
            time_since_last = now - self._last_progress_emit_time
            if time_since_last < self._progress_emit_interval:
                return  # Skip this update
        
        self._last_progress_emit_time = now
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = now - self.mipro_start_time
        
        parts = []
        
        # Overall progress percentage
        percent_overall = None
        if self.mipro_total_trials and self.mipro_completed_trials is not None:
            percent_overall = (self.mipro_completed_trials / self.mipro_total_trials) * 100
            parts.append(f"{int(percent_overall)}% complete")
        
        # Trial progress (like rollouts in GEPA)
        if self.mipro_total_trials and self.mipro_completed_trials is not None:
            parts.append(f"trials={self.mipro_completed_trials}/{self.mipro_total_trials}")
            # Calculate remaining trials
            remaining_trials = self.mipro_total_trials - self.mipro_completed_trials
            if remaining_trials > 0:
                parts.append(f"rem={remaining_trials}")
            # Show percentage
            if percent_overall is not None:
                parts.append(f"({int(percent_overall)}%)")
        elif self.mipro_completed_trials is not None:
            parts.append(f"trials={self.mipro_completed_trials}")
        
        # Iteration progress
        if self.mipro_num_iterations and self.mipro_current_iteration is not None:
            parts.append(f"iter={self.mipro_current_iteration + 1}/{self.mipro_num_iterations}")
        
        # Rollouts completed vs max (like GEPA) - always show if we have any rollouts
        if self.mipro_rollouts_completed > 0:
            # Always try to show max if available (from TOML, event, or estimate)
            max_rollouts_to_show = self.mipro_max_rollouts
            if max_rollouts_to_show is None and self.mipro_total_trials and self.mipro_batch_size:
                # Estimate max rollouts from total trials if available
                    max_rollouts_to_show = self.mipro_total_trials * self.mipro_batch_size
            
            if max_rollouts_to_show:
                rollouts_pct = (self.mipro_rollouts_completed / max_rollouts_to_show) * 100
                parts.append(f"rollouts={self.mipro_rollouts_completed}/{max_rollouts_to_show} ({int(rollouts_pct)}%)")
            else:
                parts.append(f"rollouts={self.mipro_rollouts_completed}")
        
        # Tokens (policy tokens only, like GEPA rollout_tokens) - always show max if available
        if self.mipro_policy_tokens > 0:
            rollout_tokens_millions = self.mipro_policy_tokens / 1_000_000.0
            if self.mipro_max_tokens:
                # Use max_tokens as budget for rollout tokens (approximation)
                budget_millions = self.mipro_max_tokens / 1_000_000.0
                tokens_pct = (self.mipro_policy_tokens / self.mipro_max_tokens * 100) if self.mipro_max_tokens > 0 else 0
                parts.append(f"tokens={rollout_tokens_millions:.2f}M/{budget_millions:.2f}M ({int(tokens_pct)}%)")
            else:
                parts.append(f"tokens={rollout_tokens_millions:.2f}M")
        
        # Timing (elapsed out of max, like GEPA)
        elapsed_seconds = int(elapsed)
        if self.mipro_max_time_seconds:
            elapsed_pct = (elapsed / self.mipro_max_time_seconds * 100) if self.mipro_max_time_seconds > 0 else 0
            max_time_minutes = self.mipro_max_time_seconds / 60.0
            if elapsed_seconds >= 60:
                elapsed_str = f"{elapsed_seconds / 60:.1f}min/{max_time_minutes:.1f}min ({int(elapsed_pct)}%)"
            else:
                elapsed_str = f"{elapsed_seconds}s/{int(self.mipro_max_time_seconds)}s ({int(elapsed_pct)}%)"
        else:
            if elapsed_seconds >= 60:
                elapsed_str = f"{elapsed_seconds / 60:.1f}min"
            else:
                elapsed_str = f"{elapsed_seconds}s"
        parts.append(f"elapsed={elapsed_str}")
        
        # ETA calculation (similar to GEPA) - always show if we have progress
        eta_seconds = None
        if self.mipro_completed_trials is not None and self.mipro_completed_trials > 0 and elapsed > 0:
            rate = self.mipro_completed_trials / elapsed
            if rate > 0:
                if self.mipro_total_trials:
                    # Calculate ETA based on remaining trials
                    remaining = self.mipro_total_trials - self.mipro_completed_trials
                    if remaining > 0:
                        eta_seconds = remaining / rate
                else:
                    # Estimate based on iterations if we don't have total trials
                    if self.mipro_num_iterations and self.mipro_current_iteration is not None:
                        remaining_iterations = self.mipro_num_iterations - (self.mipro_current_iteration + 1)
                        if remaining_iterations > 0 and self.mipro_trials_per_iteration:
                            # Estimate: assume same rate for remaining iterations
                            remaining_trials_estimate = remaining_iterations * self.mipro_trials_per_iteration
                            eta_seconds = remaining_trials_estimate / rate
        
        if eta_seconds is not None and eta_seconds > 0:
            eta_str = f"{eta_seconds / 60:.1f}min" if eta_seconds >= 60 else f"{int(eta_seconds)}s"
            parts.append(f"eta={eta_str}")
        
        if parts:
            progress_msg = " ".join(parts)
            self._write_log(f"[{timestamp}] Progress: {progress_msg}")
    
    def flush(self) -> None:
        """Flush buffered output and close log file."""
        if self._log_file_handle:
            try:
                from datetime import datetime
                self._log_file_handle.write("\n" + "=" * 80 + "\n")
                self._log_file_handle.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._log_file_handle.write("=" * 80 + "\n")
                self._log_file_handle.flush()
                self._log_file_handle.close()
            except Exception:
                pass
            finally:
                self._log_file_handle = None
    
    def _handle_proposal_scored(self, event_data: dict[str, Any]) -> None:
        """Handle GEPA proposal scored events (transformations).
        
        Displays transformation/proposal scoring events from GEPA optimization.
        Only called if show_transformations is True (default: False) to avoid
        verbose output. Shows the score assigned to each proposed transformation.
        
        Args:
            event_data: Event data dictionary containing:
                - data.score: Score assigned to the transformation/proposal
        """
        # Only called if show_transformations=True
        data = event_data.get("data", {})
        if not isinstance(data, dict):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        score = data.get("score")
        if score is not None:
            click.echo(f"[{timestamp}] Proposal scored: {score:.4f}")


__all__ = [
    "GraphGenHandler",
    "BufferedHandler",
    "CallbackHandler",
    "CLIHandler",
    "PromptLearningHandler",
    "JSONHandler",
    "IntegrationTestHandler",
    "LossCurveHandler",
    "RichHandler",
    "StreamHandler",
]
