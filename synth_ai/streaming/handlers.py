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
                details: list[str] = []
                if rank is not None:
                    details.append(f"Rank {rank}")
                if isinstance(train_score, (int, float)):
                    details.append(f"train={train_score:.3f} ({train_score*100:.1f}%)")
                if isinstance(test_score, (int, float)):
                    details.append(f"test={test_score:.3f} ({test_score*100:.1f}%)")
                if isinstance(lift_abs, (int, float)) and isinstance(lift_pct, (int, float)):
                    details.append(f"lift={lift_abs:+.3f} ({lift_pct:+.1f}%)")
                if details:
                    click.echo("    --- TOP-K CANDIDATE ---")
                    click.echo(f"    {' | '.join(details)}")
                    if isinstance(instruction_text, str) and instruction_text.strip():
                        snippet = instruction_text.strip()
                        click.echo(f"        Instruction: {snippet}")
                    if isinstance(demo_indices, list) and demo_indices:
                        click.echo(f"        Demo indices: {demo_indices}")
                    seed_scores = data.get("test_seed_scores")
                    if isinstance(seed_scores, list) and seed_scores:
                        formatted_scores = ", ".join(
                            f"{item.get('seed')}: {item.get('score'):.2f}"
                            for item in seed_scores
                            if isinstance(item, dict) and isinstance(item.get("seed"), int) and isinstance(item.get("score"), (int, float))
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
            metric_str = f"[{timestamp}] [metric] {name}={value:.4f}" if isinstance(value, (int, float)) else f"[{timestamp}] [metric] {name}={value}"
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
                "LossCurveHandler requires the 'rich' package. Install synth-ai[analytics] or rich>=13."
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
            if not isinstance(value, (int, float)) or not isinstance(step, int):
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
                "RichHandler requires the 'rich' package. Install synth-ai[analytics] or rich>=13."
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
                self._progress.update(self._task_id, description=description)
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
                    self._progress.update(self._task_id, completed=int(step), total=int(total_steps))
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
            self._progress.update(self._task_id, total=total)

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


__all__ = [
    "BufferedHandler",
    "CallbackHandler",
    "CLIHandler",
    "JSONHandler",
    "IntegrationTestHandler",
    "LossCurveHandler",
    "RichHandler",
    "StreamHandler",
]
