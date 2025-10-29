# Training Job Streaming Abstractions Design

## Overview
Design for flexible, configurable streaming of training job information (status, events, metrics, timeline) with sensible defaults and fine-grained control options.

---

## 1. Core Abstractions

### 1.1 Stream Types (Data Categories)

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Optional

class StreamType(Enum):
    """Categories of information streams from training jobs."""
    STATUS = auto()      # Job status transitions (queued → running → succeeded)
    EVENTS = auto()      # Structured events (sft.progress, rl.train.step, etc.)
    METRICS = auto()     # Numeric time-series (train.loss, eval.reward_mean)
    TIMELINE = auto()    # Status phase transitions with metadata
    
    @property
    def endpoint_path(self) -> str:
        """Map stream type to API endpoint suffix."""
        return {
            StreamType.STATUS: "",  # /learning/jobs/{job_id}
            StreamType.EVENTS: "/events",
            StreamType.METRICS: "/metrics",
            StreamType.TIMELINE: "/timeline",
        }[self]
```

### 1.2 Stream Messages (Unified Event Format)

```python
@dataclass
class StreamMessage:
    """Unified message format for all stream types."""
    stream_type: StreamType
    timestamp: str
    job_id: str
    data: Dict[str, Any]
    
    # Type-specific fields (populated based on stream_type)
    seq: Optional[int] = None           # Events: sequence number
    step: Optional[int] = None          # Metrics: training step
    phase: Optional[str] = None         # Timeline: status phase
    
    @property
    def key(self) -> str:
        """Unique key for deduplication."""
        if self.stream_type == StreamType.EVENTS:
            return f"event:{self.seq}"
        elif self.stream_type == StreamType.METRICS:
            return f"metric:{self.data.get('name')}:{self.step}"
        elif self.stream_type == StreamType.TIMELINE:
            return f"timeline:{self.phase}:{self.timestamp}"
        else:
            return f"status:{self.timestamp}"
    
    @classmethod
    def from_status(cls, job_id: str, status_data: Dict[str, Any]) -> "StreamMessage":
        """Create message from status endpoint response."""
        return cls(
            stream_type=StreamType.STATUS,
            timestamp=status_data.get("updated_at", ""),
            job_id=job_id,
            data=status_data,
        )
    
    @classmethod
    def from_event(cls, job_id: str, event_data: Dict[str, Any]) -> "StreamMessage":
        """Create message from event."""
        return cls(
            stream_type=StreamType.EVENTS,
            timestamp=event_data.get("created_at", ""),
            job_id=job_id,
            data=event_data,
            seq=event_data.get("seq"),
        )
    
    @classmethod
    def from_metric(cls, job_id: str, metric_data: Dict[str, Any]) -> "StreamMessage":
        """Create message from metric point."""
        return cls(
            stream_type=StreamType.METRICS,
            timestamp=metric_data.get("created_at", ""),
            job_id=job_id,
            data=metric_data,
            step=metric_data.get("step"),
        )
    
    @classmethod
    def from_timeline(cls, job_id: str, timeline_data: Dict[str, Any]) -> "StreamMessage":
        """Create message from timeline entry."""
        return cls(
            stream_type=StreamType.TIMELINE,
            timestamp=timeline_data.get("created_at", ""),
            job_id=job_id,
            data=timeline_data,
            phase=timeline_data.get("phase"),
        )
```

---

## 2. Stream Configuration

### 2.1 StreamConfig (What to Stream)

```python
@dataclass
class StreamConfig:
    """Configuration for which streams to enable and how to filter them."""
    
    # Which stream types to enable
    enabled_streams: set[StreamType] = None  # None = all streams
    
    # Event filtering
    event_types: Optional[set[str]] = None    # e.g., {"sft.progress", "sft.loss"}
    event_levels: Optional[set[str]] = None   # e.g., {"info", "warning", "error"}
    
    # Metric filtering
    metric_names: Optional[set[str]] = None   # e.g., {"train.loss", "eval.reward_mean"}
    metric_phases: Optional[set[str]] = None  # e.g., {"train", "eval"}
    
    # Timeline filtering
    timeline_phases: Optional[set[str]] = None  # e.g., {"training", "evaluating"}
    
    # Sampling/throttling
    sample_rate: float = 1.0                  # 0.0-1.0, for high-volume streams
    max_events_per_poll: Optional[int] = None # Limit events per poll cycle
    
    # Deduplication
    deduplicate: bool = True
    
    def __post_init__(self):
        """Set defaults."""
        if self.enabled_streams is None:
            self.enabled_streams = set(StreamType)  # Enable all by default
    
    @classmethod
    def default(cls) -> "StreamConfig":
        """Default configuration - stream everything."""
        return cls()
    
    @classmethod
    def minimal(cls) -> "StreamConfig":
        """Minimal configuration - status only."""
        return cls(enabled_streams={StreamType.STATUS})
    
    @classmethod
    def progress_only(cls) -> "StreamConfig":
        """Configuration for training progress only."""
        return cls(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={"sft.progress", "rl.train.step", "sft.validation.summary"},
            metric_names={"train.loss", "eval.reward_mean"},
        )
    
    @classmethod
    def errors_only(cls) -> "StreamConfig":
        """Configuration for errors and failures only."""
        return cls(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS},
            event_levels={"error", "warning"},
        )
    
    def should_include_event(self, event: Dict[str, Any]) -> bool:
        """Check if event passes filters."""
        if self.event_types and event.get("type") not in self.event_types:
            return False
        if self.event_levels and event.get("level") not in self.event_levels:
            return False
        return True
    
    def should_include_metric(self, metric: Dict[str, Any]) -> bool:
        """Check if metric passes filters."""
        if self.metric_names and metric.get("name") not in self.metric_names:
            return False
        if self.metric_phases and metric.get("phase") not in self.metric_phases:
            return False
        return True
```

---

## 3. Stream Handlers (Output Formatting)

### 3.1 StreamHandler Base Class

```python
from abc import ABC, abstractmethod
from typing import Protocol

class StreamHandler(ABC):
    """Base class for handling/formatting stream messages."""
    
    @abstractmethod
    def handle(self, message: StreamMessage) -> None:
        """Process a stream message."""
        pass
    
    def should_handle(self, message: StreamMessage) -> bool:
        """Check if this handler should process the message."""
        return True  # Default: handle all messages
    
    def flush(self) -> None:
        """Flush any buffered output (optional)."""
        pass
```

### 3.2 Built-in Handlers

```python
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.live import Live

class CLIHandler(StreamHandler):
    """Simple CLI output handler (like current behavior)."""
    
    def handle(self, message: StreamMessage) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message.stream_type == StreamType.STATUS:
            status = message.data.get("status", "unknown")
            click.echo(f"[{timestamp}] Status: {status}")
        
        elif message.stream_type == StreamType.EVENTS:
            event_type = message.data.get("type", "")
            msg = message.data.get("message", "")
            click.echo(f"[{timestamp}] [{message.seq}] {event_type}: {msg}")
        
        elif message.stream_type == StreamType.METRICS:
            name = message.data.get("name")
            value = message.data.get("value")
            step = message.data.get("step")
            click.echo(f"[{timestamp}] {name}={value} (step={step})")


class RichHandler(StreamHandler):
    """Rich terminal UI with progress bars and tables."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self.metrics_table = Table(title="Training Metrics")
        self.current_status = "unknown"
        self.latest_metrics = {}
        self.event_log = []
    
    def handle(self, message: StreamMessage) -> None:
        if message.stream_type == StreamType.STATUS:
            self.current_status = message.data.get("status", "unknown")
        
        elif message.stream_type == StreamType.EVENTS:
            event_type = message.data.get("type", "")
            
            # Update progress bar based on step events
            if event_type in ("sft.progress", "rl.train.step"):
                step = message.data.get("data", {}).get("step")
                total = message.data.get("data", {}).get("total_steps")
                if step and total:
                    # Update or create progress task
                    pass
            
            # Log important events
            if event_type in ("sft.training.started", "rl.train.started"):
                self.console.log(f"✨ Training started")
            elif "error" in event_type or message.data.get("level") == "error":
                self.console.log(f"❌ {message.data.get('message')}", style="red")
        
        elif message.stream_type == StreamType.METRICS:
            name = message.data.get("name")
            value = message.data.get("value")
            self.latest_metrics[name] = value


class JSONHandler(StreamHandler):
    """Output messages as JSON lines (for machine parsing)."""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.file_handle = None
        if output_file:
            self.file_handle = open(output_file, "w")
    
    def handle(self, message: StreamMessage) -> None:
        import json
        output = {
            "stream_type": message.stream_type.name,
            "timestamp": message.timestamp,
            "job_id": message.job_id,
            "data": message.data,
        }
        if message.seq is not None:
            output["seq"] = message.seq
        if message.step is not None:
            output["step"] = message.step
        if message.phase is not None:
            output["phase"] = message.phase
        
        line = json.dumps(output)
        if self.file_handle:
            self.file_handle.write(line + "\n")
            self.file_handle.flush()
        else:
            print(line)
    
    def flush(self) -> None:
        if self.file_handle:
            self.file_handle.flush()
    
    def __del__(self):
        if self.file_handle:
            self.file_handle.close()


class CallbackHandler(StreamHandler):
    """Custom handler that calls user-provided callbacks."""
    
    def __init__(
        self,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_metric: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_timeline: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.on_status = on_status
        self.on_event = on_event
        self.on_metric = on_metric
        self.on_timeline = on_timeline
    
    def handle(self, message: StreamMessage) -> None:
        if message.stream_type == StreamType.STATUS and self.on_status:
            self.on_status(message.data)
        elif message.stream_type == StreamType.EVENTS and self.on_event:
            self.on_event(message.data)
        elif message.stream_type == StreamType.METRICS and self.on_metric:
            self.on_metric(message.data)
        elif message.stream_type == StreamType.TIMELINE and self.on_timeline:
            self.on_timeline(message.data)


class BufferedHandler(StreamHandler):
    """Handler that buffers messages for batch processing."""
    
    def __init__(self, flush_interval: float = 5.0, max_buffer_size: int = 100):
        self.buffer: List[StreamMessage] = []
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.last_flush = time.time()
    
    def handle(self, message: StreamMessage) -> None:
        self.buffer.append(message)
        
        # Auto-flush on size or time
        if (len(self.buffer) >= self.max_buffer_size or 
            time.time() - self.last_flush >= self.flush_interval):
            self.flush()
    
    def flush(self) -> None:
        if self.buffer:
            self.process_batch(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
    
    def process_batch(self, messages: List[StreamMessage]) -> None:
        """Override this to implement batch processing."""
        pass
```

---

## 4. Stream Multiplexer (Core Polling Logic)

### 4.1 JobStreamer Class

```python
class JobStreamer:
    """Unified streamer that multiplexes multiple data streams."""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        job_id: str,
        config: Optional[StreamConfig] = None,
        handlers: Optional[List[StreamHandler]] = None,
        *,
        interval_seconds: float = 2.0,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.job_id = job_id
        self.config = config or StreamConfig.default()
        self.handlers = handlers or [CLIHandler()]
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        
        # Tracking state per stream
        self.last_seq_by_stream: Dict[str, int] = {}
        self.last_step_by_metric: Dict[str, int] = {}
        self.seen_messages: Set[str] = set()
        self.terminal_statuses = {"succeeded", "failed", "cancelled", "canceled", "completed"}
    
    async def stream_until_terminal(self) -> Dict[str, Any]:
        """Stream all configured streams until job reaches terminal state."""
        start_time = time.time()
        
        async with AsyncHttpClient(self.base_url, self.api_key) as http:
            while True:
                # Poll each enabled stream
                messages = []
                
                if StreamType.STATUS in self.config.enabled_streams:
                    messages.extend(await self._poll_status(http))
                
                if StreamType.EVENTS in self.config.enabled_streams:
                    messages.extend(await self._poll_events(http))
                
                if StreamType.METRICS in self.config.enabled_streams:
                    messages.extend(await self._poll_metrics(http))
                
                if StreamType.TIMELINE in self.config.enabled_streams:
                    messages.extend(await self._poll_timeline(http))
                
                # Process messages through handlers
                for message in messages:
                    # Deduplication
                    if self.config.deduplicate and message.key in self.seen_messages:
                        continue
                    self.seen_messages.add(message.key)
                    
                    # Apply sampling
                    if random.random() > self.config.sample_rate:
                        continue
                    
                    # Send to all handlers
                    for handler in self.handlers:
                        if handler.should_handle(message):
                            handler.handle(message)
                
                # Check for terminal status
                if messages:
                    status_msg = next(
                        (m for m in messages if m.stream_type == StreamType.STATUS),
                        None
                    )
                    if status_msg:
                        status = status_msg.data.get("status", "").lower()
                        if status in self.terminal_statuses:
                            # Flush all handlers
                            for handler in self.handlers:
                                handler.flush()
                            return status_msg.data
                
                # Timeout check
                if self.timeout_seconds and (time.time() - start_time) >= self.timeout_seconds:
                    raise TimeoutError(f"Streaming timed out after {self.timeout_seconds}s")
                
                await asyncio.sleep(self.interval_seconds)
    
    async def _poll_status(self, http: AsyncHttpClient) -> List[StreamMessage]:
        """Poll status endpoint."""
        try:
            data = await http.get(f"/api/learning/jobs/{self.job_id}")
            return [StreamMessage.from_status(self.job_id, data)]
        except Exception:
            return []
    
    async def _poll_events(self, http: AsyncHttpClient) -> List[StreamMessage]:
        """Poll events endpoint."""
        try:
            since_seq = self.last_seq_by_stream.get("events", 0)
            data = await http.get(
                f"/api/learning/jobs/{self.job_id}/events",
                params={"since_seq": since_seq, "limit": 200}
            )
            events = data.get("events", [])
            
            messages = []
            for event in events:
                # Update sequence tracking
                seq = event.get("seq", 0)
                if seq > self.last_seq_by_stream.get("events", 0):
                    self.last_seq_by_stream["events"] = seq
                
                # Apply filters
                if not self.config.should_include_event(event):
                    continue
                
                messages.append(StreamMessage.from_event(self.job_id, event))
            
            return messages
        except Exception:
            return []
    
    async def _poll_metrics(self, http: AsyncHttpClient) -> List[StreamMessage]:
        """Poll metrics endpoint."""
        try:
            # Get max step across all metrics
            after_step = max(self.last_step_by_metric.values()) if self.last_step_by_metric else -1
            
            data = await http.get(
                f"/api/learning/jobs/{self.job_id}/metrics",
                params={"after_step": after_step, "limit": 200}
            )
            points = data.get("points", [])
            
            messages = []
            for point in points:
                # Update step tracking
                name = point.get("name", "")
                step = point.get("step", 0)
                if step > self.last_step_by_metric.get(name, -1):
                    self.last_step_by_metric[name] = step
                
                # Apply filters
                if not self.config.should_include_metric(point):
                    continue
                
                messages.append(StreamMessage.from_metric(self.job_id, point))
            
            return messages
        except Exception:
            return []
    
    async def _poll_timeline(self, http: AsyncHttpClient) -> List[StreamMessage]:
        """Poll timeline endpoint."""
        try:
            data = await http.get(f"/api/learning/jobs/{self.job_id}/timeline")
            timeline = data.get("events", [])
            
            messages = []
            for entry in timeline:
                # Filter by phase if specified
                phase = entry.get("phase", "")
                if self.config.timeline_phases and phase not in self.config.timeline_phases:
                    continue
                
                messages.append(StreamMessage.from_timeline(self.job_id, entry))
            
            return messages
        except Exception:
            return []
```

---

## 5. CLI Integration

### 5.1 Enhanced Train Command

```python
@click.command("train")
@click.option("--config", "config_path", required=True)
@click.option("--type", "train_type", type=click.Choice(["rl", "sft"]))
@click.option("--poll/--no-poll", default=True)
# New streaming options
@click.option(
    "--stream",
    "stream_types",
    multiple=True,
    type=click.Choice(["status", "events", "metrics", "timeline", "all"]),
    default=["all"],
    help="Which streams to display (default: all)"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["cli", "rich", "json", "quiet"]),
    default="rich",
    help="Output format (default: rich)"
)
@click.option(
    "--filter-events",
    multiple=True,
    help="Event types to include (e.g., sft.progress)"
)
@click.option(
    "--filter-metrics",
    multiple=True,
    help="Metric names to include (e.g., train.loss)"
)
@click.option(
    "--json-output",
    type=click.Path(),
    help="Write JSON output to file"
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress all output except final status"
)
def train_command(
    config_path: str,
    train_type: str,
    poll: bool,
    stream_types: tuple[str, ...],
    output_format: str,
    filter_events: tuple[str, ...],
    filter_metrics: tuple[str, ...],
    json_output: Optional[str],
    quiet: bool,
    **kwargs
):
    """Train with configurable streaming output."""
    
    # ... existing job creation logic ...
    
    if not poll:
        click.echo(f"Job created: {job_id}")
        return
    
    # Build stream config
    if "all" in stream_types:
        enabled_streams = set(StreamType)
    else:
        enabled_streams = {
            StreamType[s.upper()] for s in stream_types
        }
    
    config = StreamConfig(
        enabled_streams=enabled_streams,
        event_types=set(filter_events) if filter_events else None,
        metric_names=set(filter_metrics) if filter_metrics else None,
    )
    
    # Build handlers
    handlers = []
    
    if quiet:
        # Only terminal status
        config = StreamConfig.minimal()
        handlers.append(CLIHandler())
    elif output_format == "rich":
        handlers.append(RichHandler())
    elif output_format == "cli":
        handlers.append(CLIHandler())
    elif output_format == "json":
        handlers.append(JSONHandler())
    
    # Add JSON file output if requested
    if json_output:
        handlers.append(JSONHandler(output_file=json_output))
    
    # Stream until completion
    streamer = JobStreamer(
        base_url=backend_base,
        api_key=api_key,
        job_id=job_id,
        config=config,
        handlers=handlers,
    )
    
    final_status = asyncio.run(streamer.stream_until_terminal())
    click.echo(f"Training completed: {final_status.get('status')}")
```

### 5.2 Example Usage

```bash
# Default: Rich UI with all streams
synth-ai train --config my_config.toml

# CLI format with only status and events
synth-ai train --config my_config.toml --format cli --stream status --stream events

# Filter to only progress and loss
synth-ai train --config my_config.toml \
  --filter-events sft.progress \
  --filter-metrics train.loss

# JSON output for machine parsing
synth-ai train --config my_config.toml --format json --json-output training.jsonl

# Quiet mode (minimal output)
synth-ai train --config my_config.toml --quiet

# Custom: Events + metrics only, JSON format
synth-ai train --config my_config.toml \
  --stream events --stream metrics \
  --format json \
  --filter-events "sft.progress,sft.validation.summary" \
  --filter-metrics "train.loss,val.loss"
```

---

## 6. SDK API (Programmatic Usage)

### 6.1 High-Level API

```python
from synth_ai.streaming import JobStreamer, StreamConfig, RichHandler

# Simple: Stream with defaults
async with JobStreamer.from_job_id(base_url, api_key, job_id) as streamer:
    final_status = await streamer.stream_until_terminal()

# Custom config
config = StreamConfig(
    enabled_streams={StreamType.EVENTS, StreamType.METRICS},
    event_types={"sft.progress", "sft.loss"},
)

streamer = JobStreamer(
    base_url=base_url,
    api_key=api_key,
    job_id=job_id,
    config=config,
    handlers=[RichHandler()],
)
final_status = await streamer.stream_until_terminal()

# Callback-based
def on_loss_update(event_data: Dict[str, Any]):
    loss = event_data.get("data", {}).get("loss")
    print(f"Loss: {loss}")

config = StreamConfig.progress_only()
handler = CallbackHandler(
    on_event=lambda e: on_loss_update(e) if e.get("type") == "sft.loss" else None
)

streamer = JobStreamer(
    base_url=base_url,
    api_key=api_key,
    job_id=job_id,
    config=config,
    handlers=[handler],
)
await streamer.stream_until_terminal()
```

---

## 7. Implementation Plan

### Phase 1: Core Abstractions (Week 1)
1. Implement `StreamType`, `StreamMessage` in `synth_ai/streaming/types.py`
2. Implement `StreamConfig` in `synth_ai/streaming/config.py`
3. Add basic `StreamHandler` and `CLIHandler` in `synth_ai/streaming/handlers.py`

### Phase 2: JobStreamer (Week 2)
1. Implement `JobStreamer` in `synth_ai/streaming/streamer.py`
2. Migrate polling logic from `JobHandle` to use new streamer
3. Add tests for deduplication, filtering, sampling

### Phase 3: Rich Handlers (Week 3)
1. Implement `RichHandler` with progress bars
2. Implement `JSONHandler` for machine parsing
3. Implement `CallbackHandler` for custom handling
4. Add `BufferedHandler` for batch processing

### Phase 4: CLI Integration (Week 4)
1. Update `synth_ai/api/train/cli.py` to use `JobStreamer`
2. Add CLI flags for stream control
3. Update help text and documentation
4. Add examples to README

### Phase 5: Testing & Polish (Week 5)
1. Integration tests with real jobs
2. Performance testing (high-volume streams)
3. Documentation and examples
4. Migration guide from old API

---

## 8. Benefits of This Design

### ✅ Flexibility
- Easy to add new stream types (e.g., `StreamType.LOGS`)
- Easy to add new handlers (e.g., `WebhookHandler`, `DatabaseHandler`)
- Easy to create custom filters and sampling strategies

### ✅ Sensible Defaults
- Default config streams everything (progressive disclosure)
- Default handler shows rich terminal UI
- Minimal configuration for common use cases

### ✅ Fine-Grained Control
- Filter by event type, metric name, log level
- Sample high-volume streams
- Buffer and batch process messages
- Multiple output formats simultaneously

### ✅ Extensibility
- Protocol-based handlers (easy to implement custom ones)
- Composable configs (combine filters, sampling, etc.)
- Pluggable architecture (swap out components)

### ✅ Backward Compatibility
- Can wrap existing `JobHandle.poll_until_terminal()` as a handler
- CLI can default to simple output if rich not available
- Progressive migration path from old API

### ✅ Performance
- Deduplication prevents reprocessing
- Sampling reduces overhead for high-volume streams
- Batching reduces handler overhead
- Async architecture supports concurrent streams

---

## 9. Example: Complete Training Session

```python
from synth_ai.streaming import (
    JobStreamer,
    StreamConfig,
    StreamType,
    RichHandler,
    JSONHandler,
    CallbackHandler,
)

# Custom callback for loss tracking
loss_history = []

def on_loss(event_data: Dict[str, Any]):
    if event_data.get("type") == "sft.loss":
        loss = event_data.get("data", {}).get("loss")
        step = event_data.get("data", {}).get("step")
        loss_history.append((step, loss))

# Create streamer with multiple handlers
config = StreamConfig(
    enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
    event_types={"sft.progress", "sft.loss", "sft.validation.summary"},
    metric_names={"train.loss", "val.loss"},
)

handlers = [
    RichHandler(),                              # Terminal UI
    JSONHandler(output_file="training.jsonl"), # Machine-readable log
    CallbackHandler(on_event=on_loss),          # Custom processing
]

streamer = JobStreamer(
    base_url="https://api.synth.ai",
    api_key=os.getenv("SYNTH_API_KEY"),
    job_id="job_abc123",
    config=config,
    handlers=handlers,
    interval_seconds=2.0,
)

# Stream until completion
final_status = await streamer.stream_until_terminal()

# Analyze results
print(f"Final status: {final_status.get('status')}")
print(f"Loss progression: {loss_history}")
```

Output would show:
- Rich terminal UI with progress bar and live metrics
- JSON log file for post-training analysis
- In-memory loss history for plotting

---

## 10. Migration Path

### For CLI Users
**Old (current):**
```bash
synth-ai train --config my_config.toml --poll
# Output: [poll] 19:12:21 0s status=running
```

**New (backward compatible):**
```bash
synth-ai train --config my_config.toml --poll
# Output: Rich UI with progress, metrics, events
```

**New (explicit simple mode):**
```bash
synth-ai train --config my_config.toml --poll --format cli
# Output: Same as old behavior
```

### For SDK Users
**Old (current):**
```python
from synth_ai.learning import JobHandle

job = JobHandle(base_url, api_key, job_id)
final_status = await job.poll_until_terminal(
    on_event=my_callback,
    on_metric=my_callback,
)
```

**New (compatible wrapper):**
```python
# JobHandle now uses JobStreamer under the hood
job = JobHandle(base_url, api_key, job_id)
final_status = await job.poll_until_terminal(
    on_event=my_callback,
    on_metric=my_callback,
)
```

**New (explicit):**
```python
from synth_ai.streaming import JobStreamer, CallbackHandler

handler = CallbackHandler(on_event=my_callback, on_metric=my_callback)
streamer = JobStreamer(base_url, api_key, job_id, handlers=[handler])
final_status = await streamer.stream_until_terminal()
```

---

## Summary

This design provides:
1. **Unified abstraction** for all stream types (status, events, metrics, timeline)
2. **Configurable filtering** at multiple levels (stream type, event type, metric name)
3. **Pluggable handlers** for different output formats (CLI, Rich, JSON, custom)
4. **Sensible defaults** (stream everything with rich UI)
5. **Fine-grained control** when needed (filters, sampling, buffering)
6. **Extensibility** for future needs (new stream types, new handlers)
7. **Backward compatibility** with existing API
8. **Clear migration path** for users

The architecture is production-ready and scales from simple CLI usage to complex programmatic control.

