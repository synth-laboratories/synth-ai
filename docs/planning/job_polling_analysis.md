# Job Status Polling Logic Analysis - SFT & RL Training

## Overview
This document analyzes the job status polling logic implemented in the synth-ai SDK and the corresponding status generation logic in the monorepo backend for both SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) training jobs.

---

## 1. CLIENT-SIDE POLLING LOGIC (synth-ai)

### 1.1 Core Polling Components

#### A. JobHandle (`synth_ai/learning/jobs.py`)
**Location**: Lines 44-202

**Purpose**: Generic job poller that works with any learning job type (SFT, RL, etc.)

**Key Features**:
- Polls multiple endpoints to find job status (flexible endpoint resolution)
- Uses `JobsApiResolver` to try different URL patterns:
  - `/api/learning/jobs/{job_id}` (primary)
  - `/api/rl/jobs/{job_id}` (fallback)
  - `/api/orchestration/jobs/{job_id}` (legacy fallback)
- Polls three data streams simultaneously:
  1. **Status** - job state (queued, running, succeeded, failed, etc.)
  2. **Events** - timestamped structured events with sequence numbers
  3. **Metrics** - numeric measurements (loss, reward, step, epoch, etc.)

**Polling Algorithm**:
```python
while not terminal:
    # 1. Fetch status from /learning/jobs/{job_id}
    status_data = await http.get(status_url)
    status = status_data.get("status")
    
    # 2. Discover linked_job_id if present (for RL workflows)
    events_job_id = status_data.get("linked_job_id")
    
    # 3. Fetch events from /learning/jobs/{job_id}/events?since_seq={last_seq}
    events = await http.get(events_url)
    for event in events:
        seq = event["seq"]
        if seq > last_seq:
            on_event(event)  # callback
            last_seq = seq
            
    # 4. Fetch metrics from /learning/jobs/{job_id}/metrics?after_step={last_step}
    metrics = await http.get(metrics_url)
    for point in metrics:
        if point["step"] > last_step_for_metric[point["name"]]:
            on_metric(point)  # callback
            
    # 5. Terminal check
    if status in TERMINAL_STATUSES or terminal_event_seen:
        return final_result
        
    # 6. Timeout & empty poll guards
    if empty_polls >= threshold:
        raise AssertionError("No events")
    if elapsed > startup_deadline and not saw_any_event:
        raise AssertionError("Startup timeout")
        
    await sleep(interval_seconds)  # Default: 2s
```

**Important Constants**:
- `TERMINAL_STATUSES`: {"succeeded", "failed", "cancelled", "canceled", "completed"}
- `TERMINAL_EVENT_SUCCESS`: {"sft.job.completed", "rl.job.completed", "workflow.completed"}
- `TERMINAL_EVENT_FAILURE`: {"sft.job.failed", "rl.job.failed", "workflow.failed"}
- Default interval: 2 seconds
- Default startup deadline: 45 seconds
- Empty polls threshold: 5

**Sequence Tracking**: Uses `last_seq_by_stream` dictionary to track last-seen sequence number per job_id stream, preventing duplicate event processing.

#### B. RlClient (`synth_ai/learning/rl/client.py`)
**Location**: Lines 21-268

**Purpose**: Specialized RL job client with similar polling logic

**Key Differences from JobHandle**:
- Simplified to only use `/api/learning/jobs/{job_id}` endpoints
- More verbose debug logging (prints to console)
- Same three-stream polling (status, events, metrics)
- Handles `linked_job_id` discovery for multi-service RL workflows
- Terminal events: `{"rl.job.completed", "workflow.completed", "rl.train.completed"}` for success

#### C. Simple CLI Pollers (`synth_ai/api/train/pollers.py`)
**Location**: Lines 19-76

**Purpose**: Synchronous blocking pollers for CLI commands

**Classes**:
1. **JobPoller** (base) - Generic GET polling with retry
2. **RLJobPoller** - Polls `/rl/jobs/{job_id}`
3. **SFTJobPoller** - Polls `/learning/jobs/{job_id}`

**Characteristics**:
- Simpler than async pollers
- No event streaming, just status field
- Used by `synth train` CLI command
- Default interval: 5 seconds
- Default timeout: 3600 seconds (1 hour)

### 1.2 Polling Invocation Points

#### CLI Train Command (`synth_ai/api/train/cli.py`)
**SFT Flow** (Lines 506-659):
1. Build payload from TOML config
2. Upload training file → get `file_id`
3. POST to `/learning/jobs` → get `job_id`
4. If `--poll`: Use `SFTJobPoller.poll_job(job_id)`

**RL Flow** (Lines 390-504):
1. Verify task app is accessible
2. Build payload from config
3. POST to `/rl/jobs` → get `job_id`
4. If `--poll`: Use `RLJobPoller.poll_job(job_id)`

---

## 2. BACKEND STATUS GENERATION LOGIC (monorepo)

### 2.1 Database Schema & Storage

#### PostgrestLearningEmitter (`backend/app/orchestration/jobs/postgrest_emitter.py`)
**Location**: Lines 17-410

**Purpose**: Write events, status, and metrics to PostgreSQL via PostgREST RPC

**Key Methods**:

**A. `append_event(job_id, type_, message, data, level)`** (Lines 229-320)
- Writes to `learning_shared_job_events` table
- Uses PostgREST RPC `learning_append_event` for atomic sequence assignment
- Retry logic for unique constraint violations (409 errors)
- Returns: `seq` (sequence number)
- Schema: `{job_id, seq, type, message, data (JSONB), level, created_at}`

**B. `append_status(job_id, phase, message, metadata)`** (Lines 103-136)
- Writes to `learning_shared_job_status` table
- Records state transitions: "queued" → "running" → "training" → "succeeded"
- Schema: `{job_id, phase, message, metadata (JSONB), created_at}`

**C. `append_metric(job_id, name, value, phase, step, epoch)`** (Lines 138-178)
- Writes to `learning_shared_job_metrics` table
- Records numeric time-series data
- Schema: `{job_id, name, value, phase, step, epoch, data (JSONB), created_at}`

**D. `append_metrics_bulk(job_id, records)`** (Lines 179-227)
- Batch insert for high-volume metric writes
- Used by training loops to minimize DB round-trips

#### LearningJobsRepositoryDB (`backend/app/orchestration/jobs/learning_repository.py`)
**Location**: Lines 47-1516

**Purpose**: SQLAlchemy-based repository for job CRUD operations

**Key Query Methods**:
- `get_job_by_job_id(job_id)` - Fetch job record (uses PostgREST then SQL fallback)
- `get_events(job_id, since_seq, limit)` - Fetch events with seq > since_seq
- `get_status_timeline(job_id, limit)` - Fetch recent status records
- `list_metrics(job_id, name, after_step, limit)` - Fetch metric points

### 2.2 Backend API Endpoints

#### Shared Routes (`backend/app/routes/simple_training/backend_routes_shared.py`)

**A. GET `/learning/jobs/{job_id}`** (Lines 2100-2150)
- Returns full job record with status field
- Includes: `{job_id, status, model_id, org_id, created_at, metadata, ...}`

**B. GET `/learning/jobs/{job_id}/status`** (Lines 2274-2302)
- Returns coarse status with progress
- Derives status from most recent timeline event
- Returns: `{job_id, status, progress (metadata), updated_at}`

**C. GET `/learning/jobs/{job_id}/events`** (Lines 2305-2328)
- Returns paginated events with `since_seq` cursor
- Returns: `{events: [{seq, type, message, data, created_at}, ...]}`

**D. GET `/learning/jobs/{job_id}/metrics`** (Lines 2219-2249)
- Returns metric points with `after_step` filter
- Returns: `{job_id, points: [{name, value, step, epoch, phase}, ...]}`

**E. GET `/learning/jobs/{job_id}/timeline`** (Lines 2252-2271)
- Returns status transition history
- Returns: `{job_id, events: [{phase, metadata, created_at}, ...]}`

### 2.3 SFT Training Status Emission

#### SFTTrainer (`backend/app/routes/simple_training/training/sft/trainer.py`)
**Location**: Lines 1049-1500+

**Event Emission During Training**:

**Initialization** (Lines 1270-1294):
```python
async def _emit_status(phase: str, metadata: Dict[str, Any]):
    await get_postgrest_emitter().append_status(
        job_id=job_id, phase=phase, metadata=metadata
    )

async def _emit_event(type_: str, message: str, data: Dict[str, Any], level: str):
    await get_postgrest_emitter().append_event(
        job_id=job_id, type_=type_, message=message, 
        data=data, level=level
    )
```

**Progress Tracking via Callbacks** (Lines 861-1022):
- `SFTProgressCallback` (HuggingFace Trainer callback)
- Emits on logging steps:
  - `sft.progress` events with {step, epoch, loss}
  - `sft.validation.summary` events after eval
  - Updates status to "evaluating" during eval

**Metric Batching** (Lines 1248-1268):
- Uses `_MetricBatcher` for efficient bulk inserts
- Batches up to 16 metrics, flushes every 1 second
- Metrics: `train.loss`, `val.loss`, `eval.reward_mean`, etc.

**Key Event Types**:
- `sft.training.started` - Training loop begins
- `sft.progress` - Periodic training updates
- `sft.loss` - Loss value updates
- `sft.validation.summary` - Evaluation completed
- `sft.job.completed` - Training succeeded
- `sft.job.failed` - Training failed

**Status Phases**:
- `queued` → `running` → `initializing_trainer` → `training` → `evaluating` → `succeeded` | `failed`

#### Modal GPU Training (`backend/app/routes/simple_training/modal_service/gpu_functions.py`)
**Location**: Lines 2771-7300+

**For Modal.com-based distributed training**:

**Callback in Trainer** (Lines 4804-4838):
```python
class _TrainerProgressCallback:
    def on_log(self, args, state, control, logs=None, **kwargs):
        payload = {
            "loss": logs.get("loss"),
            "epoch": logs.get("epoch"),
            "step": state.global_step,
            "total_steps": state.max_steps
        }
        _emit_db_event_sync(job_id, type_="sft.progress", 
                           message="Training progress", data=payload)
        
        # Also write to metrics table
        _append_metric_sync(
            job_id=job_id,
            name="train.loss",
            value=payload["loss"],
            phase="train",
            step=payload["step"],
            epoch=payload["epoch"]
        )
```

**Synchronous Emitters** (Lines 2942-2966):
```python
def emit_event(event_type: str, message: str, payload: dict):
    _emit_db_event_sync(job_id, type_=event_type, 
                       message=message, data=payload)

def update_status(status: str, extra: dict):
    _update_job_status(job_id, status, extra)

def append_metric(name: str, value: float, phase: str, 
                 step: int, epoch: int):
    _append_metric_sync(job_id, name, value, phase, step, epoch)
```

These functions use `requests` library to make synchronous HTTP calls to PostgREST from within Modal containers.

### 2.4 RL Training Status Emission

#### ClusteredGRPOLudicTrainer (`backend/app/routes/clustered_training/core/algorithms/gspo/training/clustered_trainer.py`)
**Location**: Lines 1070-8500+

**Event Emission Infrastructure** (Lines 2040-2110):

**A. `_emit_event(type_, message, data, level)`** (Lines 2040-2065):
```python
async def _emit_event(self, type_: str, message: str = "", 
                     data: Dict[str, Any] | None = None, 
                     level: str = "info"):
    try:
        emitter = get_postgrest_emitter()
        await emitter.append_event(
            job_id=str(self.job_id),
            type_=type_,
            message=message,
            data=data or {},
            level=level
        )
    except Exception:
        logger.warning("Event emission failed")
```

**B. `_emit_status(phase, message, metadata)`** (Lines 2095-2110):
```python
async def _emit_status(self, phase: str, 
                      message: str | None = None,
                      metadata: Dict[str, Any] | None = None):
    try:
        emitter = get_postgrest_emitter()
        await emitter.append_status(
            job_id=str(self.job_id),
            phase=phase,
            message=message,
            metadata=metadata or {}
        )
    except Exception:
        logger.warning("Status emission failed")
```

**C. `_emit_metrics(records)`** (Lines 2067-2093):
```python
async def _emit_metrics(self, records: list[Dict[str, Any]]):
    # Uses metrics batcher if available, else bulk emit
    if self._metrics_batcher is not None:
        for rec in records:
            await self._metrics_batcher.enqueue(rec)
    else:
        emitter = get_postgrest_emitter()
        await emitter.append_metrics_bulk(
            job_id=str(self.job_id), 
            records=records
        )
```

**Training Loop Emissions** (Lines 8402-8444):
```python
async def run_training(self):
    # Mark start
    await self._emit_event(
        type_="rl.train.started",
        message="RL training loop started",
        data={
            "model": self.config.model_name,
            "epochs": self.config.num_epochs,
            "iters_per_epoch": self.config.iterations_per_epoch,
            "batch_size": self.config.batch_size,
            "group_size": self.config.group_size
        }
    )
    
    await self._emit_status("training", metadata={
        "epoch": 0, "step": 0
    })
    
    # Training loop emits:
    # - rl.train.step events (per iteration)
    # - rl.eval.started events
    # - rl.eval.summary events (with rewards)
    # - Metrics: eval.reward_mean, train.policy_loss, etc.
```

**Key Event Types**:
- `rl.job.created` - Job record created
- `rl.train.started` - Training loop begins
- `rl.train.step` - Iteration completed
- `rl.eval.started` - Evaluation begins
- `rl.eval.summary` - Evaluation results
- `rl.metrics` - Metric batch
- `rl.job.completed` - Training succeeded
- `rl.job.failed` - Training failed
- `rl.pipeline.backpressure` - Queue congestion (async pipeline)
- `rl.pipeline.resume` - Queue recovered

**Status Phases**:
- `queued` → `initializing` → `training` → `evaluating` → `succeeded` | `failed`

**Metric Emission**:
- Metrics are batched via `_MetricBatcher` (similar to SFT)
- Per-step metrics: `eval.reward_mean`, `eval.reward_std`, `train.policy_loss`, `train.value_loss`
- Async pipeline also emits queue depth metrics

#### Pipeline RL Coordinator (`backend/app/routes/clustered_training/core/algorithms/gspo/training/pipeline/coordinator.py`)
**Location**: Lines 49-550+

**For async pipeline workflows**:

**Backpressure Events** (Lines 354-380):
```python
async def _emit_backpressure(self, reason: str, data: Dict[str, Any]):
    payload = dict(data)
    payload["reason"] = reason
    await self._emit_event(
        "rl.pipeline.backpressure",
        f"Pipeline backpressure ({reason})",
        data=payload,
        level="warning"
    )

async def _emit_resume(self, reason: str, data: Dict[str, Any]):
    await self._emit_event(
        "rl.pipeline.resume",
        f"Pipeline resume ({reason})",
        data=payload,
        level="info"
    )
```

**Queue Monitoring** (Lines 288-352):
- Periodically emits queue depth metrics
- Tracks rollout, judged, and microbatch queue sizes
- Triggers backpressure events when queues are full

---

## 3. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (synth-ai)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Submit Job (POST)                                           │
│     /learning/jobs OR /rl/jobs                                  │
│     → Returns: {job_id}                                         │
│                                                                 │
│  2. Poll Status (GET, every 2s)                                 │
│     /learning/jobs/{job_id}                                     │
│     → Returns: {status, linked_job_id, ...}                     │
│                                                                 │
│  3. Poll Events (GET, every 2s)                                 │
│     /learning/jobs/{job_id}/events?since_seq={last_seq}         │
│     → Returns: {events: [{seq, type, message, data}, ...]}      │
│                                                                 │
│  4. Poll Metrics (GET, every 2s)                                │
│     /learning/jobs/{job_id}/metrics?after_step={last_step}      │
│     → Returns: {points: [{name, value, step, epoch}, ...]}      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │ HTTP Polling
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Routes (backend_routes_shared.py):                             │
│  - GET /learning/jobs/{job_id}         → Query job table        │
│  - GET /learning/jobs/{job_id}/status  → Query status timeline  │
│  - GET /learning/jobs/{job_id}/events  → Query events (seq>N)   │
│  - GET /learning/jobs/{job_id}/metrics → Query metrics          │
│                                                                 │
│  Repository (LearningJobsRepositoryDB):                         │
│  - Uses PostgREST for reads (preferred)                         │
│  - Falls back to SQLAlchemy queries                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │ SQL/PostgREST
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      POSTGRESQL DATABASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tables:                                                        │
│  1. learning_shared_jobs                                        │
│     - id (PK), job_id (unique), org_id, status, model_id, ...  │
│                                                                 │
│  2. learning_shared_job_events                                  │
│     - id (PK), job_id (FK), seq (unique per job), type,        │
│       message, data (JSONB), level, created_at                  │
│                                                                 │
│  3. learning_shared_job_status                                  │
│     - id (PK), job_id (FK), phase, message,                    │
│       metadata (JSONB), created_at                              │
│                                                                 │
│  4. learning_shared_job_metrics                                 │
│     - id (PK), job_id (FK), name, value, phase,                │
│       step, epoch, data (JSONB), created_at                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           ▲
                           │ Writes (PostgREST RPC)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING WORKERS (Modal.com)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SFT Training:                                                  │
│  - SFTTrainer._emit_event(type, message, data)                 │
│  - SFTTrainer._emit_status(phase, metadata)                    │
│  - SFTProgressCallback.on_log() → emit every log step          │
│  - Uses _MetricBatcher for bulk metric writes                   │
│                                                                 │
│  RL Training:                                                   │
│  - ClusteredGRPOLudicTrainer._emit_event(type, message, data)  │
│  - ClusteredGRPOLudicTrainer._emit_status(phase, metadata)     │
│  - Per-iteration event emission (rl.train.step)                 │
│  - Per-epoch evaluation (rl.eval.started, rl.eval.summary)     │
│  - Async pipeline emits backpressure/resume events              │
│                                                                 │
│  Emitter (PostgrestLearningEmitter):                            │
│  - append_event() → POST /rest/v1/rpc/learning_append_event    │
│  - append_status() → POST /rest/v1/learning_shared_job_status  │
│  - append_metric() → POST /rest/v1/learning_shared_job_metrics │
│  - append_metrics_bulk() → Batch insert via PostgREST          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. KEY OBSERVATIONS & DESIGN PATTERNS

### 4.1 Event Sequencing
- Events have monotonically increasing `seq` numbers per job
- Client tracks `last_seq_by_stream` to avoid reprocessing
- PostgREST RPC `learning_append_event` ensures atomic sequence allocation
- Retry logic handles unique constraint violations (409 errors)

### 4.2 Linked Job Pattern
- RL workflows may have a `linked_job_id` for multi-service coordination
- Client discovers linked job from status response
- Polls events from both `job_id` and `linked_job_id` streams
- Prevents missing events when orchestration spans multiple services

### 4.3 Metric Batching
- Training loops emit high-volume metrics (per-step loss, rewards)
- `_MetricBatcher` accumulates up to 16 metrics, flushes every 1 second
- Reduces DB write load and PostgREST round-trips
- Critical for RL where hundreds of metrics/second are generated

### 4.4 Status vs. Events
- **Status** (job.status field): Coarse job state for simple queries
- **Events**: Fine-grained log with structured data
- **Status Timeline**: Separate table tracking phase transitions
- API `/status` endpoint derives current status from most recent timeline event

### 4.5 Error Handling
- Training workers emit failures via `append_event(level="error")`
- Status set to "failed" via `append_status(phase="failed")`
- Client polls detect terminal status and return immediately
- No retry or recovery logic in client (fail-fast)

### 4.6 Startup & Timeout Guards
- Client enforces 45-second startup deadline for first event
- Prevents infinite polling on stalled/misconfigured jobs
- Empty poll threshold (5) catches event ingestion failures
- Max timeout (default 3600s) prevents unbounded polling

### 4.7 Callback Architecture
- `on_event(event)` - Called for each new event
- `on_metric(point)` - Called for each new metric point
- Enables real-time CLI progress bars, logging, and notifications
- Decouples polling logic from presentation layer

### 4.8 PostgREST Architecture
- Training workers write directly to PostgreSQL via PostgREST HTTP API
- No message queue or intermediate buffer
- Fail-fast: if PostgREST is down, training fails (no silent data loss)
- RPC functions ensure atomic operations (e.g., seq allocation)

---

## 5. POTENTIAL ISSUES & IMPROVEMENTS

### 5.1 Current Issues

**A. Polling Inefficiency**
- Fixed 2-second interval regardless of job phase
- High DB query load during long training runs
- No adaptive polling (e.g., slower during stable training)

**B. No Server-Sent Events (SSE)**
- Client must poll, cannot push events
- Increases latency for status updates
- More backend load than necessary

**C. Empty Poll Threshold Too Strict**
- 5 consecutive empty polls triggers abort
- Can fail on slow event emission (e.g., long epochs)
- No grace period for known-long operations

**D. Linked Job Discovery**
- Relies on `linked_job_id` field in status response
- If field is missing/delayed, client misses events
- No retry or fallback for discovery

**E. Metric Deduplication**
- Client tracks `last_step_by_name` to avoid duplicate metrics
- No server-side deduplication
- Client must handle out-of-order metric delivery

**F. Error Event Ambiguity**
- Both `level="error"` events and `phase="failed"` status exist
- Not always clear which is authoritative
- Client checks both terminal events and status field

### 5.2 Suggested Improvements

**A. Implement SSE for Real-Time Updates**
```python
# Backend: Add SSE endpoint
@router.get("/learning/jobs/{job_id}/stream")
async def stream_job_events(job_id: str):
    async for event in event_stream(job_id):
        yield f"data: {json.dumps(event)}\n\n"

# Client: Use SSE instead of polling
async with aiohttp.ClientSession() as session:
    async with session.get(f"{base_url}/learning/jobs/{job_id}/stream") as resp:
        async for line in resp.content:
            event = json.loads(line.decode())
            on_event(event)
```

**B. Adaptive Polling Intervals**
```python
# Slow down polling during stable phases
if status == "training" and recent_events < 5:
    interval = 10.0  # 10 seconds
elif status in ("queued", "initializing"):
    interval = 2.0   # Fast polling during startup
else:
    interval = 5.0   # Default
```

**C. Graceful Empty Poll Handling**
```python
# Adjust threshold based on known job characteristics
if job_type == "rl" and epoch_duration > 300:
    empty_polls_threshold = 20  # More lenient for long RL epochs
```

**D. Server-Side Metric Aggregation**
```python
# Backend: Pre-aggregate metrics for polling
@router.get("/learning/jobs/{job_id}/metrics/summary")
async def get_metrics_summary(job_id: str):
    # Return downsampled metrics (e.g., every 10th step)
    # Reduces payload size and client processing
```

**E. Add Idempotency to Event Writes**
```python
# Backend: Use idempotency keys for event writes
await emitter.append_event(
    job_id=job_id,
    type_=type_,
    message=message,
    idempotency_key=f"{job_id}-{step}-{event_type}"  # Prevents duplicates
)
```

**F. Unified Terminal Signal**
```python
# Backend: Always emit both terminal event + status update atomically
async def mark_job_terminal(job_id: str, success: bool):
    async with db.transaction():
        await emitter.append_status(
            job_id=job_id,
            phase="succeeded" if success else "failed"
        )
        await emitter.append_event(
            job_id=job_id,
            type_="job.completed" if success else "job.failed"
        )
```

---

## 6. SUMMARY

### Client-Side (synth-ai)
- **Primary Logic**: `JobHandle.poll_until_terminal()` in `synth_ai/learning/jobs.py`
- **Polling Strategy**: Three parallel streams (status, events, metrics) every 2 seconds
- **Termination**: Status in TERMINAL_STATUSES OR terminal event seen
- **Failsafes**: Startup deadline (45s), empty poll threshold (5), max timeout

### Backend (monorepo)
- **Write Path**: Training workers → PostgrestLearningEmitter → PostgREST → PostgreSQL
- **Read Path**: Client → FastAPI routes → LearningJobsRepositoryDB → PostgreSQL
- **SFT Emission**: SFTTrainer + callbacks emit events/metrics per log step
- **RL Emission**: ClusteredGRPOLudicTrainer emits per iteration + evaluation
- **Storage**: Separate tables for events (seq-ordered), status (phase-based), metrics (time-series)

### Critical Components
1. **PostgrestLearningEmitter**: Central write interface with retry logic
2. **JobsApiResolver**: Flexible endpoint resolution for backward compatibility
3. **_MetricBatcher**: Efficient bulk metric insertion
4. **Sequence Tracking**: Monotonic seq numbers prevent duplicate processing
5. **Linked Jobs**: Multi-service RL workflows use linked_job_id for event federation

### Overall Assessment
The polling logic is **robust and functional** but has room for optimization:
- ✅ Handles distributed training, multi-service RL, and high-volume metrics
- ✅ Strong consistency guarantees via PostgREST RPC
- ✅ Graceful degradation with fallback endpoints
- ⚠️ Polling overhead could be reduced with SSE
- ⚠️ Fixed intervals don't adapt to job phase
- ⚠️ Empty poll threshold too strict for long-running epochs



[1 tool called]

Perfect! The job completed successfully. Now let's look at what was captured in the output file:

[3 tools called]

Perfect! I've successfully captured the SFT polling output and written a comprehensive analysis. Here's what we found:

## Summary

**You were right!** The current CLI polling only shows basic status transitions:
- `running` 
- `validating_files` (briefly)
- `succeeded`

### What We Captured
The output file `sft_poll_output.txt` shows the job polled every ~5 seconds for 80 seconds, displaying only the status field. No training progress, loss values, or metrics were shown during the run.

### What's Actually Available (But Not Shown)
According to the analysis in `logging.txt`, the backend IS emitting and storing:

**Events:**
- `sft.created`, `sft.training.started`
- `sft.progress` (step, epoch, loss)
- `sft.validation.summary`
- `sft.job.completed`

**Metrics:**
- `train.loss`, `train.learning_rate`
- `val.loss` (if validation enabled)
- Step/epoch progress

**Status Timeline:**
- Phase transitions with metadata
- Progress percentages

### Why It's Not Shown
The CLI uses a simple `SFTJobPoller` (in `synth_ai/api/train/pollers.py`) that ONLY polls the `/learning/jobs/{job_id}` endpoint for the `status` field. It doesn't fetch:
- `/learning/jobs/{job_id}/events`
- `/learning/jobs/{job_id}/metrics`
- `/learning/jobs/{job_id}/timeline`

### The Better Poller Exists
There's a much richer `JobHandle.poll_until_terminal()` in `synth_ai/learning/jobs.py` that DOES fetch all three streams (status, events, metrics), but the CLI doesn't use it.

I've written detailed findings in:
- **`sft_poll_output.txt`** - The actual captured polling output
- **`sft_polling_analysis.txt`** - Comprehensive analysis with recommendations for improving the CLI poller

The analysis includes a comparison of "Current vs Ideal Output" showing what users COULD see if we upgraded the CLI poller! 🎯

---

## 7. PROPOSED SOLUTION: STREAMING ABSTRACTIONS

See `streaming_abstractions_design.md` for a comprehensive design of flexible, configurable streaming with:

### Key Features
1. **Unified `StreamMessage` format** - All data (status, events, metrics, timeline) flows through one abstraction
2. **`StreamConfig`** - Declarative configuration for what to stream and how to filter
3. **`StreamHandler` protocol** - Pluggable output formatters (CLI, Rich, JSON, callbacks)
4. **`JobStreamer`** - Core multiplexer that polls all streams and routes to handlers
5. **Sensible defaults** - Stream everything with Rich UI by default
6. **Fine-grained control** - Filter by event type, metric name, sample rate, etc.

### Example Usage

**Default (all streams, Rich UI):**
```bash
synth-ai train --config my_config.toml
```

**Custom filtering:**
```bash
synth-ai train --config my_config.toml \
  --stream events --stream metrics \
  --filter-events sft.progress \
  --filter-metrics train.loss
```

**Programmatic:**
```python
config = StreamConfig(
    enabled_streams={StreamType.EVENTS, StreamType.METRICS},
    event_types={"sft.progress", "sft.loss"},
)
streamer = JobStreamer(base_url, api_key, job_id, config=config)
await streamer.stream_until_terminal()
```

### Benefits
✅ Progressive disclosure (simple by default, powerful when needed)
✅ Backward compatible (can wrap existing API)
✅ Extensible (easy to add new stream types, handlers)
✅ Testable (clear separation of concerns)
✅ Production-ready (handles deduplication, sampling, buffering)