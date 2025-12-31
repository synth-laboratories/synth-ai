# Verifier V3 Trace Alignment Plan

## Problem Statement

The current verifier implementation uses a simplified demo format that doesn't align with the actual synth-ai v3 trace structures. This creates:
- Incompatibility with the real tracing system
- Inability to properly link rewards to events
- Potential data corruption/confusion

## Current State (WRONG)

### Demo Format Being Used
```python
# WRONG - simplified demo format
{
    "session_id": "trace_001",
    "timesteps": [
        {"turn": 0, "action": "move_right", "reward": 0.0, "obs": {...}},
        {"turn": 1, "action": "do", "reward": 1.0, "obs": {...}},
    ],
    "outcome": {"achievements_count": 2, "total_steps": 3}
}
```

### What Backend Code Expects
- `routes.py`: Parses `trace.get("timesteps", [])` and `trace.get("outcome", {})`
- `graph_prompts.py`: Teaches verifier to output `event_rewards: [0.0, 1.0, ...]` array
- Response models: `EventReward` with string `event_id` like `"trace_001_turn_0"`

## Target State (CORRECT)

### V3 Trace Structure (from synth_ai/core/tracing_v3/abstractions.py)

```python
@dataclass
class SessionTrace:
    session_id: str
    created_at: datetime
    session_time_steps: list[SessionTimeStep]
    event_history: list[BaseEvent]
    markov_blanket_message_history: list[SessionEventMarkovBlanketMessage]
    metadata: dict[str, Any]

@dataclass
class SessionTimeStep:
    step_id: str
    step_index: int
    timestamp: datetime
    turn_number: int | None
    events: list[BaseEvent]  # RuntimeEvent, EnvironmentEvent, LMCAISEvent
    markov_blanket_messages: list[SessionEventMarkovBlanketMessage]
    step_metadata: dict[str, Any]
    completed_at: datetime | None

@dataclass
class EnvironmentEvent(BaseEvent):
    reward: float  # Environment-provided reward
    terminated: bool
    truncated: bool
    system_state_before: dict | None
    system_state_after: dict | None
```

### V3 Reward Structures (from synth_ai/core/tracing_v3/turso/models.py)

```python
class EventReward(Base):
    __tablename__ = "event_rewards"
    id: int  # PK
    event_id: int  # FK to events.id (INTEGER, not string!)
    session_id: str  # FK to session_traces.session_id
    message_id: int | None  # FK to messages.id
    turn_number: int | None
    reward_value: float
    reward_type: str  # shaped | sparse | achievement | penalty | evaluator | human
    key: str | None  # e.g., achievement name
    annotation: JSON  # free-form
    source: str | None  # environment | runner | evaluator | human
    created_at: datetime

class OutcomeReward(Base):
    __tablename__ = "outcome_rewards"
    id: int  # PK
    session_id: str  # FK to session_traces.session_id
    total_reward: int  # NEEDS TO BE FLOAT
    achievements_count: int
    total_steps: int
    reward_metadata: JSON
    created_at: datetime
```

### Dataclass Records (from synth_ai/data/rewards.py)

```python
@dataclass
class EventRewardRecord:
    event_id: str  # Note: str here, int in DB
    session_id: str
    reward_value: float
    reward_type: str | None
    key: str | None
    turn_number: int | None
    source: str | None
    annotation: dict[str, Any]
    created_at: datetime | None

@dataclass
class OutcomeRewardRecord:
    session_id: str
    total_reward: float  # float in dataclass
    achievements_count: int
    total_steps: int
    metadata: dict[str, Any]
    created_at: datetime | None
```

## Required Changes

### 1. synth-ai Changes

#### 1.1 Update OutcomeReward DB Model
**File**: `synth_ai/core/tracing_v3/turso/models.py`
```python
# BEFORE
total_reward = Column(Integer, nullable=False)

# AFTER
total_reward = Column(Float, nullable=False)
annotation = Column(JSONText)  # ADD this field
```

#### 1.2 Add Migration
Create alembic migration to alter `outcome_rewards.total_reward` from INTEGER to FLOAT and add `annotation` column.

### 2. Backend (monorepo) Changes

#### 2.1 Update API Models
**File**: `backend/app/routes/graphgen/routes.py`

```python
# BEFORE - wrong format
class EventReward(BaseModel):
    event_id: str  # String like "trace_001_turn_0"
    session_id: str
    reward_value: float
    ...

# AFTER - align with v3
class EventRewardResponse(BaseModel):
    \"\"\"Matches synth_ai EventRewardRecord structure.\"\"\"
    event_id: int  # Integer FK to events table
    session_id: str
    reward_value: float
    reward_type: Optional[Literal[\"shaped\", \"sparse\", \"achievement\", \"penalty\", \"evaluator\", \"human\"]] = \"evaluator\"
    key: Optional[str] = None
    turn_number: Optional[int] = None
    source: Optional[Literal[\"environment\", \"runner\", \"evaluator\", \"human\"]] = \"evaluator\"
    annotation: Optional[Dict[str, Any]] = None

class OutcomeRewardResponse(BaseModel):
    \"\"\"Matches synth_ai OutcomeRewardRecord structure.\"\"\"
    session_id: str
    total_reward: float
    achievements_count: int = 0
    total_steps: int = 0
    metadata: Optional[Dict[str, Any]] = None
    annotation: Optional[Dict[str, Any]] = None  # Add annotation field
```

#### 2.2 Update Verifier Request Model
**File**: `backend/app/routes/graphgen/routes.py`

The verifier endpoint should accept a proper V3 SessionTrace, not the demo format:

```python
# BEFORE
class GraphGenGraphVerifierRequest(BaseModel):
    job_id: str
    trace: Dict[str, Any]  # Accepts anything
    ...

# AFTER
class GraphGenGraphVerifierRequest(BaseModel):
    job_id: str
    session_trace: SessionTraceInput  # Properly typed V3 trace
    ...

class SessionTraceInput(BaseModel):
    \"\"\"V3-compatible session trace input.\"\"\"
    session_id: str
    session_time_steps: List[SessionTimeStepInput]
    metadata: Optional[Dict[str, Any]] = None

class SessionTimeStepInput(BaseModel):
    \"\"\"V3-compatible timestep input.\"\"\"
    step_id: str
    step_index: int
    turn_number: Optional[int] = None
    events: List[EventInput]
    step_metadata: Optional[Dict[str, Any]] = None

class EventInput(BaseModel):
    \"\"\"V3-compatible event input.\"\"\"
    event_type: Literal[\"environment\", \"runtime\", \"cais\"]
    event_id: int  # Must be provided for reward linking
    system_instance_id: Optional[str] = None
    # Environment event fields
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None
    system_state_before: Optional[Dict[str, Any]] = None
    system_state_after: Optional[Dict[str, Any]] = None
    # Runtime event fields
    actions: Optional[List[int]] = None
    # CAIS event fields
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Common
    metadata: Optional[Dict[str, Any]] = None
```

#### 2.3 Update Verifier Response Parsing
**File**: `backend/app/routes/graphgen/routes.py`

The parsing logic needs to:
1. Build a map of valid event_ids from the input trace
2. Validate that output event_ids exist in the input
3. Parse event rewards keyed by event_id

```python
# In graph_verifier endpoint:
def parse_verifier_output(
    raw_output: dict,
    session_trace: SessionTraceInput,
) -> Tuple[List[EventRewardResponse], OutcomeRewardResponse]:
    \"\"\"Parse verifier output and validate event_id references.\"\"\"

    # Build map of valid event_ids from input trace
    valid_event_ids: Dict[int, EventInput] = {}
    event_index = 0
    for step in session_trace.session_time_steps:
        for event in step.events:
            valid_event_ids[event.event_id] = event
            event_index += 1

    # Parse event rewards (keyed by event_id from verifier output)
    event_rewards = []
    raw_event_rewards = raw_output.get(\"event_rewards\", [])

    for reward_data in raw_event_rewards:
        event_id = reward_data.get(\"event_id\")

        # Validate event_id exists in input trace
        if event_id not in valid_event_ids:
            raise ValueError(f\"Verifier output references unknown event_id: {event_id}\")

        event_rewards.append(EventRewardResponse(
            event_id=event_id,  # Directly from verifier output
            session_id=session_trace.session_id,
            reward_value=float(reward_data.get(\"value\", 0.0)),
            reward_type=\"evaluator\",
            source=\"evaluator\",
            annotation=reward_data.get(\"annotation\"),
        ))

    # Build outcome reward
    outcome_data = raw_output.get(\"outcome\", {})
    outcome_reward = OutcomeRewardResponse(
        session_id=session_trace.session_id,
        total_reward=float(outcome_data.get(\"total_reward\", 0.0)),
        achievements_count=outcome_data.get(\"achievements_count\", 0),
        total_steps=len(valid_event_ids),
        annotation=outcome_data.get(\"annotation\"),
    )

    return event_rewards, outcome_reward
```

#### 2.4 Update Verifier Graph Prompts
**File**: `backend/graphs/gepa_integration/graph_prompts.py`

Update the verifier generation prompt to:
1. Accept V3 SessionTrace format (with event_ids in input)
2. Output rewards keyed by actual event_id (not index)
3. Use proper reward_type vocabulary

```python
# Input to verifier includes event_ids:
{
    "session_id": "sess_abc123",
    "session_time_steps": [
        {
            "step_id": "step_0",
            "events": [
                {"event_id": 12345, "event_type": "environment", "reward": 0.0, ...},
                {"event_id": 12346, "event_type": "runtime", "actions": [0], ...}
            ]
        },
        ...
    ]
}

# The verifier should output (keyed by event_id):
{
    "event_rewards": [
        {"event_id": 12345, "value": 0.0, "annotation": {"reason": "No progress made"}},
        {"event_id": 12346, "value": 0.5, "annotation": {"reason": "Action taken but suboptimal"}},
    ],
    "outcome": {
        "session_id": "sess_abc123",
        "total_reward": 0.267,
        "achievements_count": 4,
        "annotation": {"feedback": "Agent made moderate progress..."}
    }
}
```

#### 2.5 Update Graph Execution to Record LMCAISEvents
When the verifier graph runs LLM calls, those should be recorded as LMCAISEvent entries that can be traced back.

### 3. Frontend Changes

#### 3.1 Update Dataset Upload Format
The frontend dataset upload should validate V3 trace format, not accept arbitrary JSON.

#### 3.2 Update Job Creation
When creating verifier jobs, ensure the dataset contains valid V3 traces.

### 4. Validation & Migration

#### 4.1 Input Validation
Add strict validation that rejects non-V3 trace formats:

```python
def validate_v3_trace(data: dict) -> SessionTraceInput:
    \"\"\"Validate and parse V3 trace format. Reject demo formats.\"\"\"

    # Reject demo format
    if "timesteps" in data and "session_time_steps" not in data:
        raise ValueError(
            "Invalid trace format. Expected V3 SessionTrace with 'session_time_steps', "
            "got demo format with 'timesteps'. Please convert to V3 format."
        )

    # Validate required fields
    if "session_id" not in data:
        raise ValueError("Missing required field: session_id")
    if "session_time_steps" not in data:
        raise ValueError("Missing required field: session_time_steps")

    return SessionTraceInput(**data)
```

#### 4.2 Backward Compatibility
Decide whether to:
- A) Hard reject old format (breaking change)
- B) Auto-convert old format with deprecation warning
- C) Support both with feature flag

**Recommendation**: Option A - hard reject. The old format was never production-ready.

### 5. Testing

#### 5.1 Create V3 Test Fixtures
Create proper V3 trace fixtures for testing:

```python
def create_v3_test_trace() -> SessionTraceInput:
    return SessionTraceInput(
        session_id="test_session_001",
        session_time_steps=[
            SessionTimeStepInput(
                step_id="step_0",
                step_index=0,
                turn_number=0,
                events=[
                    EventInput(
                        event_type="environment",
                        event_id=1,
                        reward=0.0,
                        terminated=False,
                        system_state_after={"hp": 9, "food": 9},
                    ),
                    EventInput(
                        event_type="runtime",
                        event_id=2,
                        actions=[0],  # move_right
                    ),
                ],
            ),
            # ... more steps
        ],
        metadata={"environment": "CrafterClassic"},
    )
```

#### 5.2 Integration Tests
- Test verifier endpoint with V3 traces
- Test that EventReward.event_id links to actual events
