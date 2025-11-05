# Dataclass Migration Plan: Moving Away from Untyped Dicts

## Problem Statement

We've encountered multiple issues due to loose, untyped dictionary usage:
1. **JSON serialization failures**: Non-serializable objects (PromptTransformation) hidden in nested dicts
2. **Silent data loss**: Validation seeds disappearing without errors (N=0 issue)
3. **Implicit behavior**: `get/or` chains with fallback defaults masking missing data
4. **Type safety**: No IDE autocomplete, no type checking, easy to misname keys
5. **Runtime crashes**: AttributeError when accessing missing keys as attributes

## Success Criteria

✅ **Type Safety**: All data structures have explicit types, caught at dev time
✅ **Validation**: Data validated at boundary (parse from dict/JSON)
✅ **Serialization**: All dataclasses JSON-serializable by design
✅ **Explicit Errors**: Missing/wrong data raises clear errors with field names
✅ **IDE Support**: Autocomplete and type hints everywhere

## Migration Strategy

### Phase 1: Core Configuration (✅ DONE)
- [x] `GEPAEvaluationConfig` - validation seeds, pools, top-k
- [x] `GEPARolloutConfig` - budget, concurrency
- [x] `GEPAMutationConfig` - LLM settings
- [x] `GEPAPopulationConfig` - pool management
- [x] `GEPAArchiveConfig` - Pareto archive settings
- [x] `GEPAFullConfig` - top-level GEPA config

### Phase 2: Event Payloads (🚧 IN PROGRESS)

#### 2.1 Archive & Candidates
```python
@dataclass
class PromptScore:
    """Score metrics for a prompt."""
    accuracy: float
    prompt_length: int
    tool_call_rate: float
    instance_scores: List[float]
    
    def __post_init__(self):
        assert 0.0 <= self.accuracy <= 1.0, f"accuracy must be [0,1], got {self.accuracy}"
        assert self.prompt_length > 0, f"prompt_length must be >0, got {self.prompt_length}"
        assert 0.0 <= self.tool_call_rate <= 1.0, f"tool_call_rate must be [0,1], got {self.tool_call_rate}"
        assert isinstance(self.instance_scores, list), f"instance_scores must be list"

@dataclass
class OptimizedCandidate:
    """A candidate from the Pareto archive."""
    rank: int
    score: PromptScore
    payload_kind: str  # "transformation" | "template" | "pattern"
    object: Dict[str, Any]  # Serialized prompt object
    trace: Dict[str, Any]
    instance_scores: List[float]  # Explicit copy for API responses
    
    def __post_init__(self):
        assert self.rank >= 0, f"rank must be >=0, got {self.rank}"
        assert self.payload_kind in ("transformation", "template", "pattern"), \
            f"payload_kind must be transformation/template/pattern, got {self.payload_kind}"
        # Ensure object is JSON-serializable
        try:
            json.dumps(self.object)
        except (TypeError, ValueError) as e:
            raise ValueError(f"object must be JSON-serializable: {e}")
    
    @classmethod
    def from_archive_item(cls, item: Dict[str, Any], rank: int) -> "OptimizedCandidate":
        """Parse from archive item with explicit validation."""
        score_dict = item.get("score", {})
        payload = item.get("payload", {})
        
        if not score_dict:
            raise ValueError(f"Archive item {rank} missing 'score'")
        if not payload:
            raise ValueError(f"Archive item {rank} missing 'payload'")
        
        score = PromptScore(
            accuracy=float(score_dict.get("accuracy", 0.0)),
            prompt_length=int(score_dict.get("prompt_length", 0)),
            tool_call_rate=float(score_dict.get("tool_call_rate", 0.0)),
            instance_scores=list(score_dict.get("instance_scores", [])),
        )
        
        return cls(
            rank=rank,
            score=score,
            payload_kind=str(payload.get("kind", "unknown")),
            object=payload.get("object", {}),
            trace=payload.get("trace", {}),
            instance_scores=list(payload.get("instance_scores", [])),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "rank": self.rank,
            "score": {
                "accuracy": self.score.accuracy,
                "prompt_length": self.score.prompt_length,
                "tool_call_rate": self.score.tool_call_rate,
                "instance_scores": self.score.instance_scores,
            },
            "payload_kind": self.payload_kind,
            "object": self.object,
            "trace": self.trace,
            "instance_scores": self.instance_scores,
        }
```

#### 2.2 Validation Results
```python
@dataclass
class ValidationResult:
    """Validation evaluation result for a single candidate."""
    rank: int
    template: Dict[str, Any]  # Serialized PromptTemplate
    accuracy: float
    instance_scores: List[float]
    train_accuracy: float
    lift_pct_vs_baseline: Optional[float]
    split: str  # "validation"
    seeds: List[int]
    pool: str  # "test" | "train" | "validation"
    
    def __post_init__(self):
        assert self.rank >= 0
        assert 0.0 <= self.accuracy <= 1.0
        assert 0.0 <= self.train_accuracy <= 1.0
        assert self.split == "validation", f"split must be 'validation', got {self.split}"
        assert self.pool in ("test", "train", "validation", "val")
        assert len(self.seeds) > 0, "seeds cannot be empty"
        assert len(self.instance_scores) == len(self.seeds), \
            f"instance_scores length {len(self.instance_scores)} != seeds length {len(self.seeds)}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Parse with explicit validation."""
        required = ["rank", "template", "accuracy", "instance_scores", 
                   "train_accuracy", "split", "seeds", "pool"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"ValidationResult missing required fields: {missing}")
        
        return cls(**{k: data[k] for k in required}, 
                   lift_pct_vs_baseline=data.get("lift_pct_vs_baseline"))

@dataclass
class ValidationSummary:
    """Complete validation results for a GEPA run."""
    baseline: Optional[ValidationResult]
    results: List[ValidationResult]
    split: str  # "validation"
    seeds: List[int]
    pool: str
    top_k: int
    
    def __post_init__(self):
        assert len(self.results) <= self.top_k, \
            f"results length {len(self.results)} > top_k {self.top_k}"
        assert all(isinstance(r, ValidationResult) for r in self.results)
```

#### 2.3 CLI Top-K Results
```python
@dataclass
class TopKPromptResult:
    """A single top-K prompt with train/val scores (for CLI display)."""
    rank: int
    train_accuracy: float
    val_accuracy: Optional[float]
    prompt_preview: str
    
    def __post_init__(self):
        assert self.rank >= 0
        assert 0.0 <= self.train_accuracy <= 1.0
        if self.val_accuracy is not None:
            assert 0.0 <= self.val_accuracy <= 1.0
    
    @classmethod
    def from_event_message(cls, event: Dict[str, Any]) -> Optional["TopKPromptResult"]:
        """Parse from optimized.scored event message."""
        msg = event.get("message", "")
        if "optimized[" not in msg:
            return None
        
        try:
            idx = int(msg.split("optimized[")[1].split("]")[0])
            train_acc = float(msg.split("train_accuracy=")[1].split()[0])
            
            val_acc = None
            if "val_accuracy=" in msg:
                val_acc = float(msg.split("val_accuracy=")[1].split()[0])
            
            prompt_preview = ""
            if "✨ TRANSFORMATION:" in msg:
                prompt_preview = msg.split("✨ TRANSFORMATION:")[1].strip()[:200]
            
            return cls(
                rank=idx,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                prompt_preview=prompt_preview,
            )
        except (IndexError, ValueError, AttributeError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "prompt_preview": self.prompt_preview,
        }
```

### Phase 3: API Request/Response Types

#### 3.1 Job Creation
```python
@dataclass
class PromptLearningJobRequest:
    """Request to create a prompt learning job."""
    config: Dict[str, Any]  # Raw TOML parsed
    initial_prompt: Dict[str, Any]  # Serialized PromptPattern/Template
    task_app_url: str
    task_app_api_key: str
    algorithm: str  # "gepa" | "mipro"
    
    def __post_init__(self):
        assert self.algorithm in ("gepa", "mipro")
        assert self.task_app_url.startswith("http")

@dataclass
class PromptLearningJobResponse:
    """Response from job creation."""
    job_id: str
    status: str
    created_at: str
    org_id: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptLearningJobResponse":
        return cls(
            job_id=str(data["job_id"]),
            status=str(data.get("status", "created")),
            created_at=str(data.get("created_at", "")),
            org_id=str(data.get("org_id", "")),
        )
```

### Phase 4: Internal Optimizer State

#### 4.1 Archive Items (in optimizer.py)
```python
@dataclass
class ArchivePayload:
    """Payload stored in Pareto archive."""
    kind: str  # "transformation" | "template"
    object: Any  # PromptTransformation | PromptTemplate
    trace: Dict[str, Any]
    instance_scores: List[float]
    
    def to_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        if isinstance(self.object, PromptTemplate):
            obj_dict = _serialize_prompt_template(self.object)
        elif isinstance(self.object, PromptTransformation):
            obj_dict = _serialize_prompt_transformation(self.object)
        else:
            obj_dict = {"repr": repr(self.object)}
        
        # Sanitize trace
        safe_trace = {
            k: v for k, v in self.trace.items()
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
        }
        
        return {
            "kind": self.kind,
            "object": obj_dict,
            "trace": safe_trace,
            "instance_scores": list(self.instance_scores),
        }
```

## Implementation Checklist

### 1. Create Dataclass Module
- [ ] Create `backend/app/routes/prompt_learning/types/` package
- [ ] `types/scores.py` - PromptScore, scoring metrics
- [ ] `types/candidates.py` - OptimizedCandidate, AttemptedCandidate
- [ ] `types/validation.py` - ValidationResult, ValidationSummary
- [ ] `types/archive.py` - ArchivePayload, ArchiveItem
- [ ] `types/requests.py` - API request dataclasses
- [ ] `types/responses.py` - API response dataclasses
- [ ] `types/__init__.py` - Export all types

### 2. Create Unit Tests
- [ ] `tests/unit/routes/prompt_learning/test_types_scores.py`
- [ ] `tests/unit/routes/prompt_learning/test_types_candidates.py`
- [ ] `tests/unit/routes/prompt_learning/test_types_validation.py`
- [ ] `tests/unit/routes/prompt_learning/test_serialization.py`
- [ ] `tests/unit/routes/prompt_learning/test_archive_payload.py`

### 3. Create SDK Dataclasses
- [ ] `synth_ai/learning/types/` package
- [ ] `types/results.py` - TopKPromptResult, job results
- [ ] Unit tests in `tests/unit/learning/test_cli_types.py`

### 4. Migrate Code Incrementally
- [ ] Replace `optimized_candidates` list[dict] → list[OptimizedCandidate]
- [ ] Replace `validation_results` list[dict] → list[ValidationResult]
- [ ] Replace archive item dicts → ArchivePayload
- [ ] Add `.to_dict()` calls at serialization boundaries
- [ ] Remove all `get/or` chains, replace with dataclass parsing
- [ ] Add assertions in `__post_init__` for all invariants

### 5. Add Comprehensive Tests
- [ ] Test JSON serialization round-trip for all dataclasses
- [ ] Test validation failures (missing fields, wrong types, out-of-range values)
- [ ] Test `from_dict` parsing with malformed data
- [ ] Test `to_dict` produces JSON-serializable output
- [ ] Integration test: full GEPA run with dataclasses end-to-end

## Benefits

1. **Catch errors at boundaries**: Data validated when parsing from dicts/JSON
2. **Explicit serialization**: `to_dict()` methods ensure JSON-safe output
3. **IDE autocomplete**: Type hints enable intelligent code completion
4. **Refactoring safety**: Rename fields, IDEs update all references
5. **Documentation**: Dataclass fields are self-documenting
6. **Testing**: Easy to create test fixtures with known-good data

## Example: Before vs After

### Before (Loose Dict)
```python
# Silent failure - typo goes unnoticed
candidate = {
    "scor": {"accuracy": 0.8},  # TYPO: "scor" instead of "score"
    "instance_scores": [0, 1, 1],
}
acc = candidate.get("score", {}).get("accuracy", 0.0)  # Returns 0.0, bug hidden!
```

### After (Dataclass)
```python
# Explicit failure at parse time
try:
    candidate = OptimizedCandidate.from_archive_item(item, rank=0)
except ValueError as e:
    logger.error(f"Failed to parse candidate: {e}")  # Clear error message
    raise

# Type-safe access
acc = candidate.score.accuracy  # IDE autocomplete, type checked
```

## Migration Timeline

- **Week 1**: Create dataclass modules + unit tests (Phases 1-2)
- **Week 2**: Migrate backend `online_jobs.py` to use dataclasses
- **Week 3**: Migrate SDK CLI to use dataclasses
- **Week 4**: Integration testing, remove old dict code

## Risk Mitigation

1. **Incremental**: Migrate one dataclass at a time
2. **Parallel**: Keep dict code working during migration
3. **Tests**: Add tests before and after migration
4. **Review**: Each dataclass gets code review before merging
5. **Rollback**: Can revert individual dataclasses if issues arise

