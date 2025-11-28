# SDK Research Agent Parity - Scoping Document

## Overview

The `ResearchAgentJob` SDK currently uses a generic `algorithm_config: Dict[str, Any]` approach, which lacks type safety and doesn't expose all the configuration options that the backend supports. This document scopes the changes needed to achieve full parity with the backend API.

## Current State

### SDK (`synth_ai/sdk/api/research_agent/job.py`)

```python
@dataclass
class ResearchAgentJobConfig:
    algorithm: AlgorithmType  # "research" only
    repo_url: str = ""
    repo_branch: str = "main"
    repo_commit: Optional[str] = None
    inline_files: Optional[Dict[str, str]] = None
    backend: BackendType = "daytona"
    model: str = "gpt-4o"
    use_synth_proxy: bool = True
    algorithm_config: Dict[str, Any] = field(default_factory=dict)  # ← Untyped!
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Backend (`backend/app/routes/research_agent/models.py`)

The backend accepts `ResearchAgentJobRequest` which has:
- All the fields the SDK has
- **Plus** typed `ResearchConfig` for the "research" algorithm
- **Plus** spend limits (`max_agent_spend_usd`, `max_synth_spend_usd`)
- **Plus** `reasoning_effort` for supported models

## Gap Analysis

### Missing from SDK

| Feature | Backend Support | SDK Support | Priority |
|---------|----------------|-------------|----------|
| `max_agent_spend_usd` | ✅ Default $10 | ❌ | **High** |
| `max_synth_spend_usd` | ✅ Default $100 | ❌ | **High** |
| `reasoning_effort` | ✅ low/medium/high | ❌ | **High** |
| Typed `ResearchConfig` | ✅ Full Pydantic model | ❌ Uses Dict | **High** |
| `DatasetSource` types | ✅ huggingface/upload/inline | ❌ Manual dict | **Medium** |
| `PermittedModelsConfig` | ✅ Typed | ❌ Manual dict | **Medium** |
| `GEPAModelConfig` | ✅ Typed | ❌ Manual dict | **Medium** |
| `MIPROModelConfig` | ✅ Typed | ❌ Manual dict | **Medium** |

### Backend Models to Mirror in SDK

The SDK should have Python equivalents of these backend models:

1. **ResearchConfig** - Main config for "research" algorithm
2. **DatasetSource** - Dataset configuration (huggingface, upload, inline)
3. **PermittedModelsConfig** - Models allowed in pipeline
4. **PermittedModel** - Single model config
5. **GEPAModelConfig** - GEPA-specific settings
6. **MIPROModelConfig** - MIPRO-specific settings
7. **ModelProvider** - Enum: openai, groq, google
8. **OptimizationTool** - Enum: mipro, gepa

## Proposed SDK Structure

### New File: `synth_ai/sdk/api/research_agent/config.py`

```python
"""Typed configuration models for Research Agent jobs."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

class ModelProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"

class OptimizationTool(str, Enum):
    MIPRO = "mipro"
    GEPA = "gepa"

@dataclass
class PermittedModel:
    model: str
    provider: ModelProvider

@dataclass
class PermittedModelsConfig:
    models: List[PermittedModel] = field(default_factory=list)
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

@dataclass
class DatasetSource:
    source_type: Literal["huggingface", "upload", "inline"]
    description: Optional[str] = None
    # HuggingFace
    hf_repo_id: Optional[str] = None
    hf_split: str = "train"
    hf_subset: Optional[str] = None
    # Upload
    file_ids: Optional[List[str]] = None
    # Inline
    inline_data: Optional[dict[str, str]] = None

@dataclass
class GEPAConfig:
    mutation_model: str = "openai/gpt-oss-120b"
    mutation_provider: ModelProvider = ModelProvider.GROQ
    mutation_temperature: float = 0.7
    mutation_max_tokens: int = 8192
    population_size: int = 20
    num_generations: int = 10
    elite_fraction: float = 0.2
    # Proposer settings
    proposer_type: Literal["dspy", "spec"] = "dspy"
    proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    proposer_output_tokens: Literal["RAPID", "FAST", "SLOW"] = "FAST"
    spec_path: Optional[str] = None
    # Seed pool sizes
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    reference_size: Optional[int] = None

@dataclass
class MIPROConfig:
    meta_model: str = "llama-3.3-70b-versatile"
    meta_provider: ModelProvider = ModelProvider.GROQ
    meta_temperature: float = 0.7
    meta_max_tokens: int = 4096
    num_candidates: int = 20
    num_trials: int = 10
    # Proposer settings
    proposer_effort: Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    proposer_output_tokens: Literal["RAPID", "FAST", "SLOW"] = "FAST"
    # Seed pool sizes
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    reference_size: Optional[int] = None

@dataclass
class ResearchConfig:
    """Configuration for prompt/pipeline research optimization."""

    task_description: str
    tools: List[OptimizationTool] = field(default_factory=lambda: [OptimizationTool.MIPRO])

    # Datasets
    datasets: List[DatasetSource] = field(default_factory=list)

    # Metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=list)

    # Optimization params
    num_iterations: int = 10
    population_size: int = 20
    timeout_minutes: int = 60
    max_eval_samples: Optional[int] = None

    # Model configs
    permitted_models: Optional[PermittedModelsConfig] = None
    gepa_config: Optional[GEPAConfig] = None
    mipro_config: Optional[MIPROConfig] = None

    # Initial prompt/pipeline
    initial_prompt: Optional[str] = None
    pipeline_entrypoint: Optional[str] = None
```

### Updated: `synth_ai/sdk/api/research_agent/job.py`

```python
@dataclass
class ResearchAgentJobConfig:
    """Configuration for a research agent job."""

    # Research config (typed!)
    research: ResearchConfig

    # Repository (optional if inline_files provided)
    repo_url: str = ""
    repo_branch: str = "main"
    repo_commit: Optional[str] = None
    inline_files: Optional[Dict[str, str]] = None

    # Execution
    backend: BackendType = "daytona"
    model: str = "gpt-4o"
    use_synth_proxy: bool = True

    # NEW: Spend limits
    max_agent_spend_usd: float = 10.0
    max_synth_spend_usd: float = 100.0

    # NEW: Reasoning effort
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None

    # API configuration
    backend_url: str = ""
    api_key: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Changes Required

### SDK Changes (synth-ai repo)

1. **New file**: `synth_ai/sdk/api/research_agent/config.py`
   - All typed config dataclasses
   - Enums for ModelProvider, OptimizationTool
   - ~200 lines

2. **Update**: `synth_ai/sdk/api/research_agent/job.py`
   - Import new config types
   - Replace `algorithm_config: Dict` with `research: ResearchConfig`
   - Add `max_agent_spend_usd`, `max_synth_spend_usd`, `reasoning_effort`
   - Update `submit()` to serialize typed config to JSON
   - Update TOML parsing to construct typed objects
   - ~100 lines changed

3. **Update**: `synth_ai/sdk/api/research_agent/__init__.py`
   - Export new config types

4. **New/Update**: Tests
   - Unit tests for config serialization
   - Integration test with backend

### Backend Changes (monorepo)

**None required.** The backend already accepts all these fields - the SDK just needs to expose them.

### Backward Compatibility

The current SDK uses `algorithm_config: Dict[str, Any]` which gets passed directly to the backend. We have two options:

**Option A: Breaking change**
- Remove `algorithm_config`, require typed `ResearchConfig`
- Cleaner API, forces users to update

**Option B: Deprecation path**
- Keep `algorithm_config` as fallback
- If `research` is provided, use it; otherwise fall back to `algorithm_config`
- Add deprecation warning

**Recommendation**: Option A (breaking change) since the SDK is new and likely has few users.

## Example Usage After Changes

```python
from synth_ai.sdk.api.research_agent import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
    ResearchConfig,
    DatasetSource,
    MIPROConfig,
    PermittedModelsConfig,
    PermittedModel,
    ModelProvider,
    OptimizationTool,
)

# Create typed config
research_config = ResearchConfig(
    task_description="Optimize prompt for banking intent classification",
    tools=[OptimizationTool.MIPRO],
    datasets=[
        DatasetSource(
            source_type="huggingface",
            hf_repo_id="PolyAI/banking77",
            hf_split="train",
        )
    ],
    permitted_models=PermittedModelsConfig(
        models=[
            PermittedModel(model="gpt-4o-mini", provider=ModelProvider.OPENAI),
            PermittedModel(model="llama-3.3-70b-versatile", provider=ModelProvider.GROQ),
        ]
    ),
    mipro_config=MIPROConfig(
        meta_model="llama-3.3-70b-versatile",
        num_trials=15,
        proposer_effort="HIGH",
    ),
    num_iterations=10,
    timeout_minutes=120,
)

# Create job config
job_config = ResearchAgentJobConfig(
    research=research_config,
    repo_url="https://github.com/my-org/my-pipeline",
    repo_branch="main",
    model="gpt-5.1-codex-mini",
    reasoning_effort="medium",
    max_agent_spend_usd=25.0,
    max_synth_spend_usd=150.0,
)

# Submit job
job = ResearchAgentJob(config=job_config)
job_id = job.submit()
result = job.poll_until_complete(timeout=7200)
```

## Estimated Effort

| Task | Estimate |
|------|----------|
| Create `config.py` with typed dataclasses | 2-3 hours |
| Update `job.py` with new fields and serialization | 2-3 hours |
| Update TOML parsing to construct typed objects | 1-2 hours |
| Write unit tests | 2-3 hours |
| Write integration test | 1-2 hours |
| Documentation/examples | 1 hour |
| **Total** | **~10-14 hours** |

## Open Questions

1. **TOML format**: Should we update the TOML schema to match the typed config structure? Currently it's quite flat.

2. **CLI support**: The SDK has a CLI (`synth-ai agent run`). Should it expose all these options as CLI flags, or require a config file for complex configs?

3. **Validation**: Should we validate configs client-side (SDK) or rely on backend validation? Recommend: basic validation in SDK, full validation on backend.

4. **Dataset upload flow**: The frontend has a file upload flow that returns `file_id`s. Should the SDK have helpers for this, or just accept `file_id`s?
