# Graph GEPA: Optimize LLM Graph Workflows

Graph GEPA uses evolutionary optimization to automatically improve multi-step LLM workflows (graphs). It evolves graph structure and prompts to maximize performance on your specific task.

## Installation

```bash
pip install synth-ai
# or
uv add synth-ai
```

## Quick Start (10 lines)

```python
from synth_ai.products.graph_gepa import GraphOptimizationConfig, GraphOptimizationClient

# 1. Define your dataset
dataset = {
    "tasks": [
        {"task_id": "q1", "input": {"question": "What is the capital of France?"}},
        {"task_id": "q2", "input": {"question": "Who wrote Romeo and Juliet?"}},
    ],
    "gold_outputs": [
        {"task_id": "q1", "output": {"answer": "Paris"}, "score": 1.0},
        {"task_id": "q2", "output": {"answer": "William Shakespeare"}, "score": 1.0},
    ],
    "metadata": {"name": "qa_demo", "task_description": "Answer questions accurately"}
}

# 2. Configure optimization
config = GraphOptimizationConfig(
    dataset_name="qa_demo",
    dataset=dataset,
    graph_type="policy",
    graph_structure="single_prompt",
)

# 3. Run optimization (job + events pattern)
async with GraphOptimizationClient(base_url="http://localhost:8000") as client:
    job_id = await client.start_job(config)
    
    async for event in client.stream_events(job_id):
        if event.get("type") == "generation_completed":
            print(f"Gen {event['data']['generation']}: {event['data']['best_score']:.2%}")
    
    result = await client.get_result(job_id)
    print(f"Best score: {result['best_score']:.2%}")
```

## Using TOML Configuration

Create a config file (`config.toml`):

```toml
[graph_optimization]
algorithm = "graph_gepa"
dataset_name = "my_task"
graph_type = "policy"
graph_structure = "dag"

[graph_optimization.evolution]
num_generations = 5
children_per_generation = 3

[graph_optimization.proposer]
model = "gpt-4.1"

[graph_optimization.seeds]
train = [0, 1, 2, 3, 4]
validation = [100, 101, 102]

[graph_optimization.limits]
max_spend_usd = 10.0
```

Then run:

```python
from synth_ai.products.graph_gepa import GraphOptimizationConfig, GraphOptimizationClient

config = GraphOptimizationConfig.from_toml("config.toml")

async with GraphOptimizationClient() as client:
    job_id = await client.start_job(config)
    async for event in client.stream_events(job_id):
        print(event)
    result = await client.get_result(job_id)
```

## Graph Types

### Policy Graphs
Solve tasks: take input, produce output.
```python
config = GraphOptimizationConfig(
    graph_type="policy",  # Input -> Output
    task_description="Answer questions about documents",
)
```

### Verifier Graphs
Score/judge outputs: calibrate against human labels.
```python
config = GraphOptimizationConfig(
    graph_type="verifier",  # Input + Trace -> Score
    task_description="Evaluate code quality on a 0-1 scale",
)
```

### RLM Graphs (Recursive Language Model)
Handle massive context (1M+ tokens) by keeping it out of prompts and searching via tools.
```python
config = GraphOptimizationConfig(
    graph_type="rlm",  # Input with large context -> Output
    task_description="Answer questions about a large document corpus",
)
```

**RLM graphs automatically get these tools:**
- `materialize_context` - Store input fields for fast searching (~1ms local)
- `local_grep` - Regex search on materialized content (~1ms)
- `local_search` - Substring search (~1ms)
- `query_lm` - Sub-LM calls for processing chunks
- `codex_exec` - Shell execution for complex operations

**When to use RLM:**
- Context exceeds ~100K tokens (too large for prompt)
- You need to search/filter large datasets
- RAG-style workflows over massive corpora

**Example RLM Dataset:**
```json
{
  "tasks": [
    {
      "task_id": "q1",
      "input": {
        "question": "What was the quarterly revenue?",
        "documents": "<4MB of financial reports>"
      }
    }
  ],
  "gold_outputs": [
    {"task_id": "q1", "output": {"answer": "$4.2B"}, "score": 1.0}
  ],
  "metadata": {
    "name": "financial_qa",
    "task_description": "Answer questions by searching financial documents"
  }
}
```

The system automatically detects large fields (>4M chars / ~1M tokens) and uses RLM patterns.

## Graph Structures

| Structure | Description | Use Case |
|-----------|-------------|----------|
| `single_prompt` | One LLM call | Simple tasks, fast iteration |
| `dag` | Multiple nodes in sequence | Complex reasoning, pipelines |
| `conditional` | Branching logic | Routing, self-consistency |

## Dataset Format (ADAS)

### Policy Dataset Example
```json
{
  "tasks": [
    {
      "task_id": "q1",
      "input": {"question": "What is 2+2?", "context": "Basic math"}
    }
  ],
  "gold_outputs": [
    {
      "task_id": "q1",
      "output": {"answer": "4"},
      "score": 1.0
    }
  ],
  "metadata": {
    "name": "math_qa",
    "task_description": "Answer math questions"
  }
}
```

### Verifier Dataset Example (for calibration)
```json
{
  "tasks": [
    {
      "task_id": "trace_1",
      "input": {
        "code": "def add(a, b): return a + b",
        "test_result": "PASS"
      }
    }
  ],
  "gold_outputs": [
    {
      "task_id": "trace_1",
      "output": {},
      "score": 0.95
    }
  ],
  "metadata": {
    "task_description": "Rate code quality 0-1"
  }
}
```

## Configuration Reference

### Core Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | str | required | Dataset identifier |
| `graph_type` | str | "policy" | "policy", "verifier", or "rlm" (see Graph Types) |
| `graph_structure` | str | "dag" | "single_prompt", "dag", "conditional" |
| `topology_guidance` | str | None | Custom guidance for graph structure (auto-set for RLM) |
| `allowed_policy_models` | list | ["gpt-4o-mini", "gpt-4o"] | Models allowed in graph nodes |

### Evolution Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_generations` | int | 5 | Number of evolution iterations |
| `children_per_generation` | int | 3 | Variants to try per generation |

### Proposer Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | "gpt-4.1" | Model for proposing changes |
| `temperature` | float | 0.7 | Sampling temperature |

### Limits
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_spend_usd` | float | 10.0 | Maximum total spend |
| `timeout_seconds` | int | 3600 | Job timeout |

## Streaming Events

Monitor optimization progress via SSE:

```python
async for event in client.stream_events(job_id):
    if event.type == "generation_complete":
        print(f"Gen {event.generation}: best={event.best_score:.2%}")
    elif event.type == "job_complete":
        print(f"Done! Final score: {event.best_score:.2%}")
```

## Cookbook Examples

- [HotpotQA Optimization](../../cookbooks/graph_gepa/hotpotqa.py) - Multi-hop QA
- [Crafter Verifier Calibration](../../cookbooks/graph_gepa/crafter_verifier.py) - Game trace scoring

## Troubleshooting

### "Dataset has 0 tasks"
Ensure your dataset has the correct structure:
```json
{"tasks": [...], "gold_outputs": [...], "metadata": {...}}
```

### "Invalid task_id reference"
Each gold_output.task_id must match a task.task_id in the tasks list.

### Optimization stuck at low score
- Try more seeds: `seeds.train = [0, 1, 2, ..., 9]`
- Increase generations: `evolution.num_generations = 10`
- Use a simpler structure: `graph_structure = "single_prompt"`

## API Reference

See the [Pydantic models](./config.py) for complete schema documentation.
