# GRPO Synth Envs Hosted Service

This service provides hosted environment and policy management for GRPO (Group Relative Policy Optimization) training with synthetic environments.

## Architecture

The service implements a FastAPI-based HTTP API that manages:
- **Environments**: Stateful environment instances (currently Crafter)
- **Policies**: Thin policy clients that prepare inference requests
- **Rollouts**: Coordinated execution of environment-policy interaction loops
- **Snapshots**: State persistence using Modal Volumes
- **Branching**: Creating multiple copies of environments/policies for exploration

## Key Components

### Core Modules
- `hosted_app.py`: FastAPI app factory and configuration
- `registry.py`: In-memory registries for active instances
- `storage/volume.py`: Modal Volume operations for snapshots
- `inference/openai_client.py`: OpenAI-compatible inference client

### API Routers
- `environment_routes.py`: Environment lifecycle endpoints
- `policy_routes.py`: Policy lifecycle endpoints
- `rollout.py`: Rollout coordinator and run management
- `branching.py`: Branching operations

### Environment Implementations
- `envs/crafter/`: Crafter environment and policy implementations

## API Endpoints

### Service Discovery
- `GET /info`: Service configuration and endpoints
- `GET /health`: Health check

### Environment Management
- `POST /env/create`: Create new environment
- `POST /env/reset`: Reset environment
- `POST /env/step`: Execute environment step
- `POST /env/snapshot`: Save environment state
- `POST /env/restore`: Restore from snapshot
- `POST /env/terminate`: Clean up environment

### Policy Management
- `POST /policy/create`: Create new policy
- `POST /policy/step`: Generate actions (with optional inference)
- `POST /policy/snapshot`: Save policy state
- `POST /policy/restore`: Restore from snapshot
- `POST /policy/terminate`: Clean up policy

### Coordination
- `POST /rollout`: Execute coordinated rollout
- `POST /branch`: Create environment/policy branches
- `POST /run/abort`: Abort running rollout
- `GET /run/status/{run_id}`: Check run status

## Local Development

```bash
# Install dependencies
pip install fastapi uvicorn httpx pydantic

# Run the service
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --port 8000
```

## Modal Deployment

```bash
# Deploy to Modal
modal deploy main.py

# Run once
modal run main.py
```

## Environment Variables

- `SERVICE_BASE_URL`: Base URL for this service (default: http://localhost:8000)
- `VLLM_BASE_URL`: Base URL for vLLM inference service (default: http://localhost:8001)
- `DEFAULT_MODEL`: Default model name for inference

## Storage

The service uses Modal Volumes for persistent storage:
- Volume name: `synth-env-state`
- Mount path: `/data/state`
- Layout: `/data/state/runs/{rl_run_id}/{kind}/{shard}/{snapshot_id}.tar.gz`

## Example Usage

```python
import httpx

# Create environment
env_response = httpx.post(
    "http://localhost:8000/env/create",
    json={
        "env_name": "crafter",
        "config": {},
        "seed": 42,
        "rl_run_id": "test-run-1"
    }
)
env_id = env_response.json()["env_id"]

# Create policy
policy_response = httpx.post(
    "http://localhost:8000/policy/create",
    json={
        "policy_name": "crafter-react",
        "config": {"inference_url": "http://vllm:8001"},
        "rl_run_id": "test-run-1",
        "bound_env_id": env_id
    }
)
policy_id = policy_response.json()["policy_id"]

# Execute rollout
rollout_response = httpx.post(
    "http://localhost:8000/rollout",
    json={
        "run_id": "test-run-1",
        "env": {"env_id": env_id},
        "policy": {"policy_id": policy_id},
        "ops": ["agent", "env"] * 10,
        "on_done": "reset"
    }
)
trajectories = rollout_response.json()["trajectories"]
```

## Testing

The implementation follows the plan outlined in `plan.md` and decisions in `decisions.md`. Key test areas:
- Environment create/step/reset lifecycle
- Policy inference request building
- Snapshot/restore round trips
- Rollout coordination with abort support
- Branching operations

4b
"aggregate": {
    "completed": 20,
    "total": 20,
    "avg_turns": 10.0,
    "avg_achievements": 1.3,
    "achievements_freq": {
      "collect_wood": 9,
      "collect_sapling": 8,
      "collect_drink": 7,
      "place_plant": 2
    }
  }


groq qwen/qwen3-32b
 ],
  "aggregate": {
    "completed": 20,
    "total": 20,
    "avg_turns": 10.0,
    "avg_achievements": 1.0,
    "achievements_freq": {
      "collect_sapling": 7,
      "collect_wood": 9,
      "collect_drink": 4
    }
  }
