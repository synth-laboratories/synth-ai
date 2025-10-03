"""Modal app for Crafter task service with OpenAI proxy.

App name: grpo-crafter-task-app_warming_up_ex

Provides:
- Hosted env/policy/rollout endpoints from synth_envs_hosted.create_app (Crafter only)
- GET /health (inherited)
- POST /proxy/v1/chat/completions (for direct OpenAI usage)

Secrets expected in Modal secret bundle:
- ENVIRONMENT_API_KEY (required) or dev_environment_api_key fallback
- OPENAI_API_KEY (required for proxy endpoint)
- SYNTH_API_KEY (optional; for backend-mediated flows)

To run locally for testing:
    python grpo_crafter_task_app.py --local
"""

from __future__ import annotations

from modal import App, Image, asgi_app, Secret
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parent.parent.parent.resolve()
# Use local copy of synth_envs_hosted within this task_app folder
TASK_SRC = (BASE_DIR / "./synth_envs_hosted").resolve()

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core server
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        # Crafter deps come via synth-ai; keep minimal here
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        # Hosted env/policy/rollout
        "synth-ai==0.2.4.dev6",
        # Proxy deps
        "httpx>=0.24.0",
    )
    .add_local_dir(str(TASK_SRC), "/app/synth_envs_hosted")
    .add_local_dir(str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai")
    .add_local_dir(str(BASE_DIR), "/app/grpo_crafter")
)


app = App("grpo-crafter-task-app-final_warming_up_ex")


@app.function(
    image=image,
    timeout=600,
    memory=16384,
    cpu=4,
    min_containers=1,
    max_containers=10,
    secrets=[
        Secret.from_name("crafter-environment-sdk"),
        Secret.from_name("groq-api-key"),
        Secret.from_name("openai-api-key"),
    ],
)
@asgi_app()
def fastapi_app():
    import os
    import sys

    # Ensure packaged modules resolve when running under Modal before any imports that rely on them
    sys.path.insert(0, "/opt/synth_ai_repo")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/grpo_crafter")
    sys.path.insert(0, "/app/synth_envs_hosted")
    sys.path.insert(0, str(BASE_DIR))

    import json
    from typing import Any, Dict, List, Optional

    import crafter
    import crafter.constants as C
    import httpx
    from fastapi import Body, Depends, FastAPI, Header, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from synth_envs_hosted.hosted_app import create_app as _create_app

    from synth_ai.task import (
        INTERACT_TOOL_SCHEMA,
        TaskDatasetRegistry,
        TaskDatasetSpec,
        TaskInfo,
        extract_message_text,
        get_openai_key_or_503,
        get_groq_key_or_503,
        inject_system_hint,
        is_api_key_header_authorized,
        load_rubric,
        normalize_environment_api_key,
        normalize_vendor_keys,
        parse_tool_call_from_text,
        prepare_for_groq,
        prepare_for_openai,
        require_api_key_dependency,
        synthesize_tool_call_if_missing,
        to_jsonable,
    )
    from synth_ai.task.errors import http_exception
    from synth_ai.environments.examples.crafter_classic.taskset import TRAIT_BOUNDS, world_traits

    normalize_environment_api_key()
    normalize_vendor_keys()

    if not os.getenv("ENVIRONMENT_API_KEY"):
        raise RuntimeError("ENVIRONMENT_API_KEY missing in task app environment")

    api = _create_app(allowed_environments=["crafter"])

    # Remove legacy /health and /info to replace with standardized endpoints
    api.router.routes = [
        route
        for route in api.router.routes
        if getattr(route, "path", None) not in {"/health", "/info", "/rollout"}
    ]

    # Dataset configuration — procedural seeds controlling Crafter world generation
    DATASET_SPEC = TaskDatasetSpec(
        id="crafter_classic_procedural",
        name="Crafter Classic Procedural Seeds",
        version="1.0.0",
        splits=["train"],
        default_split="train",
        description="Procedural Crafter Classic seeds with reproducible world traits.",
    )

    class CrafterDataset:
        def __init__(self, spec: TaskDatasetSpec) -> None:
            self.spec = spec
            self.default_seed = int(os.getenv("CRAFTER_DEFAULT_SEED", "42"))
            self.seed_min = 0
            self.seed_max = int(os.getenv("CRAFTER_MAX_SEED", str(2**31 - 1)))
            self.area = tuple(int(x) for x in os.getenv("CRAFTER_AREA", "64,64").split(","))
            self.length = int(os.getenv("CRAFTER_EPISODE_LENGTH", "10000"))
            self._cache: Dict[int, Dict[str, Any]] = {}

        def config_for_seed(self, seed: int) -> Dict[str, Any]:
            return {
                "seed": int(seed),
                "area": list(self.area),
                "length": self.length,
            }

        def describe_seed(self, seed: int) -> Dict[str, Any]:
            seed = int(seed)
            if seed in self._cache:
                return self._cache[seed]
            env = crafter.Env(area=self.area, length=self.length, seed=seed)
            env.reset()
            traits = world_traits(env)
            player = getattr(env, "_player", None)
            inventory = dict(getattr(player, "inventory", {})) if player else {}
            position = getattr(player, "pos", None)
            env.close()
            summary = {
                "seed": seed,
                "difficulty": self._difficulty(traits),
                "traits": traits,
                "inventory": inventory,
                "player_position": list(position) if position is not None else None,
                "config": self.config_for_seed(seed),
            }
            self._cache[seed] = summary
            return summary

        def _difficulty(self, traits: Dict[str, int]) -> str:
            for difficulty, bounds in TRAIT_BOUNDS.items():
                if (
                    traits.get("trees", 0) >= bounds.get("min_trees", 0)
                    and traits.get("hostiles", 0) <= bounds.get("max_hostiles", 0)
                ):
                    return difficulty
            return "custom"

        @property
        def seed_range(self) -> List[int]:
            return [self.seed_min, self.seed_max]

    dataset_registry = TaskDatasetRegistry()
    crafter_dataset = CrafterDataset(DATASET_SPEC)
    dataset_registry.register(DATASET_SPEC, lambda spec: crafter_dataset, cache=True)

    OUTCOME_RUBRIC = load_rubric(
        {
            "version": "1",
            "goal_text": "Reward unlocking Crafter achievements and survival.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "achievements",
                    "description": "Unlock achievements or crafting milestones.",
                    "weight": 1.0,
                },
                {
                    "id": "survival",
                    "description": "Maintain health, food, and drink levels.",
                    "weight": 1.0,
                },
            ],
        }
    )
    EVENTS_RUBRIC = load_rubric(
        {
            "version": "1",
            "goal_text": "Encourage purposeful step-wise exploration and crafting.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "progress_steps",
                    "description": "Actions progress quests, crafting, or exploration.",
                    "weight": 1.0,
                }
            ],
        }
    )

    def _base_task_info() -> TaskInfo:
        return TaskInfo(
            task={"id": "crafter_classic", "name": "Crafter Classic", "version": "1.0.0"},
            environments=["crafter"],
            action_space={
                "type": "discrete",
                "size": len(C.actions),
                "actions": list(C.actions),
            },
            observation={
                "summary": "RGB frame plus inventory, achievements, and semantic map patches.",
                "keys": ["image", "inventory", "achievements", "semantic_map_patch7"],
                "image_shape": [64, 64, 3],
            },
            dataset={
                **DATASET_SPEC.model_dump(),
                "seed_range": crafter_dataset.seed_range,
                "default_seed": crafter_dataset.default_seed,
            },
            rubric={
                "version": OUTCOME_RUBRIC.version if OUTCOME_RUBRIC else "unknown",
                "criteria_count": len(OUTCOME_RUBRIC.criteria) if OUTCOME_RUBRIC else 0,
                "source": "inline",
                "aggregation": OUTCOME_RUBRIC.aggregation if OUTCOME_RUBRIC else "weighted_sum",
            },
            inference={
                "supports_proxy": True,
                "endpoints": {
                    "openai": "/proxy/v1/chat/completions",
                    "groq": "/proxy/groq/v1/chat/completions",
                },
                "tool": {"name": "interact", "parallel_tool_calls": False},
            },
            capabilities={
                "supports_rollout": True,
                "supports_env_lifecycle": True,
                "requires_api_key_header": True,
            },
            limits={"max_ops": 100000, "max_time_s": 3600},
        )

    BASE_TASK_INFO = _base_task_info()

    def _auth_dependency(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        x_api_keys: Optional[str] = Header(default=None, alias="X-API-Keys"),
    ) -> None:
        expected = normalize_environment_api_key()
        if not expected:
            raise http_exception(503, "missing_environment_api_key", "ENVIRONMENT_API_KEY is not configured")
        provided: List[str] = []
        if x_api_key:
            provided.append(x_api_key)
        if x_api_keys:
            provided.extend([part.strip() for part in x_api_keys.split(",") if part.strip()])
        if provided and expected not in provided:
            raise http_exception(401, "unauthorized", "Invalid API key")

    @api.get("/")
    async def root() -> Dict[str, Any]:
        return to_jsonable({"status": "ok", "service": "crafter-task-app"})

    @api.head("/")
    async def root_head() -> Dict[str, Any]:
        return to_jsonable({"status": "ok"})

    @api.get("/health")
    async def health(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        x_api_keys: Optional[str] = Header(default=None, alias="X-API-Keys"),
    ) -> Dict[str, Any]:
        expected = normalize_environment_api_key()
        if not expected:
            raise http_exception(503, "missing_environment_api_key", "ENVIRONMENT_API_KEY is not configured")

        provided: List[str] = []
        if x_api_key:
            provided.append(x_api_key)
        if x_api_keys:
            provided.extend([part.strip() for part in x_api_keys.split(",") if part.strip()])

        if provided and expected not in provided:
            raise http_exception(401, "unauthorized", "Invalid API key")

        return to_jsonable({"healthy": True})

    @api.get("/info", dependencies=[Depends(_auth_dependency)])
    async def info() -> Dict[str, Any]:
        dataset_meta = {
            **DATASET_SPEC.model_dump(),
            "seed_range": crafter_dataset.seed_range,
            "default_seed": crafter_dataset.default_seed,
        }
        return to_jsonable(
            {
                "service": {"task": BASE_TASK_INFO.task, "version": BASE_TASK_INFO.task.get("version")},
                "dataset": dataset_meta,
                "rubrics": {
                    "outcome": OUTCOME_RUBRIC.model_dump() if OUTCOME_RUBRIC else None,
                    "events": EVENTS_RUBRIC.model_dump() if EVENTS_RUBRIC else None,
                },
                "inference": BASE_TASK_INFO.inference,
            }
        )

    @api.get("/task_info", dependencies=[Depends(_auth_dependency)])
    async def task_info(
        request: Request,
        seed: Optional[List[int]] = Query(default=None),
        seeds: Optional[List[int]] = Query(default=None),
    ) -> Any:
        all_seeds: List[int] = []
        if seed:
            all_seeds.extend(int(s) for s in seed)
        if seeds:
            all_seeds.extend(int(s) for s in seeds)

        if not all_seeds:
            descriptor = {
                **DATASET_SPEC.model_dump(),
                "seed_range": crafter_dataset.seed_range,
                "default_seed": crafter_dataset.default_seed,
                "config": {
                    "area": list(crafter_dataset.area),
                    "length": crafter_dataset.length,
                },
            }
            return to_jsonable({"taskset": descriptor})

        infos: List[TaskInfo] = []
        for seed_value in all_seeds:
            summary = crafter_dataset.describe_seed(seed_value)
            infos.append(
                TaskInfo(
                    task=BASE_TASK_INFO.task,
                    environments=BASE_TASK_INFO.environments,
                    action_space=BASE_TASK_INFO.action_space,
                    observation={
                        **BASE_TASK_INFO.observation,
                        "seed": seed_value,
                        "traits": summary["traits"],
                        "inventory": summary["inventory"],
                        "player_position": summary["player_position"],
                    },
                    dataset={
                        **BASE_TASK_INFO.dataset,
                        "seed": seed_value,
                        "difficulty": summary["difficulty"],
                        "config": summary["config"],
                    },
                    rubric=BASE_TASK_INFO.rubric,
                    inference=BASE_TASK_INFO.inference,
                    capabilities=BASE_TASK_INFO.capabilities,
                    limits=BASE_TASK_INFO.limits,
                )
            )

        payload = [to_jsonable(info.model_dump()) for info in infos]
        return payload if len(payload) > 1 else payload[0]

    def _normalise_op(op_value: Any, index: int) -> str:
        if isinstance(op_value, str):
            candidate = op_value
        elif isinstance(op_value, dict):
            candidate = op_value.get("type") or op_value.get("op")
        else:
            candidate = None
        if not candidate:
            raise http_exception(400, "invalid_op", f"Missing op type at index {index}")
        lowered = str(candidate).strip().lower()
        if lowered in {"policy", "agent", "model"}:
            return "agent"
        if lowered in {"env", "environment", "step"}:
            return "env"
        raise http_exception(400, "invalid_op", f"Unsupported op type '{candidate}' at index {index}")

    @api.post("/rollout", dependencies=[Depends(_auth_dependency)])
    async def rollout_endpoint(rollout_request: RolloutRequest, request: Request) -> Dict[str, Any]:
        from synth_envs_hosted.rollout import (
            RolloutRequest as LegacyRolloutRequest,
            RolloutEnvSpec as LegacyRolloutEnvSpec,
            RolloutPolicySpec as LegacyRolloutPolicySpec,
            RolloutRecordConfig as LegacyRolloutRecordConfig,
            RolloutSafetyConfig as LegacyRolloutSafetyConfig,
            execute_rollout as legacy_execute_rollout,
        )

        converted_ops: List[str] = []
        for idx, op in enumerate(rollout_request.ops):
            converted_ops.append(_normalise_op(op, idx))

        legacy_request = LegacyRolloutRequest(
            run_id=rollout_request.run_id,
            env=LegacyRolloutEnvSpec(
                env_id=rollout_request.env.env_id,
                env_name=rollout_request.env.env_name,
                config=rollout_request.env.config or {},
                seed=rollout_request.env.seed,
            ),
            policy=LegacyRolloutPolicySpec(
                policy_id=rollout_request.policy.policy_id,
                policy_name=rollout_request.policy.policy_name,
                config=rollout_request.policy.config or {},
            ),
            ops=converted_ops,
            record=LegacyRolloutRecordConfig(**rollout_request.record.model_dump()),
            on_done=rollout_request.on_done,
            branch=None,
            safety=LegacyRolloutSafetyConfig(**rollout_request.safety.model_dump()),
            training_session_id=rollout_request.training_session_id,
            synth_base_url=rollout_request.synth_base_url,
        )

        legacy_response = await legacy_execute_rollout(legacy_request, request)
        data = legacy_response.model_dump()
        metrics = data.get("metrics", {}) or {}
        metrics.setdefault("outcome_score", None)
        metrics.setdefault("events_score", None)
        metrics.setdefault("details", {})
        data["metrics"] = metrics
        shared_response = RolloutResponse.model_validate(data)
        return to_jsonable(shared_response.model_dump())

    CRAFTING_RULES_SYSTEM_HINT = (
        "Crafter crafting rules (from the paper):\n"
        "- Make Wood Pickaxe: Nearby a table; have wood in inventory.\n"
        "- Make Stone Pickaxe: Nearby a table; have wood and stone in inventory.\n"
        "- Make Iron Pickaxe: Nearby a table; furnace exists; have wood, coal, and iron in inventory.\n"
        "- Make Wood Sword: Nearby a table; have wood in inventory.\n"
        "- Make Stone Sword: Nearby a table; have wood and stone in inventory.\n"
        "- Make Iron Sword: Nearby a table; furnace exists; have wood, coal, and iron in inventory."
    )

    async def _call_vendor(url: str, payload: dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
            return await client.post(url, json=payload, headers=headers)

    def _log_proxy_request(route: str, payload: dict[str, Any]) -> None:
        try:
            messages = payload.get("messages") if isinstance(payload, dict) else None
            msg_count = len(messages) if isinstance(messages, list) else 0
            tool_count = len(payload.get("tools") or []) if isinstance(payload, dict) else 0
            model = payload.get("model") if isinstance(payload, dict) else None
            print(f"[proxy:{route}] model={model} messages={msg_count} tools={tool_count}", flush=True)
        except Exception:
            pass

    @api.post("/proxy/v1/chat/completions", dependencies=[Depends(_auth_dependency)])
    async def proxy_openai(body: dict[str, Any] = Body(...)) -> Dict[str, Any]:
        key = get_openai_key_or_503()
        model = body.get("model") if isinstance(body.get("model"), str) else None
        prepared = prepare_for_openai(model, body)
        prepared = inject_system_hint(prepared, CRAFTING_RULES_SYSTEM_HINT)
        _log_proxy_request("openai", prepared)
        response = await _call_vendor(
            "https://api.openai.com/v1/chat/completions",
            prepared,
            {"Authorization": f"Bearer {key}"},
        )
        data = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {"raw": response.text}
        )
        if response.status_code >= 400:
            raise http_exception(
                response.status_code,
                "openai_error",
                "OpenAI proxy error",
                extra={"status": response.status_code, "body": data},
            )
        sanitized = synthesize_tool_call_if_missing(data)
        return to_jsonable(sanitized)

    @api.post("/proxy/groq/v1/chat/completions", dependencies=[Depends(_auth_dependency)])
    async def proxy_groq(body: dict[str, Any] = Body(...)) -> Dict[str, Any]:
        key = get_groq_key_or_503()
        model = body.get("model") if isinstance(body.get("model"), str) else None
        prepared = prepare_for_groq(model, body)
        prepared = inject_system_hint(prepared, CRAFTING_RULES_SYSTEM_HINT)
        _log_proxy_request("groq", prepared)
        response = await _call_vendor(
            os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions").rstrip("/"),
            prepared,
            {"Authorization": f"Bearer {key}"},
        )
        data = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {"raw": response.text}
        )
        if response.status_code >= 400:
            raise http_exception(
                response.status_code,
                "groq_error",
                "Groq proxy error",
                extra={"status": response.status_code, "body": data},
            )
        sanitized = synthesize_tool_call_if_missing(data)
        return to_jsonable(sanitized)

    @api.get("/debug/env", dependencies=[Depends(_auth_dependency)])
    async def debug_env() -> Dict[str, Any]:
        def _mask(value: Optional[str]) -> str:
            if not value:
                return ""
            return f"{value[:6]}…" if len(value) > 6 else value

        return to_jsonable(
            {
                "has_ENVIRONMENT_API_KEY": bool(os.getenv("ENVIRONMENT_API_KEY")),
                "OPENAI_API_KEY_prefix": _mask(os.getenv("OPENAI_API_KEY")),
                "GROQ_API_KEY_prefix": _mask(os.getenv("GROQ_API_KEY")),
            }
        )

    return api


# Local development mode
if __name__ == "__main__":
    import argparse
    import uvicorn
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally for development")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()

    if args.local:
        # Set up environment for local development
        if not os.getenv("ENVIRONMENT_API_KEY"):
            # Try to load from backend .env.dev
            env_dev_path = Path(__file__).parent.parent.parent.parent.parent / "backend" / ".env.dev"
            if env_dev_path.exists():
                try:
                    import dotenv
                    dotenv.load_dotenv(env_dev_path)
                    print(f"Loaded environment from {env_dev_path}")
                except ImportError:
                    print("dotenv not available, using existing environment")
            else:
                print(f"Warning: {env_dev_path} not found")

        # Ensure required environment variables
        required_vars = ["ENVIRONMENT_API_KEY", "OPENAI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"Missing required environment variables: {missing}")
            sys.exit(1)

        print(f"Starting task app locally on {args.host}:{args.port}")
        print("Rollout endpoint: http://localhost:8001/rollout")
        print("Health endpoint: http://localhost:8001/health")

        # Create and run the app
        app = fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port, reload=True)
    else:
        print("Use --local to run locally, or deploy to Modal")
