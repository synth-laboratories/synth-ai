from __future__ import annotations

"""Experimental Task App implementation using shared abstractions."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modal import App, Image, Secret, asgi_app

HERE = Path(__file__).resolve()
ROOT = HERE.parent


image = Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.6.0",
    "httpx>=0.24.0",
    "numpy>=1.24.0",
    "aiohttp>=3.8.0",
    "datasets>=2.16.0",
    "synth-ai",
)

app = App("hendrycks-math-task-app-v2")

SECRET_NAME = (
    os.getenv("TASK_APP_SECRET_NAME")
    or os.getenv("MATH_TASK_APP_SECRET")
    or os.getenv("TASK_APP_NAME", "").strip()
)
if not SECRET_NAME:
    SECRET_NAME = "synth-math-demo-secret"
elif not SECRET_NAME.endswith("-secret"):
    SECRET_NAME = f"{SECRET_NAME}-secret"


@app.function(
    image=image,
    timeout=600,
    memory=16384,
    cpu=4,
    min_containers=1,
    secrets=[Secret.from_name(SECRET_NAME)],
)
@asgi_app()
def fastapi_app():
    import json
    import httpx
    from fastapi import Body, Depends, FastAPI, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from synth_ai.task import (
        INTERACT_TOOL_SCHEMA,
        RolloutMetrics,
        RolloutRequest,
        RolloutResponse,
        RolloutStep,
        RolloutTrajectory,
        TaskDatasetRegistry,
        TaskDatasetSpec,
        TaskInfo,
        blend_rubrics,
        get_openai_key_or_503,
        get_groq_key_or_503,
        inject_system_hint,
        is_api_key_header_authorized,
        load_rubric,
        normalize_environment_api_key,
        normalize_vendor_keys,
        extract_message_text,
        parse_tool_call_from_text,
        prepare_for_groq,
        prepare_for_openai,
        require_api_key_dependency,
        score_events_against_rubric,
        score_outcome_against_rubric,
        synthesize_tool_call_if_missing,
        to_jsonable,
        http_exception,
    )

    normalize_environment_api_key()
    normalize_vendor_keys()
    env_key = os.getenv("ENVIRONMENT_API_KEY")
    if not env_key:
        raise RuntimeError("ENVIRONMENT_API_KEY missing in task app environment")

    registry = TaskDatasetRegistry()
    DATASET_SPEC = TaskDatasetSpec(
        id="hendrycks_math",
        name="Hendrycks MATH",
        version="1.0.0",
        splits=["train", "test"],
        default_split="test",
        description="Classic Hendrycks MATH competition problems",
    )

    @lru_cache(maxsize=16)
    def _load_dataset_split(subject: str, split: str):
        from datasets import load_dataset  # type: ignore

        slice_spec = os.getenv("HENDRYCKS_MATH_SLICE")
        dataset_split = split
        if slice_spec:
            dataset_split = f"{split}{slice_spec}"
        return load_dataset("nlile/hendrycks-MATH-benchmark", subject, split=dataset_split)

    def _normalize_text(s: str) -> str:
        import re

        return re.sub(r"[^0-9A-Za-z.+\-/*=]", "", (s or "").strip()).lower()

    def _extract_boxed(text: str) -> str:
        import re

        matches = list(re.finditer(r"\\boxed\{([^}]+)\}", text or ""))
        return matches[-1].group(1) if matches else ""

    def _load_problem(seed: int, subject: str) -> tuple[str, str]:
        dataset = _load_dataset_split(subject, os.getenv("HENDRYCKS_MATH_SPLIT", DATASET_SPEC.default_split or "test"))
        total = len(dataset) if hasattr(dataset, "__len__") else 0
        if total == 0:
            raise RuntimeError("Hendrycks dataset returned empty split")
        index = abs(int(seed)) % total
        example = dataset[int(index)]
        problem = example.get("problem") or example.get("question") or example.get("prompt")
        answer = example.get("solution") or example.get("answer") or ""
        if not problem:
            raise RuntimeError("Dataset item missing problem text")
        return str(problem), str(answer)

    registry.register(DATASET_SPEC, lambda spec: spec)

    BASE_OUTCOME_RUBRIC = load_rubric(
        {
            "version": "1",
            "goal_text": "Evaluate whether the submitted answer matches the Hendrycks solution.",
            "criteria": [
                {
                    "id": "correctness",
                    "description": "Final answer matches the ground-truth solution.",
                    "weight": 1.0,
                    "required": True,
                }
            ],
            "aggregation": "weighted_sum",
        }
    )

    BASE_EVENTS_RUBRIC = load_rubric(
        {
            "version": "1",
            "goal_text": "Evaluate reasoning completeness during math problem solving.",
            "criteria": [
                {
                    "id": "reasoning_detail",
                    "description": "Provides a step-by-step reasoning before submitting final answer.",
                    "weight": 1.0,
                }
            ],
            "aggregation": "weighted_sum",
        }
    )

    def _task_info_base() -> TaskInfo:
        return TaskInfo(
            task={
                "id": "hendrycks_math_single_step",
                "name": "Hendrycks Math Single-Step",
                "version": "0.2.0",
            },
            environments=["math"],
            action_space={
                "type": "text",
                "description": "Submit a textual answer for the presented math problem.",
                "actions": ["submit_answer"],
                "size": 1,
            },
            observation={
                "summary": "Problem statement from the Hendrycks MATH dataset",
                "keys": ["problem"],
            },
            dataset={
                "id": DATASET_SPEC.id,
                "name": DATASET_SPEC.name,
                "version": DATASET_SPEC.version,
                "splits": DATASET_SPEC.splits,
                "default_split": DATASET_SPEC.default_split,
                "cardinality": DATASET_SPEC.cardinality,
            },
            rubric={
                "version": BASE_OUTCOME_RUBRIC.version if BASE_OUTCOME_RUBRIC else "unknown",
                "criteria_count": len(BASE_OUTCOME_RUBRIC.criteria) if BASE_OUTCOME_RUBRIC else 0,
                "source": "inline",
                "aggregation": BASE_OUTCOME_RUBRIC.aggregation if BASE_OUTCOME_RUBRIC else "weighted_sum",
            },
            inference={
                "supports_proxy": True,
                "endpoints": {
                    "openai": "/proxy/v1/chat/completions",
                    "groq": "/proxy/groq/v1/chat/completions",
                },
                "tool": {
                    "name": "interact",
                    "parallel_tool_calls": False,
                },
            },
            capabilities={
                "supports_rollout": True,
                "supports_env_lifecycle": False,
                "requires_api_key_header": True,
            },
            limits={
                "max_ops": 1024,
                "max_time_s": 600,
            },
        )

    def _task_info_for_seed(seed: int, subject: str) -> TaskInfo:
        question, answer = _load_problem(seed, subject)
        base = _task_info_base()
        observation = dict(base.observation)
        observation.update(
            {
                "sample": {
                    "seed": seed,
                    "subject": subject,
                    "problem": question,
                }
            }
        )
        dataset_info = dict(base.dataset)
        dataset_info["subject"] = subject
        dataset_info["seed"] = seed
        rubric_info = dict(base.rubric)
        rubric_info["expected_answer_boxed"] = _extract_boxed(answer) or answer
        return TaskInfo(
            task=base.task,
            environments=base.environments,
            action_space=base.action_space,
            observation=observation,
            dataset=dataset_info,
            rubric=rubric_info,
            inference=base.inference,
            capabilities=base.capabilities,
            limits=base.limits,
        )

    app = FastAPI(title="Hendrycks Math Task App (v2)", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _auth_dependency(request: Request) -> None:
        require_api_key_dependency(request)

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return to_jsonable({"status": "ok", "service": "hendrycks-math-task-app"})

    @app.head("/")
    async def root_head() -> Dict[str, Any]:
        return to_jsonable({"status": "ok"})

    @app.get("/health")
    async def health(request: Request) -> JSONResponse | Dict[str, Any]:
        if not os.getenv("ENVIRONMENT_API_KEY"):
            return JSONResponse(status_code=503, content=to_jsonable({"healthy": False, "detail": "ENVIRONMENT_API_KEY missing"}))
        provided = request.headers.get("X-API-Key") or request.headers.get("X-API-Keys")
        if provided and not is_api_key_header_authorized(request):
            return JSONResponse(status_code=401, content=to_jsonable({"healthy": False, "detail": "Invalid API key"}))
        return to_jsonable({"healthy": True})

    @app.get("/info", dependencies=[Depends(_auth_dependency)])
    async def info() -> Dict[str, Any]:
        dataset = registry.describe(DATASET_SPEC.id)
        return to_jsonable(
            {
                "service": {
                    "task": "hendrycks_math_single_step",
                    "version": "0.2.0",
                },
                "dataset": dataset.model_dump(),
                "inference": {
                    "openai": "/proxy/v1/chat/completions",
                    "groq": "/proxy/groq/v1/chat/completions",
                },
                "rubrics": {
                    "outcome": BASE_OUTCOME_RUBRIC.model_dump() if BASE_OUTCOME_RUBRIC else None,
                    "events": BASE_EVENTS_RUBRIC.model_dump() if BASE_EVENTS_RUBRIC else None,
                },
            }
        )

    @app.get("/task_info", dependencies=[Depends(_auth_dependency)])
    async def task_info(
        subject: str = Query(default=os.getenv("HENDRYCKS_MATH_CONFIG", "algebra")),
        seed: Optional[List[int]] = Query(default=None),
        seeds: Optional[List[int]] = Query(default=None),
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        requested_seeds: List[int] = []
        if seeds:
            requested_seeds.extend(int(s) for s in seeds)
        if seed:
            requested_seeds.extend(int(s) for s in seed)

        if not requested_seeds:
            return to_jsonable(
                {
                    "taskset": {
                        "id": DATASET_SPEC.id,
                        "name": DATASET_SPEC.name,
                        "version": DATASET_SPEC.version,
                        "splits": DATASET_SPEC.splits,
                        "default_split": DATASET_SPEC.default_split,
                        "cardinality": DATASET_SPEC.cardinality,
                        "subjects": subject,
                    }
                }
            )

        infos: List[TaskInfo] = []
        for seed_value in requested_seeds:
            infos.append(_task_info_for_seed(seed_value, subject))
        return [to_jsonable(info.model_dump()) for info in infos]

    async def _call_openai(payload: dict[str, Any], heuristic_hint: Optional[str] = None) -> dict[str, Any]:
        key = get_openai_key_or_503()
        model = payload.get("model") if isinstance(payload.get("model"), str) else None
        prepared = prepare_for_openai(model, payload)
        if heuristic_hint:
            prepared = inject_system_hint(prepared, heuristic_hint)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=prepared,
            )
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
        if response.status_code >= 400:
            raise http_exception(response.status_code, "openai_error", "OpenAI proxy error", extra={"status": response.status_code, "body": data})
        return synthesize_tool_call_if_missing(data)

    async def _call_groq(payload: dict[str, Any], heuristic_hint: Optional[str] = None) -> dict[str, Any]:
        key = get_groq_key_or_503()
        model = payload.get("model") if isinstance(payload.get("model"), str) else None
        prepared = prepare_for_groq(model, payload)
        if heuristic_hint:
            prepared = inject_system_hint(prepared, heuristic_hint)
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=prepared,
            )
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
        if response.status_code >= 400:
            raise http_exception(response.status_code, "groq_error", "Groq proxy error", extra={"status": response.status_code, "body": data})
        return synthesize_tool_call_if_missing(data)

    @app.post("/proxy/v1/chat/completions", dependencies=[Depends(_auth_dependency)])
    async def proxy_openai(body: dict[str, Any] = Body(...)) -> Dict[str, Any]:
        result = await _call_openai(body, heuristic_hint=None)
        return to_jsonable(result)

    @app.post("/proxy/groq/v1/chat/completions", dependencies=[Depends(_auth_dependency)])
    async def proxy_groq(body: dict[str, Any] = Body(...)) -> Dict[str, Any]:
        result = await _call_groq(body, heuristic_hint=None)
        return to_jsonable(result)

    def _compute_reward(expected: str, candidate_actions: Iterable[str], llm_text: Optional[str]) -> float:
        candidate = ""
        for action in reversed(list(candidate_actions)):
            if action.strip():
                candidate = action.strip()
                break
        if not candidate and llm_text:
            candidate = _extract_boxed(llm_text) or llm_text
        if not expected or not candidate:
            return 0.0
        expected_norm = _normalize_text(_extract_boxed(expected) or expected)
        candidate_norm = _normalize_text(candidate)
        if expected_norm and expected_norm == candidate_norm:
            return 1.0
        if expected_norm and expected_norm in candidate_norm:
            return 1.0
        return 0.0

    @app.post("/rollout", dependencies=[Depends(_auth_dependency)])
    async def rollout(request: RolloutRequest) -> Dict[str, Any]:
        env_config = request.env.config or {}
        subject = env_config.get("subject") or os.getenv("HENDRYCKS_MATH_CONFIG", "algebra")
        seed = request.env.seed
        if seed is None:
            seed = env_config.get("seed", 0)
        seed = int(seed)

        problem_text, answer_text = _load_problem(seed, subject)
        policy_config = request.policy.config or {}
        inference_url = (policy_config.get("inference_url") or "").rstrip("/")
        model_name = policy_config.get("model")

        steps: List[RolloutStep] = []
        events: List[Dict[str, Any]] = []
        outcome: Dict[str, Any] = {}
        last_actions: List[str] = []
        last_text: Optional[str] = None

        async def _call_inference(payload: dict[str, Any]) -> dict[str, Any]:
            if not inference_url:
                return await _call_openai(payload)
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), follow_redirects=True) as client:
                response = await client.post(f"{inference_url}/v1/chat/completions", json=payload)
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"raw": response.text}
            if response.status_code >= 400:
                raise http_exception(response.status_code, "inference_error", "Downstream inference call failed", extra={"status": response.status_code, "body": data})
            return synthesize_tool_call_if_missing(data)

        for op in request.ops:
            if isinstance(op, str):
                op_type = op
                payload = {}
            elif isinstance(op, dict):
                op_type = op.get("type") or op.get("op")
                payload = op
            else:
                continue

            if op_type == "policy":
                llm_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                if not llm_payload:
                    llm_payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Solve the given Hendrycks MATH problem. Provide reasoning before calling the interact tool to submit the final answer.",
                            },
                            {
                                "role": "user",
                                "content": problem_text,
                            },
                        ],
                        "tools": INTERACT_TOOL_SCHEMA,
                    }
                response = await _call_inference(prepare_for_openai(model_name, llm_payload))
                message = None
                actions: List[str] = []
                reasoning_text = ""
                try:
                    choices = response.get("choices") if isinstance(response, dict) else None
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message")
                        if isinstance(message, dict):
                            tool_calls = message.get("tool_calls")
                            if isinstance(tool_calls, list) and tool_calls:
                                fn = tool_calls[0].get("function") or {}
                                args = fn.get("arguments")
                                if isinstance(args, str):
                                    parsed = json.loads(args)
                                elif isinstance(args, dict):
                                    parsed = args
                                else:
                                    parsed = {}
                                if isinstance(parsed, dict):
                                    actions_field = parsed.get("actions")
                                    if isinstance(actions_field, list):
                                        actions = [str(a) for a in actions_field]
                                    elif isinstance(actions_field, str):
                                        actions = [a.strip() for a in actions_field.split(",") if a.strip()]
                                    reasoning_text = parsed.get("reasoning", "")
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                last_text = content
                        else:
                            message = None
                    if not actions and message is not None:
                        message_text = extract_message_text(message)
                        parsed_actions, parsed_reasoning = parse_tool_call_from_text(message_text)
                        actions = parsed_actions or actions
                        if parsed_reasoning:
                            reasoning_text = parsed_reasoning
                except Exception:
                    pass

                last_actions = actions
                events.append({"id": "reasoning_detail", "score": 1.0 if reasoning_text else 0.0})
                steps.append(
                    RolloutStep(
                        obs={"problem": problem_text},
                        tool_calls=[
                            {
                                "tool_name": "interact",
                                "arguments": json.dumps({"actions": actions, "reasoning": reasoning_text}),
                            }
                        ],
                        reward=None,
                        done=False,
                        truncated=False,
                        info={"events": [{"actions": actions, "reasoning": reasoning_text}]},
                    )
            elif op_type == "env":
                reward = _compute_reward(answer_text, last_actions, last_text)
                outcome = {"criteria": {"correctness": reward}}
                steps.append(
                    RolloutStep(
                        obs={"problem": problem_text},
                        tool_calls=[],
                        reward=reward,
                        done=True,
                        truncated=False,
                        info={
                            "events": [],
                            "achievements": ["correct_answer"] if reward > 0 else [],
                        },
                    )
                )
                break

        outcome_rubric = blend_rubrics(BASE_OUTCOME_RUBRIC, None)
        events_rubric = blend_rubrics(BASE_EVENTS_RUBRIC, None)

        outcome_score = score_outcome_against_rubric(outcome, outcome_rubric)
        events_score = score_events_against_rubric(events, events_rubric)

        metrics = RolloutMetrics(
            episode_returns=[steps[-1].reward or 0.0] if steps else [0.0],
            mean_return=float(steps[-1].reward or 0.0) if steps else 0.0,
            num_steps=len(steps),
            num_episodes=1,
            outcome_score=outcome_score.get("score"),
            events_score=events_score.get("score"),
            details={
                "outcome": outcome_score,
                "events": events_score,
            },
        )

        trajectory = RolloutTrajectory(
            env_id=str(request.env.env_id or request.env.env_name or "math"),
            policy_id=str(request.policy.policy_id or request.policy.policy_name or "math-react"),
            steps=steps,
            final={"outcome": outcome},
            length=len(steps),
        )

        response = RolloutResponse(
            run_id=request.run_id,
            trajectories=[trajectory],
            metrics=metrics,
            aborted=False,
            ops_executed=len(steps),
        )
        return to_jsonable(response.model_dump())

    return app
