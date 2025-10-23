from __future__ import annotations

import json
import os
from typing import Any

import click
from synth_ai.cli.lib import (
    ensure_task_app_ready,
    http_request,
    key_preview,
    normalize_endpoint_url,
    popen_stream,
    select_or_create_config,
)
from synth_ai.demos import core as demo_core


def run_job(
    *,
    config: str | None,
    batch_size: int | None,
    group_size: int | None,
    model: str | None,
    timeout: int,
    dry_run: bool,
) -> int:
    demo_dir = demo_core.load_demo_dir()
    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    env = demo_core.load_env()

    synth_key = (env.synth_api_key or "").strip()
    if not synth_key:
        entered = input("Enter SYNTH_API_KEY (required): ").strip()
        if not entered:
            print("SYNTH_API_KEY is required.")
            return 1
        os.environ["SYNTH_API_KEY"] = entered
        demo_core.persist_api_key(entered)
    env = demo_core.load_env()
    synth_key = (env.synth_api_key or "").strip()
    if not synth_key:
        print("SYNTH_API_KEY missing after persist.")
        return 1

    if not env.dev_backend_url:
        print("Backend URL missing. Set DEV_BACKEND_URL or BACKEND_OVERRIDE.")
        return 1

    try:
        env = ensure_task_app_ready(env, synth_key, label="run")
    except RuntimeError as exc:
        print(exc)
        return 1

    os.environ["ENVIRONMENT_API_KEY"] = env.env_api_key

    import tomllib

    try:
        cfg_path = select_or_create_config(config, env)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    launcher = "/Users/joshpurtell/Documents/GitHub/monorepo/tests/applications/math/rl/start_math_clustered.py"
    if os.path.isfile(launcher):
        backend_base = (
            env.dev_backend_url[:-4]
            if env.dev_backend_url.endswith("/api")
            else env.dev_backend_url
        )
        run_env = os.environ.copy()
        run_env["BACKEND_URL"] = backend_base
        run_env["SYNTH_API_KEY"] = env.synth_api_key
        run_env["TASK_APP_BASE_URL"] = env.task_app_base_url
        run_env["ENVIRONMENT_API_KEY"] = env.env_api_key
        run_env["RL_CONFIG_PATH"] = cfg_path
        run_env["TRAINER_START_URL"] = run_env.get("TRAINER_START_URL", "")
        if batch_size is not None:
            run_env["RL_BATCH_SIZE"] = str(int(batch_size))
        if group_size is not None:
            run_env["RL_GROUP_SIZE"] = str(int(group_size))
        if model:
            run_env["RL_MODEL"] = model
        cmd = ["uv", "run", "python", launcher]
        print(f"Launching monorepo clustered runner: {' '.join(cmd)}")
        code = popen_stream(cmd, env=run_env)
        if code != 0:
            print(f"Clustered runner exited with code {code}")
            try:
                base_url = backend_base.rstrip("/") + "/api"
            except Exception:
                base_url = backend_base
            sk = (env.synth_api_key or "").strip()
            ek = (env.env_api_key or "").strip()
            print("Hint: If backend responded 401, verify SYNTH_API_KEY for:", base_url)
            if sk:
                print(f"  {key_preview(sk, 'SYNTH_API_KEY')}")
            if ek:
                print(f"  {key_preview(ek, 'ENVIRONMENT_API_KEY')}")
            print(
                "Ensure the ENVIRONMENT_API_KEY you deployed with matches the task app and remains exported."
            )
        return code

    with open(cfg_path, "rb") as fh:
        inline_cfg = tomllib.load(fh)
    with open(cfg_path) as fh2:
        toml_text = fh2.read()
    if batch_size is not None:
        inline_cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    if group_size is not None:
        inline_cfg.setdefault("training", {})["group_size"] = int(group_size)

    backend_base = env.dev_backend_url.rstrip("/") if env.dev_backend_url else ""

    if env.task_app_base_url:
        services_cfg = inline_cfg.setdefault("services", {})
        services_cfg["task_url"] = env.task_app_base_url
        if isinstance(toml_text, str):
            toml_text = toml_text.replace(
                'task_url = "http://localhost:8101"',
                f'task_url = "{env.task_app_base_url}"',
            )

    if env.dev_backend_url:
        policy_cfg = inline_cfg.setdefault("policy", {})
        policy_cfg["inference_url"] = f"{env.dev_backend_url.rstrip('/')}/inference"
        if isinstance(toml_text, str):
            toml_text = toml_text.replace(
                'inference_url = "http://localhost:8000/api/inference"',
                f'inference_url = "{env.dev_backend_url.rstrip("/")}/inference"',
            )

    model_name = model or (inline_cfg.get("model", {}) or {}).get("name", "Qwen/Qwen3-0.6B")
    api = env.dev_backend_url.rstrip("/") + ("" if env.dev_backend_url.endswith("/api") else "/api")
    try:
        sk = (env.synth_api_key or "").strip()
        print(f"[run] Backend API: {api}")
        print(f"[run] {key_preview(sk, 'SYNTH_API_KEY')}")
    except Exception:
        pass

    endpoint_base_url = normalize_endpoint_url(env.task_app_base_url)
    metadata_block: dict[str, Any] = {
        "source": "synth-ai demo",
        "cwd": os.getcwd(),
        "disable_dev_run": True,
    }
    if backend_base:
        metadata_block["backend_base_url"] = backend_base

    data_fragment: dict[str, Any] = {
        "model": model_name,
        "endpoint_base_url": endpoint_base_url,
        "config": inline_cfg,
        "config_toml": toml_text,
        "config_source": "toml_inline",
        "metadata": {"disable_dev_run": True},
    }
    if env.env_api_key:
        data_fragment["environment_api_key"] = env.env_api_key
    for key in ("training", "evaluation", "rollout", "topology", "vllm"):
        if isinstance(inline_cfg.get(key), dict):
            data_fragment[key] = inline_cfg[key]
    compute: dict[str, Any] = {}
    if isinstance(inline_cfg.get("compute"), dict):
        if inline_cfg["compute"].get("gpu_type"):
            compute["gpu_type"] = str(inline_cfg["compute"]["gpu_type"]).upper()
        if inline_cfg["compute"].get("gpu_count"):
            compute["gpu_count"] = int(inline_cfg["compute"]["gpu_count"])
    if not compute:
        topo = inline_cfg.get("topology") or {}
        gshape = str(topo.get("gpu_type") or "")
        if ":" in gshape:
            t_val, count = gshape.split(":", 1)
            compute = {"gpu_type": t_val.upper(), "gpu_count": int(count)}
    body: dict[str, Any] = {
        "job_type": "rl",
        "data": data_fragment,
    }
    if metadata_block:
        body["metadata"] = metadata_block
    if compute:
        body["compute"] = compute

    if dry_run:
        print(json.dumps(body, indent=2))
        return 0

    code, js = http_request(
        "POST",
        api + "/rl/jobs",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env.synth_api_key}",
        },
        body=body,
    )
    if code not in (200, 201) or not isinstance(js, dict):
        print("Job create failed:", code)
        print(f"Backend: {api}")
        try:
            if isinstance(js, dict):
                print(json.dumps(js, indent=2))
            else:
                print(str(js))
        except Exception:
            print(str(js))
        print("Request body was:\n" + json.dumps(body, indent=2))
        try:
            auth_preview = key_preview(env.synth_api_key or "", "SYNTH_API_KEY (auth)")
            print(f"[run] {auth_preview}")
        except Exception:
            pass
        try:
            data_block = body.get("data") if isinstance(body, dict) else None
            env_key_body = ""
            if isinstance(data_block, dict):
                env_key_body = str(data_block.get("environment_api_key") or "")
            if env_key_body:
                print(f"[run] {key_preview(env_key_body, 'environment_api_key (body)')}")
        except Exception:
            pass
        try:
            current_env_key = env.env_api_key or ""
            if current_env_key:
                print(f"[run] {key_preview(current_env_key, 'ENVIRONMENT_API_KEY (current)')}")
        except Exception:
            pass
        if isinstance(js, dict):
            detail = js.get("detail")
            if isinstance(detail, dict):
                sent_key = detail.get("sent_key")
                if isinstance(sent_key, str):
                    print(f"[run] Backend detail.sent_key {key_preview(sent_key, 'detail.sent_key')}")
                sent_keys = detail.get("sent_keys")
                if isinstance(sent_keys, list | tuple):
                    previews = []
                    for idx, val in enumerate(sent_keys):
                        if isinstance(val, str):
                            previews.append(key_preview(val, f"detail.sent_keys[{idx}]"))
                    if previews:
                        print(f"[run] Backend detail.sent_keys previews: {'; '.join(previews)}")
                key_prefix = detail.get("sent_key_prefix")
                if isinstance(key_prefix, str):
                    print(f"[run] Backend detail.sent_key_prefix={key_prefix}")
                health_url = detail.get("health_url")
                if isinstance(health_url, str):
                    print(f"[run] Backend detail.health_url={health_url}")
        try:
            sk = (env.synth_api_key or "").strip()
            if int(code) == 401 or (
                isinstance(js, dict)
                and any(isinstance(v, str) and "Invalid API key" in v for v in js.values())
            ):
                base_url = env.dev_backend_url
                print(
                    "Hint: HTTP 401 Unauthorized from backend. Verify SYNTH_API_KEY for:", base_url
                )
                if sk:
                    print(f"  {key_preview(sk, 'SYNTH_API_KEY')}")
                print(
                    "Ensure the ENVIRONMENT_API_KEY and OPENAI_API_KEY used for deployment remain valid."
                )
        except Exception:
            pass
        return 2

    job_id = js.get("job_id") or js.get("id") or ""
    if not job_id:
        print("Job id missing in response:", js)
        print("Request body was:\n" + json.dumps(body, indent=2))
        return 2
    print("JOB_ID:", job_id)

    http_request(
        "POST",
        api + f"/rl/jobs/{job_id}/start",
        headers={"Authorization": f"Bearer {env.synth_api_key}"},
    )
    print(
        "Your job is running. Visit https://usesynth.ai to track progress or poll the backend API."
    )
    return 0


def register(group):
    @group.command("run")
    @click.option("--batch-size", type=int, default=None)
    @click.option("--group-size", type=int, default=None)
    @click.option("--model", type=str, default=None)
    @click.option("--timeout", type=int, default=600)
    @click.option("--config", type=str, default=None, help="Path to TOML config (skip prompt)")
    @click.option("--dry-run", is_flag=True, help="Print request body and exit")
    def demo_run(
        batch_size: int | None,
        group_size: int | None,
        model: str | None,
        timeout: int,
        config: str | None,
        dry_run: bool,
    ):
        code = run_job(
            config=config,
            batch_size=batch_size,
            group_size=group_size,
            model=model,
            timeout=timeout,
            dry_run=dry_run,
        )
        if code:
            raise click.exceptions.Exit(code)
