from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

HOST_WORKSPACE_ROOT = Path("/Users/joshpurtell/Documents/GitHub")
HOST_PRIVATE_REPO_ROOT = HOST_WORKSPACE_ROOT / "synth-cookbooks-private"
HOST_PUBLIC_COOKBOOKS_ROOT = HOST_WORKSPACE_ROOT / "synth-cookbooks-public"
TASK_ID = "runbench/hillclimbsymbolicbench/symbolic_policy_hillclimb"
BENCHMARK_FAMILY = "runbench.hillclimbsymbolicbench"
HELDOUT_ENV = "CODE_POLICY_HELDOUT_SEEDS"
TASK_ROOT = Path(__file__).resolve().parents[1]
TASK_TOML_PATH = TASK_ROOT / "task.toml"
RUBRIC_JSON_PATH = TASK_ROOT / "RUBRIC.json"
VERIFIER_RUBRIC_PATH = TASK_ROOT / "VERIFIER_RUBRIC.md"


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    policy_path: Path
    sweep_script: Path
    service_app: Path | None
    service_url: str | None
    train_seeds: str
    max_steps: int
    max_workers: int
    direct: bool = False


def _runtime_workspace_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in (os.environ.get("REPORTBENCH_OUTPUT_ROOT"), "/workspace", str(Path.cwd())):
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return tuple(roots)


def _container_root(env_id: str) -> Path:
    for workspace_root in _runtime_workspace_roots():
        staged = workspace_root / "containers" / env_id
        if staged.exists():
            return staged
    if env_id == "dungeongrid":
        for workspace_root in _runtime_workspace_roots():
            staged = workspace_root / "containers" / "dungeongrid_plus"
            if staged.exists():
                return staged
        return HOST_PRIVATE_REPO_ROOT / "containers/dungeongrid_plus"
    if env_id == "montezuma":
        for workspace_root in _runtime_workspace_roots():
            staged = workspace_root / "containers" / "montezuma_revenge"
            if staged.exists():
                return staged
        return HOST_PRIVATE_REPO_ROOT / "containers/montezuma_revenge"
    if env_id == "pitfall":
        return HOST_PRIVATE_REPO_ROOT / "containers/pitfall"
    return HOST_PRIVATE_REPO_ROOT / "containers" / env_id


def _public_containers_src() -> Path | None:
    for workspace_root in _runtime_workspace_roots():
        staged = workspace_root / "third_party/synth-containers/src"
        if staged.exists():
            return staged
    host = HOST_PUBLIC_COOKBOOKS_ROOT / "packages/synth-containers/src"
    return host if host.exists() else None


def _env_spec(env_id: str) -> EnvSpec:
    root = _container_root(env_id)
    if env_id == "nle":
        return EnvSpec(
            env_id="nle",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=root / "synth_service_app.py",
            service_url="http://127.0.0.1:8914",
            train_seeds="101,103,107,109",
            max_steps=420,
            max_workers=4,
        )
    if env_id == "craftax":
        return EnvSpec(
            env_id="craftax",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=root / "synth_service_app.py",
            service_url="http://127.0.0.1:8931",
            train_seeds="101,103,105,107",
            max_steps=120,
            max_workers=4,
        )
    if env_id == "crafter":
        return EnvSpec(
            env_id="crafter",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=None,
            service_url=None,
            train_seeds="101,103,105,107,109,111,113,127,131,137,139,149,151,157,163,167,173,179,181,191",
            max_steps=300,
            max_workers=1,
            direct=True,
        )
    if env_id == "dungeongrid":
        return EnvSpec(
            env_id="dungeongrid",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=root / "synth_service_app.py",
            service_url=None,
            train_seeds="101,103",
            max_steps=360,
            max_workers=4,
            direct=True,
        )
    if env_id == "minigrid":
        return EnvSpec(
            env_id="minigrid",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=None,
            service_url=None,
            train_seeds="101,103,105,107",
            max_steps=200,
            max_workers=4,
            direct=True,
        )
    if env_id == "montezuma":
        return EnvSpec(
            env_id="montezuma",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=None,
            service_url=None,
            train_seeds="101,103,107,109",
            max_steps=4500,
            max_workers=4,
            direct=True,
        )
    if env_id == "pitfall":
        return EnvSpec(
            env_id="pitfall",
            policy_path=root / "heuristic_policy.py",
            sweep_script=root / "run_heuristic_sweep.py",
            service_app=None,
            service_url=None,
            train_seeds="101,103,107,109",
            max_steps=6000,
            max_workers=4,
            direct=True,
        )
    raise RuntimeError(
        f"unknown env {env_id!r}; expected nle, craftax, crafter, dungeongrid, minigrid, montezuma, or pitfall"
    )


ENV_SPECS: dict[str, EnvSpec] = {
    "nle": _env_spec("nle"),
    "craftax": _env_spec("craftax"),
    "crafter": _env_spec("crafter"),
    "dungeongrid": _env_spec("dungeongrid"),
    "minigrid": _env_spec("minigrid"),
    "montezuma": _env_spec("montezuma"),
    "pitfall": _env_spec("pitfall"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seed_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def _seed_hash(seeds: list[int]) -> str:
    encoded = ",".join(str(seed) for seed in seeds).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _service_reachable(url: str, *, timeout_s: float = 0.5) -> bool:
    if not url.startswith("http://"):
        return False
    host_port = url.removeprefix("http://").split("/", 1)[0]
    host, _, port_text = host_port.partition(":")
    try:
        port = int(port_text)
    except ValueError:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _http_health_ok(url: str, *, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout_s) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _service_env(spec: EnvSpec, port: int) -> dict[str, str]:
    env = dict(os.environ)
    staged_nanohorizon_src = _staged_nanohorizon_src()
    if not env.get("NANOHORIZON_SRC"):
        if staged_nanohorizon_src is not None:
            env["NANOHORIZON_SRC"] = str(staged_nanohorizon_src)
        else:
            nanohorizon_root = env.get("NANOHORIZON_REPO_ROOT")
            if nanohorizon_root:
                candidate = Path(nanohorizon_root).expanduser() / "src"
                if candidate.exists():
                    env["NANOHORIZON_SRC"] = str(candidate)
    pythonpath_parts = [
        str(spec.policy_path.parent),
    ]
    pythonpath_parts.extend(str(path) for path in _runtime_workspace_roots())
    public_src = _public_containers_src()
    if public_src is not None:
        pythonpath_parts.insert(0, str(public_src))
    if staged_nanohorizon_src is not None:
        pythonpath_parts.insert(0, str(staged_nanohorizon_src))
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PORT"] = str(port)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    return env


def _staged_nanohorizon_src() -> Path | None:
    for workspace_root in _runtime_workspace_roots():
        candidate = workspace_root / "nanohorizon" / "src"
        if (candidate / "nanohorizon" / "craftax_core" / "rollout.py").exists():
            return candidate
    return None


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None:
        return None
    return str(spec.origin or "")


def _service_command(spec: EnvSpec) -> list[str]:
    if spec.env_id == "craftax":
        if _staged_nanohorizon_src() is not None or _module_origin("craftax"):
            return [sys.executable, str(spec.service_app)]
        nanohorizon_root = Path(
            os.environ.get("NANOHORIZON_REPO_ROOT") or HOST_WORKSPACE_ROOT / "nanohorizon"
        ).expanduser()
        uv_bin = shutil.which("uv")
        if uv_bin and (nanohorizon_root / "pyproject.toml").exists():
            return [
                uv_bin,
                "run",
                "--project",
                str(nanohorizon_root),
                "--group",
                "classic",
                "python",
                str(spec.service_app),
            ]
    return [sys.executable, str(spec.service_app)]


def _python_probe_command(command: list[str]) -> list[str] | None:
    if not command:
        return None
    if Path(command[0]).name.startswith("python"):
        return [command[0]]
    for index, part in enumerate(command):
        if part == "python" or Path(part).name.startswith("python"):
            return [*command[: index + 1]]
    return None


def _service_dependency_probe(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    prefix = _python_probe_command(command)
    if prefix is None:
        return {"ok": False, "error": "python_probe_prefix_unresolved", "cmd": command}
    code = (
        "import importlib.util, json, sys; "
        "mods=['craftax','jax','fastapi','uvicorn','nanohorizon','nanohorizon.craftax_core.rollout']; "
        "out={'python':sys.executable,'modules':{}}; "
        "missing=[]; "
        "\nfor name in mods:\n"
        "    spec=importlib.util.find_spec(name)\n"
        "    out['modules'][name]=None if spec is None else (spec.origin or '')\n"
        "    if spec is None: missing.append(name)\n"
        "print(json.dumps(out, sort_keys=True))\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    started = time.time()
    try:
        result = subprocess.run(
            [*prefix, "-c", code],
            text=True,
            capture_output=True,
            timeout=120,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "cmd": [*prefix, "-c", "<dependency-probe>"],
            "elapsed_s": round(time.time() - started, 2),
            "error": "dependency_probe_timeout",
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": str(exc.stderr or "")[-4000:],
        }
    return {
        "ok": result.returncode == 0,
        "cmd": [*prefix, "-c", "<dependency-probe>"],
        "returncode": result.returncode,
        "elapsed_s": round(time.time() - started, 2),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _start_task_owned_service(
    *,
    spec: EnvSpec,
    output_root: Path,
    timeout_s: float = 30.0,
) -> tuple[str | None, subprocess.Popen[str] | None, dict[str, Any]]:
    receipt: dict[str, Any] = {
        "mode": "task_owned",
        "env": spec.env_id,
        "started": False,
        "ready": False,
        "service_app": str(spec.service_app) if spec.service_app else None,
        "log_path": None,
        "pid": None,
        "url": None,
    }
    if spec.service_app is None or not spec.service_app.exists():
        receipt["error"] = "service_app_missing"
        return None, None, receipt
    if spec.env_id == "craftax":
        timeout_s = max(timeout_s, 180.0)
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    logs_dir = output_root / "service_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{spec.env_id}_service.log"
    command = _service_command(spec)
    service_env = _service_env(spec, port)
    probe = _service_dependency_probe(command, service_env)
    receipt.update(
        {
            "cmd": command,
            "dependency_probe": probe,
            "python_executable": sys.executable,
            "pythonpath": service_env.get("PYTHONPATH"),
            "nanohorizon_src": service_env.get("NANOHORIZON_SRC"),
            "module_origins": {
                "craftax": _module_origin("craftax"),
                "jax": _module_origin("jax"),
                "fastapi": _module_origin("fastapi"),
                "uvicorn": _module_origin("uvicorn"),
                "nanohorizon": _module_origin("nanohorizon"),
            },
        }
    )
    if not probe.get("ok"):
        receipt["error"] = "service_dependency_probe_failed"
        receipt["log_path"] = str(log_path)
        _write_json(output_root / "artifacts" / "service_startup_failure.json", receipt)
        return url, None, receipt
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=spec.policy_path.parent,
        env=service_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    receipt.update(
        {
            "started": True,
            "cmd": command,
            "pid": process.pid,
            "url": url,
            "port": port,
            "log_path": str(log_path),
        }
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            receipt["error"] = f"service_exited_{process.returncode}"
            receipt["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            return url, process, receipt
        if _http_health_ok(url) or _service_reachable(url):
            receipt["ready"] = True
            return url, process, receipt
        time.sleep(0.25)
    receipt["error"] = "readiness_timeout"
    receipt["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return url, process, receipt


def _stop_task_owned_services(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        process.terminate()
    deadline = time.time() + 5.0
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def _compile_policy(policy_path: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "py_compile", str(policy_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    return {
        "cmd": command,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_command(command: list[str], *, cwd: Path, timeout_s: int = 1800) -> dict[str, Any]:
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "cmd": command,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "elapsed_s": round(time.time() - started, 2),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": command,
            "ok": False,
            "returncode": None,
            "elapsed_s": round(time.time() - started, 2),
            "stdout_tail": str(exc.stdout or "")[-4000:],
            "stderr_tail": str(exc.stderr or "")[-4000:],
            "error": f"TimeoutExpired after {timeout_s}s",
        }


def _prepare_policy_copy(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _selected_envs(raw_env: str) -> list[str]:
    if raw_env == "all":
        return list(ENV_SPECS)
    if raw_env not in ENV_SPECS:
        raise RuntimeError(
            f"unknown env {raw_env!r}; expected all, nle, craftax, crafter, dungeongrid, minigrid, montezuma, or pitfall"
        )
    return [raw_env]


def _parse_candidate_id_filter(raw: str) -> tuple[str, ...]:
    candidate_ids = tuple(part.strip() for part in str(raw or "").split(",") if part.strip())
    invalid = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id in {".", ".."}
        or "/" in candidate_id
        or "\\" in candidate_id
        or Path(candidate_id).name != candidate_id
    ]
    if invalid:
        raise RuntimeError(f"invalid candidate id filter: {', '.join(invalid)}")
    return candidate_ids


def _candidate_paths(
    candidate_root: Path | None,
    env_id: str,
    *,
    candidate_ids: tuple[str, ...] = (),
) -> list[tuple[str, Path]]:
    if candidate_root is None:
        return []
    env_root = candidate_root / env_id
    if candidate_ids:
        candidates: list[tuple[str, Path]] = []
        missing: list[str] = []
        for candidate_id in candidate_ids:
            policy_path = env_root / candidate_id / "heuristic_policy.py"
            if policy_path.is_file():
                candidates.append((candidate_id, policy_path))
            else:
                missing.append(str(policy_path))
        if missing:
            raise RuntimeError("assigned candidate policy missing: " + ", ".join(missing))
        return candidates
    if not env_root.exists():
        return []
    candidates: list[tuple[str, Path]] = []
    for policy_path in sorted(env_root.glob("*/heuristic_policy.py")):
        candidates.append((policy_path.parent.name, policy_path))
    return candidates


def _run_sweep(
    *,
    spec: EnvSpec,
    policy_path: Path,
    output_dir: Path,
    seed_csv: str,
    heldout: bool,
    strict_env: bool,
    service_url: str | None,
    offline_fixture_on_unavailable: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_result = _compile_policy(policy_path)
    if not compile_result["ok"]:
        return {
            "ok": False,
            "status": "compile_failed",
            "compile": compile_result,
            "summary": None,
            "score": 0.0,
        }
    effective_service_url = service_url or spec.service_url
    if effective_service_url and not _service_reachable(effective_service_url):
        if offline_fixture_on_unavailable:
            summary = {
                "schema_version": "hillclimbsymbolicbench.offline_fixture.v1",
                "env": spec.env_id,
                "episode_count": 1,
                "completed": 1,
                "reward": {"mean": 8.4, "median": 8.4},
                "achievement_frequency": {
                    "collect_wood": 1,
                    "collect_stone": 1,
                    "collect_coal": 1,
                    "collect_iron": 1,
                    "make_wood_pickaxe": 1,
                    "make_stone_pickaxe": 1,
                    "make_iron_pickaxe": 1,
                    "collect_diamond": 0,
                },
                "failure_modes": {"offline_fixture": 1},
            }
            _write_json(output_dir / "summary.json", summary)
            return {
                "ok": True,
                "status": "completed",
                "compile": compile_result,
                "summary": summary,
                "score": _score_summary(spec.env_id, summary),
                "service_url": effective_service_url,
                "fixture": "offline_service_unavailable",
            }
        status = "service_unavailable"
        if strict_env:
            status = "service_unavailable_strict"
        return {
            "ok": not strict_env,
            "status": status,
            "compile": compile_result,
            "summary": None,
            "score": 0.0,
            "service_url": effective_service_url,
        }

    command = [
        *_sweep_python_prefix(spec),
        str(spec.sweep_script),
        "--policy-path",
        str(policy_path),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(spec.max_steps),
        "--max-workers",
        str(spec.max_workers),
    ]
    if effective_service_url:
        command.extend(["--base-url", effective_service_url])
    if spec.env_id == "craftax":
        command.extend(["--env-kind", os.environ.get("CRAFTAX_ENV_KIND", "classic")])
    if seed_csv:
        if spec.env_id == "craftax":
            seeds = _parse_seed_csv(seed_csv)
            if seeds:
                command.extend(
                    ["--seed-start", str(seeds[0]), "--seed-step", "2", "--count", str(len(seeds))]
                )
        else:
            command.extend(["--seeds", seed_csv])
    if spec.env_id == "dungeongrid":
        command.append("--no-traces")
        if heldout:
            command.append("--redact-seeds")

    run_result = _run_command(command, cwd=spec.policy_path.parent)
    summary_path = output_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    )
    rollout_failures = int((summary or {}).get("failed") or 0)
    rollout_completed = int((summary or {}).get("completed") or 0)
    rollout_seed_count = int(
        (summary or {}).get("seed_count") or rollout_completed + rollout_failures
    )
    strict_rollout_failure = bool(
        strict_env
        and summary is not None
        and (rollout_failures > 0 or rollout_completed < max(rollout_seed_count, 1))
    )
    score = _score_summary(spec.env_id, summary or {})
    return {
        "ok": bool(run_result["ok"] and summary is not None and not strict_rollout_failure),
        "status": "sweep_failed_strict"
        if strict_rollout_failure
        else ("completed" if run_result["ok"] and summary is not None else "sweep_failed"),
        "compile": compile_result,
        "sweep": run_result,
        "summary": summary,
        "score": score,
    }


def _sweep_python_prefix(spec: EnvSpec) -> list[str]:
    if spec.env_id == "minigrid":
        uv = shutil.which("uv")
        evals_project = HOST_WORKSPACE_ROOT / "evals"
        if uv and (evals_project / "pyproject.toml").exists():
            return [uv, "run", "--project", str(evals_project), "--group", "minigrid", "python"]
    return [sys.executable]


def _score_summary(env_id: str, summary: dict[str, Any]) -> float:
    reward = summary.get("reward") if isinstance(summary.get("reward"), dict) else {}
    mean_reward = float(reward.get("mean") or 0.0)
    median_reward = float(reward.get("median") or 0.0)
    achievements = summary.get("achievement_frequency")
    achievement_diversity = (
        len(achievements)
        if isinstance(achievements, dict)
        else float(summary.get("achievement_count_mean") or 0.0)
    )
    failure_modes = (
        summary.get("failure_modes") if isinstance(summary.get("failure_modes"), dict) else {}
    )
    completed = float(
        summary.get("completed") or summary.get("episode_count") or summary.get("seed_count") or 1
    )
    success_rate = float(summary.get("successes") or 0.0) / max(completed, 1.0)
    if env_id == "craftax":
        iron_rate = _achievement_rate(achievements, "make_iron_pickaxe", completed)
        diamond_rate = _achievement_rate(achievements, "collect_diamond", completed)
        return round(
            mean_reward
            + 0.25 * median_reward
            + iron_rate
            + 2.0 * diamond_rate
            + 0.02 * achievement_diversity,
            4,
        )
    if env_id == "nle":
        descent_rate = _failure_rate(failure_modes, "descended_to_level_2_only", completed)
        dl3_rate = _failure_rate(failure_modes, "reached_dungeon_level_3", completed)
        return round(
            mean_reward + 0.5 * descent_rate + 0.75 * dl3_rate + 0.02 * achievement_diversity, 4
        )
    if env_id == "crafter":
        wood_rate = _achievement_rate(achievements, "collect_wood", completed)
        table_rate = _achievement_rate(achievements, "place_table", completed)
        pickaxe_rate = _achievement_rate(achievements, "make_wood_pickaxe", completed)
        health_mean = float(summary.get("health_mean") or 0.0)
        return round(
            mean_reward
            + 0.25 * wood_rate
            + 0.4 * table_rate
            + 0.6 * pickaxe_rate
            + 0.02 * achievement_diversity
            + 0.01 * health_mean,
            4,
        )
    if env_id == "dungeongrid":
        treasure = float(summary.get("treasure_mean") or 0.0)
        explored = float(summary.get("explored_tiles_mean") or 0.0)
        alive = float(summary.get("heroes_alive_mean") or 0.0)
        return round(
            mean_reward + 0.75 * success_rate + 0.20 * treasure + 0.015 * explored + 0.08 * alive, 4
        )
    if env_id == "minigrid":
        mission_rate = _achievement_rate(achievements, "mission_complete", completed)
        goal_rate = _achievement_rate(achievements, "goal_reached", completed)
        pickup_rate = _achievement_rate(achievements, "first_object_pickup", completed)
        door_rate = _achievement_rate(achievements, "first_door_toggled", completed)
        return round(
            mean_reward
            + 1.5 * success_rate
            + 0.5 * mission_rate
            + 0.25 * goal_rate
            + 0.1 * pickup_rate
            + 0.1 * door_rate
            + 0.02 * achievement_diversity,
            4,
        )
    if env_id == "montezuma":
        key_rate = _achievement_rate(achievements, "first_key_collected", completed)
        door_rate = _achievement_rate(achievements, "first_door_opened", completed)
        room2_rate = _achievement_rate(achievements, "room_2_reached", completed)
        room5_rate = _achievement_rate(achievements, "room_5_reached", completed)
        return round(
            mean_reward
            + 0.75 * key_rate
            + 1.0 * door_rate
            + 1.5 * room2_rate
            + 2.0 * room5_rate
            + 0.03 * achievement_diversity,
            4,
        )
    if env_id == "pitfall":
        screen_rate = _achievement_rate(achievements, "first_screen_traversed", completed)
        treasure_rate = _achievement_rate(achievements, "first_treasure_collected", completed)
        vine_rate = _achievement_rate(achievements, "first_vine_swing", completed)
        pit_rate = _achievement_rate(achievements, "first_pit_jumped", completed)
        max_screen = float(summary.get("max_screen_mean") or 0.0)
        collisions = float(summary.get("collision_proxy_mean") or 0.0)
        return round(
            mean_reward
            + 0.5 * screen_rate
            + 0.75 * treasure_rate
            + 0.25 * vine_rate
            + 0.25 * pit_rate
            + 0.03 * max_screen
            - 0.15 * collisions
            + 0.02 * achievement_diversity,
            4,
        )
    return round(mean_reward + 0.02 * achievement_diversity, 4)


def _achievement_rate(achievement_frequency: Any, name: str, total: float) -> float:
    if not isinstance(achievement_frequency, dict):
        return 0.0
    return float(achievement_frequency.get(name) or 0.0) / max(total, 1.0)


def _failure_rate(failure_modes: dict[str, Any], name: str, total: float) -> float:
    return float(failure_modes.get(name) or 0.0) / max(total, 1.0)


def _achievement_diversity(env_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_env: dict[str, Any] = {}
    for result in env_results:
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        frequency = summary.get("achievement_frequency")
        if isinstance(frequency, dict):
            by_env[result["env"]] = {
                "unique_count": len(frequency),
                "achievement_frequency": frequency,
            }
        else:
            by_env[result["env"]] = {
                "achievement_count_mean": summary.get("achievement_count_mean"),
                "quest_achievement_count_mean": summary.get("quest_achievement_count_mean"),
            }
    return {"schema_version": "hillclimbsymbolicbench.achievement_diversity.v1", "envs": by_env}


def _score_config() -> dict[str, Any]:
    return {
        "schema_version": "hillclimbsymbolicbench.scoring.v1",
        "score_source_precedence": ["heldout", "train"],
        "acceptance_delta": 0.01,
        "formulas": {
            "craftax": "mean_reward + 0.25*median_reward + iron_pickaxe_rate + 2*diamond_rate + 0.02*achievement_diversity",
            "nle": "mean_reward + 0.5*descent_rate + 0.75*dungeon_level_3_rate + 0.02*achievement_diversity",
            "crafter": "mean_reward + 0.25*wood_rate + 0.4*table_rate + 0.6*wood_pickaxe_rate + 0.02*achievement_diversity + 0.01*health_mean",
            "dungeongrid": "mean_reward + 0.75*success_rate + 0.20*treasure_mean + 0.015*explored_tiles_mean + 0.08*heroes_alive_mean",
            "minigrid": "mean_reward + 1.5*success_rate + 0.5*mission_complete_rate + 0.25*goal_reached_rate + 0.1*object_pickup_rate + 0.1*door_toggled_rate + 0.02*achievement_diversity",
        },
    }


def _starting_container_manifest(
    env_ids: list[str],
    *,
    service_receipts: dict[str, Any],
) -> dict[str, Any]:
    envs: dict[str, Any] = {}
    for env_id in env_ids:
        spec = ENV_SPECS[env_id]
        policy_exists = spec.policy_path.exists()
        sweep_exists = spec.sweep_script.exists()
        train_seeds = _parse_seed_csv(spec.train_seeds)
        envs[env_id] = {
            "container_root": str(spec.policy_path.parent),
            "baseline_policy_path": str(spec.policy_path),
            "baseline_policy_present": policy_exists,
            "baseline_policy_sha256": _sha256_file(spec.policy_path) if policy_exists else None,
            "sweep_script_path": str(spec.sweep_script),
            "sweep_script_present": sweep_exists,
            "service_url": spec.service_url,
            "service_required": spec.service_url is not None,
            "task_owned_service": service_receipts.get(env_id),
            "service_available": bool((service_receipts.get(env_id) or {}).get("ready"))
            if spec.service_url
            else True,
            "train_seed_count": len(train_seeds),
            "train_seed_hash": _seed_hash(train_seeds) if train_seeds else None,
            "max_steps": spec.max_steps,
            "max_workers": spec.max_workers,
        }
    return {
        "schema_version": "hillclimbsymbolicbench.starting_container.v1",
        "runtime_workspace_roots": [str(path) for path in _runtime_workspace_roots()],
        "host_private_repo_root_fallback": str(HOST_PRIVATE_REPO_ROOT),
        "public_containers_src": str(_public_containers_src() or ""),
        "envs": envs,
    }


def _best_score(results: list[dict[str, Any]]) -> float:
    scored = [
        float(item.get("score") or 0.0) for item in results if item.get("status") == "completed"
    ]
    return round(statistics.mean(scored), 4) if scored else 0.0


def _copy_best_policy(best_policy: Path, workproduct: Path) -> Path:
    dest = workproduct / "best_policy" / "heuristic_policy.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_policy, dest)
    return dest


def _copy_candidate_policy(
    *,
    policy_path: Path,
    workproduct: Path,
    env_id: str,
    candidate_id: str,
) -> Path:
    dest = workproduct / "candidates" / env_id / candidate_id / "heuristic_policy.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_path, dest)
    return dest


def _copy_result_artifacts(
    *,
    source_root: Path,
    workproduct: Path,
    output_root: Path,
    env_id: str,
    candidate_id: str,
) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for phase_name in ("train", "heldout"):
        phase_src = source_root / phase_name
        if not phase_src.exists():
            continue
        phase_dest = workproduct / "experiment_results" / env_id / candidate_id / phase_name
        phase_refs: dict[str, str] = {}
        for filename, ref_key in (
            ("summary.json", "summary_path"),
            ("results.json", "per_seed_path"),
        ):
            source = phase_src / filename
            if not source.exists():
                continue
            dest = phase_dest / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            phase_refs[ref_key] = str(dest.relative_to(output_root))
        if phase_refs:
            refs[phase_name] = phase_refs
    return refs


def _experiment_contract(
    *,
    env_id: str,
    candidate_id: str,
    candidate_label: str,
    source_kind: str,
    candidate_hash: str,
    parent_hash: str,
    candidate_policy_path: str,
    train_seeds: list[int],
    baseline_value: float | None,
    result_value: float,
    delta: float | None,
    score_source: str,
    accepted: bool,
    status: str | None,
    result_artifacts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    candidate_ref = f"{env_id}:{candidate_label}"
    protocol = {
        "benchmark_family": BENCHMARK_FAMILY,
        "metric": "symbolic_policy_score",
        "metric_direction": "higher_is_better",
        "split_name": score_source,
        "seed_set": train_seeds,
        "sample_size": len(train_seeds),
        "acceptance_delta": 0.01,
        "scorer": "workspace/run_hillclimb_symbolicbench_task.py",
    }
    summary_ref = result_artifacts.get(score_source, {}).get("summary_path")
    per_seed_ref = result_artifacts.get(score_source, {}).get("per_seed_path")
    return {
        "baseline_snapshot": {
            "candidate_id": f"{env_id}:baseline",
            "candidate_kind": "code_policy",
            "digest": parent_hash,
            "entrypoint": "Policy.act",
        },
        "candidate_snapshot": {
            "candidate_id": candidate_id,
            "candidate_kind": "code_policy",
            "candidate_label": candidate_label,
            "source_kind": source_kind,
            "digest": candidate_hash,
            "entrypoint": "Policy.act",
            "artifact_path": candidate_policy_path,
        },
        "protocol_snapshot": protocol,
        "artifact_refs": {
            "candidate_policy_path": candidate_policy_path,
            "result_artifacts": result_artifacts,
            "selected_summary_path": summary_ref,
            "selected_per_seed_path": per_seed_ref,
        },
        "result_summary": {
            "candidate_id": candidate_id,
            "candidate_kind": "code_policy",
            "candidate_label": candidate_label,
            "metric": "symbolic_policy_score",
            "metric_direction": "higher_is_better",
            "value": result_value,
            "baseline_value": baseline_value,
            "delta": delta,
            "sample_size": len(train_seeds),
            "seed_set": train_seeds,
            "split_name": score_source,
            "summary_artifact_path": summary_ref,
            "per_seed_artifact_path": per_seed_ref,
            "truth_status": "accepted" if accepted else "observed",
            "status": status,
        },
        "decision_summary": {
            "candidate_id": candidate_ref,
            "accepted": bool(accepted),
            "improved_over_baseline": bool(delta is not None and delta > 0.0),
            "status": status,
        },
    }


def _write_reproduction(
    *,
    path: Path,
    envs: list[str],
    score_source: str,
    best_candidate_id: str,
    best_score: float,
    heldout_meta: dict[str, Any] | None,
) -> None:
    heldout_line = "not configured"
    if heldout_meta:
        heldout_line = f"{heldout_meta['seed_count']} seeds, hash {heldout_meta['seed_hash']}"
    path.write_text(
        "\n".join(
            [
                "# HillclimbSymbolicBench Workproduct",
                "",
                f"Environments: {', '.join(envs)}",
                f"Score source: {score_source}",
                f"Best candidate: {best_candidate_id}",
                f"Best score: {best_score}",
                f"Heldout: {heldout_line}",
                "",
                "Re-run from the evals checkout with:",
                "",
                "```bash",
                f"python3 workspace/run_hillclimb_symbolicbench_task.py run --output-root /tmp/hillclimb-symbolicbench --env {envs[0]} --iterations 0",
                "```",
                "",
                "The SMR DEO worker loop publishes from artifacts/workproduct_container/eval_summary.json.",
                "Run the score subcommand only as a separate verifier gate after report artifacts are complete.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _best_non_baseline_record(eval_summary: dict[str, Any]) -> dict[str, Any] | None:
    best_candidate_id = str(eval_summary.get("best_candidate_id") or "")
    records = eval_summary.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if (
            isinstance(record, dict)
            and record.get("source_kind") != "baseline"
            and str(record.get("candidate_id") or "") == best_candidate_id
        ):
            return record
    for record in records:
        if isinstance(record, dict) and record.get("source_kind") != "baseline":
            return record
    return None


def _achievement_frequency(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    train = record.get("train")
    if not isinstance(train, dict):
        return {}
    summary = train.get("summary")
    if not isinstance(summary, dict):
        return {}
    frequency = summary.get("achievement_frequency")
    return frequency if isinstance(frequency, dict) else {}


def _record_achievement_frequency(record: dict[str, Any]) -> dict[str, Any]:
    return _achievement_frequency(record)


def _candidate_progression_lines(eval_summary: dict[str, Any]) -> list[str]:
    records = eval_summary.get("records")
    if not isinstance(records, list):
        return ["- No candidate records found."]
    lines = [
        "| Candidate | Kind | Status | Score | Delta | Path |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        if not isinstance(record, dict):
            continue
        lines.append(
            "| {candidate} | {kind} | {status} | {score} | {delta} | {path} |".format(
                candidate=record.get("candidate_id") or "",
                kind=record.get("source_kind") or "",
                status=record.get("status") or "",
                score=record.get("score"),
                delta=record.get("score_delta"),
                path=record.get("candidate_policy_path") or "",
            )
        )
    return lines


def _achievement_progression_lines(eval_summary: dict[str, Any]) -> list[str]:
    records = [record for record in eval_summary.get("records", []) if isinstance(record, dict)]
    if not records:
        return ["- No achievement records found."]
    achievements = sorted(
        {name for record in records for name in _record_achievement_frequency(record)}
    )
    if not achievements:
        return ["- No achievement frequencies recorded."]
    candidate_ids = [str(record.get("candidate_id") or "") for record in records]
    lines = [
        "| Achievement | " + " | ".join(candidate_ids) + " |",
        "| --- | " + " | ".join("---:" for _ in candidate_ids) + " |",
    ]
    for achievement in achievements:
        cells = [
            str(_record_achievement_frequency(record).get(achievement, 0)) for record in records
        ]
        lines.append("| " + achievement + " | " + " | ".join(cells) + " |")
    return lines


def _write_final_report(
    *,
    path: Path,
    eval_summary: dict[str, Any],
    seed_count: int,
) -> None:
    best_record = _best_non_baseline_record(eval_summary)
    candidate_path = (
        str(best_record.get("candidate_policy_path") or "") if isinstance(best_record, dict) else ""
    )
    candidate_id = (
        str(best_record.get("candidate_id") or "")
        if isinstance(best_record, dict)
        else str(eval_summary.get("best_candidate_id") or "")
    )
    candidate_score = (
        best_record.get("score")
        if isinstance(best_record, dict)
        else eval_summary.get("best_score")
    )
    candidate_delta = (
        best_record.get("score_delta")
        if isinstance(best_record, dict)
        else eval_summary.get("best_score_delta")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Crafter Candidate Evidence Report",
                "",
                "## Summary",
                f"- Best candidate: {eval_summary.get('best_candidate_id')}",
                f"- Reported candidate: {candidate_id}",
                f"- Candidate path: {candidate_path or 'unavailable'}",
                f"- Baseline score: {eval_summary.get('baseline_score')}",
                f"- Candidate score: {candidate_score}",
                f"- Score delta: {candidate_delta}",
                f"- Seed count: {seed_count}",
                f"- Score source: {eval_summary.get('score_source')}",
                "",
                "## Candidate Progression",
                *_candidate_progression_lines(eval_summary),
                "",
                "## Per-Achievement Frequencies",
                *_achievement_progression_lines(eval_summary),
                "",
                "## Artifact Paths",
                "- eval_summary: artifacts/workproduct_container/eval_summary.json",
                "- experiment_results: artifacts/workproduct_container/experiment_results.json",
                "- candidate_ledger: artifacts/workproduct_container/candidate_ledger.jsonl",
                "- achievement_diversity: artifacts/workproduct_container/achievement_diversity.json",
                "- reproduction: artifacts/workproduct_container/reproduction.md",
                "- workproduct_archive: artifacts/workproduct_container.tar.gz",
                "",
                "Publish this file with publish_report_work_product before set_task_state(done).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _tar_workproduct(workproduct: Path) -> Path:
    tar_path = workproduct.parent / "workproduct_container.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(workproduct, arcname="workproduct_container")
    return tar_path


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    artifacts = output_root / "artifacts"
    workproduct = artifacts / "workproduct_container"
    candidates_out = output_root / "candidate_runs"
    if workproduct.exists():
        shutil.rmtree(workproduct)
    workproduct.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts / "scoring_config.json", _score_config())

    env_ids = _selected_envs(args.env)
    heldout_seeds = _parse_seed_csv(os.environ.get(args.heldout_seeds_env or HELDOUT_ENV, ""))
    heldout_meta = (
        {"seed_count": len(heldout_seeds), "seed_hash": _seed_hash(heldout_seeds)}
        if heldout_seeds
        else None
    )
    train_seed_overrides = _seed_overrides(args.train_seeds)
    candidate_root = (
        Path(args.candidate_root).expanduser().resolve() if args.candidate_root else None
    )
    candidate_id_filter = _parse_candidate_id_filter(args.candidate_id)
    ledger_path = workproduct / "candidate_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    best_policy: Path | None = None
    best_candidate_id = ""
    best_score = -1.0
    directed_outcomes: list[dict[str, Any]] = []
    service_receipts: dict[str, Any] = {}
    service_urls: dict[str, str | None] = {}
    service_processes: list[subprocess.Popen[str]] = []

    try:
        for env_id in env_ids:
            spec = ENV_SPECS[env_id]
            if args.max_steps is not None:
                spec = replace(spec, max_steps=max(1, int(args.max_steps)))
            if args.max_workers is not None:
                spec = replace(spec, max_workers=max(1, int(args.max_workers)))
            if spec.service_url and not args.no_start_services:
                service_url, process, receipt = _start_task_owned_service(
                    spec=spec,
                    output_root=output_root,
                )
                service_receipts[env_id] = receipt
                service_urls[env_id] = service_url
                if process is not None:
                    service_processes.append(process)
            else:
                service_receipts[env_id] = {
                    "mode": "not_required" if not spec.service_url else "disabled",
                    "ready": spec.service_url is None or _service_reachable(spec.service_url),
                    "url": spec.service_url,
                }
                service_urls[env_id] = spec.service_url
        starting_container = _starting_container_manifest(
            env_ids,
            service_receipts=service_receipts,
        )
        _write_json(workproduct / "starting_container.json", starting_container)
        service_failures = {
            env_id: receipt
            for env_id, receipt in service_receipts.items()
            if ENV_SPECS[env_id].service_url and not receipt.get("ready")
        }
        if args.strict_env and service_failures:
            _write_json(
                artifacts / "service_startup_failure.json",
                {
                    "schema_version": "hillclimbsymbolicbench.service_startup_failure.v1",
                    "strict_env": True,
                    "failures": service_failures,
                },
            )
            return 2

        for env_id in env_ids:
            spec = ENV_SPECS[env_id]
            baseline_policy = _prepare_policy_copy(
                spec.policy_path,
                output_root / "baselines" / env_id / "heuristic_policy.py",
            )
            train_csv = train_seed_overrides.get(env_id) or spec.train_seeds
            candidate_entries = [("baseline", baseline_policy)] + _candidate_paths(
                candidate_root,
                env_id,
                candidate_ids=candidate_id_filter,
            )
            incumbent_env_score = -1.0
            baseline_env_score: float | None = None
            for candidate_id, policy_path in candidate_entries:
                parent_hash = _sha256_file(baseline_policy)
                candidate_hash = _sha256_file(policy_path)
                candidate_policy_out = _copy_candidate_policy(
                    policy_path=policy_path,
                    workproduct=workproduct,
                    env_id=env_id,
                    candidate_id=candidate_id,
                )
                candidate_run_root = candidates_out / env_id / candidate_id
                train_result = _run_sweep(
                    spec=spec,
                    policy_path=policy_path,
                    output_dir=candidate_run_root / "train",
                    seed_csv=train_csv,
                    heldout=False,
                    strict_env=bool(args.strict_env),
                    service_url=service_urls.get(env_id),
                    offline_fixture_on_unavailable=False,
                )
                heldout_result: dict[str, Any] | None = None
                if heldout_seeds and train_result.get("status") == "completed":
                    heldout_result = _run_sweep(
                        spec=spec,
                        policy_path=policy_path,
                        output_dir=candidate_run_root / "heldout",
                        seed_csv=",".join(str(seed) for seed in heldout_seeds),
                        heldout=True,
                        strict_env=bool(args.strict_env),
                        service_url=service_urls.get(env_id),
                        offline_fixture_on_unavailable=False,
                    )
                score_source = (
                    "heldout"
                    if heldout_result and heldout_result.get("status") == "completed"
                    else "train"
                )
                effective_result = heldout_result if score_source == "heldout" else train_result
                env_score = float((effective_result or {}).get("score") or 0.0)
                baseline_value = None if candidate_id == "baseline" else baseline_env_score
                score_delta = (
                    round(env_score - baseline_value, 4) if baseline_value is not None else None
                )
                accepted = candidate_id == "baseline" or env_score >= incumbent_env_score + 0.01
                if accepted:
                    incumbent_env_score = env_score
                if candidate_id == "baseline":
                    baseline_env_score = env_score
                result_artifacts = _copy_result_artifacts(
                    source_root=candidate_run_root,
                    workproduct=workproduct,
                    output_root=output_root,
                    env_id=env_id,
                    candidate_id=candidate_id,
                )
                candidate_policy_rel = str(candidate_policy_out.relative_to(output_root))
                record = {
                    "schema_version": "hillclimbsymbolicbench.candidate.v1",
                    "env": env_id,
                    "candidate_id": f"{env_id}:{candidate_id}",
                    "candidate_hash": candidate_hash,
                    "candidate_policy_path": candidate_policy_rel,
                    "parent_policy_hash": parent_hash,
                    "source_kind": "baseline" if candidate_id == "baseline" else "candidate",
                    "score": env_score,
                    "score_source": score_source,
                    "baseline_value": baseline_value,
                    "score_delta": score_delta,
                    "accepted": accepted,
                    "status": (effective_result or {}).get("status"),
                    "train": _compact_eval_ref(train_result),
                    "heldout": _compact_eval_ref(heldout_result) if heldout_result else None,
                    "result_artifacts": result_artifacts,
                }
                record["experiment_contract"] = _experiment_contract(
                    env_id=env_id,
                    candidate_id=str(record["candidate_id"]),
                    candidate_label=candidate_id,
                    source_kind=str(record["source_kind"]),
                    candidate_hash=candidate_hash,
                    parent_hash=parent_hash,
                    candidate_policy_path=candidate_policy_rel,
                    train_seeds=_parse_seed_csv(train_csv),
                    baseline_value=baseline_value,
                    result_value=env_score,
                    delta=score_delta,
                    score_source=score_source,
                    accepted=accepted,
                    status=(effective_result or {}).get("status"),
                    result_artifacts=result_artifacts,
                )
                _append_jsonl(ledger_path, record)
                directed_outcomes.append(_directed_outcome_record(record))
                if accepted and env_score > best_score:
                    best_score = env_score
                    best_candidate_id = f"{env_id}:{candidate_id}"
                    best_policy = policy_path
    finally:
        _stop_task_owned_services(service_processes)

    if best_policy is None:
        fallback = ENV_SPECS[env_ids[0]].policy_path
        best_policy = fallback
        best_candidate_id = f"{env_ids[0]}:baseline"
        best_score = 0.0
    best_policy_out = _copy_best_policy(best_policy, workproduct)
    all_records = _read_jsonl(ledger_path)
    completed_records = [record for record in all_records if record.get("status") == "completed"]
    baseline_records = [
        record for record in completed_records if record.get("source_kind") == "baseline"
    ]
    non_baseline_records = [
        record for record in all_records if record.get("source_kind") != "baseline"
    ]
    completed_non_baseline_records = [
        record for record in non_baseline_records if record.get("status") == "completed"
    ]
    baseline_score = max(
        (float(record.get("score") or 0.0) for record in baseline_records), default=0.0
    )
    best_source_kind = next(
        (
            str(record.get("source_kind") or "")
            for record in all_records
            if record.get("candidate_id") == best_candidate_id
        ),
        "",
    )
    eval_summary = {
        "schema_version": "hillclimbsymbolicbench.eval_summary.v1",
        "envs": env_ids,
        "candidate_count": len(all_records),
        "completed_candidate_count": len(completed_records),
        "non_baseline_candidate_count": len(non_baseline_records),
        "completed_non_baseline_candidate_count": len(completed_non_baseline_records),
        "best_candidate_id": best_candidate_id,
        "best_source_kind": best_source_kind,
        "best_score": best_score,
        "baseline_score": baseline_score,
        "best_score_delta": round(best_score - baseline_score, 4),
        "score_source": "heldout" if heldout_meta else "train",
        "heldout": {"details_redacted": True, **heldout_meta}
        if heldout_meta
        else {"enabled": False},
        "records": all_records,
        "experiment_result_contract": {
            "backend_result_rows": "one aggregate row per candidate metric/split",
            "per_seed_detail": "artifacts/workproduct_container/experiment_results/*/*/*/results.json",
        },
    }
    experiment_results = {
        "schema_version": "hillclimbsymbolicbench.experiment_results.v1",
        "results": [
            {
                "candidate_id": str(record.get("candidate_id") or ""),
                "source_kind": str(record.get("source_kind") or ""),
                **(
                    record.get("experiment_contract")
                    if isinstance(record.get("experiment_contract"), dict)
                    else {}
                ),
            }
            for record in all_records
        ],
    }
    diversity = _achievement_diversity(
        [
            {
                "env": str(record.get("env")),
                "summary": (
                    (record.get("heldout") or record.get("train") or {}).get("summary")
                    if isinstance(record.get("heldout") or record.get("train"), dict)
                    else None
                ),
            }
            for record in all_records
        ]
    )
    _write_json(workproduct / "eval_summary.json", eval_summary)
    _write_json(workproduct / "experiment_results.json", experiment_results)
    _write_json(workproduct / "achievement_diversity.json", diversity)
    _write_json(
        workproduct / "directed_effort_outcomes.json",
        {
            "schema_version": "hillclimbsymbolicbench.directed_effort_outcomes.v1",
            "outcomes": directed_outcomes,
        },
    )
    _write_reproduction(
        path=workproduct / "reproduction.md",
        envs=env_ids,
        score_source=eval_summary["score_source"],
        best_candidate_id=best_candidate_id,
        best_score=best_score,
        heldout_meta=heldout_meta,
    )
    report_seed_count = len(
        _parse_seed_csv(train_seed_overrides.get(env_ids[0]) or ENV_SPECS[env_ids[0]].train_seeds)
    )
    _write_final_report(
        path=output_root / "reports" / "final_report.md",
        eval_summary=eval_summary,
        seed_count=report_seed_count,
    )
    tar_path = workproduct.parent / "workproduct_container.tar.gz"
    manifest = {
        "schema_version": "hillclimbsymbolicbench.manifest.v1",
        "task_id": TASK_ID,
        "benchmark_family": BENCHMARK_FAMILY,
        "created_at_unix": int(time.time()),
        "best_policy": {
            "present": best_policy_out.exists(),
            "path": str(best_policy_out.relative_to(output_root)),
            "sha256": _sha256_file(best_policy_out),
            "candidate_id": best_candidate_id,
        },
        "candidate_policies": [
            {
                "candidate_id": str(record.get("candidate_id") or ""),
                "path": str(record.get("candidate_policy_path") or ""),
                "sha256": str(record.get("candidate_hash") or ""),
            }
            for record in all_records
            if str(record.get("candidate_policy_path") or "")
        ],
        "workproduct_tar": str(tar_path.relative_to(output_root)),
        "compatible_models": ["deepseek/deepseek-v4-flash", "gpt-5.4-mini"],
        "starting_container": {
            "present": (workproduct / "starting_container.json").exists(),
            "path": str((workproduct / "starting_container.json").relative_to(output_root)),
        },
        "experiment_results": {
            "present": (workproduct / "experiment_results.json").exists(),
            "path": str((workproduct / "experiment_results.json").relative_to(output_root)),
        },
    }
    _write_json(workproduct / "manifest.json", manifest)
    _tar_workproduct(workproduct)
    reportbench_output = {
        "task_id": TASK_ID,
        "benchmark_family": BENCHMARK_FAMILY,
        "reward": {"primary_metric": "symbolic_policy_score", "value": max(best_score, 0.0)},
        "best_candidate_id": best_candidate_id,
        "verifier": {"score": 1.0 if best_policy_out.exists() and completed_records else 0.0},
        "runtime": {"cost_usd": 0.0},
    }
    _write_json(artifacts / "reportbench_output.json", reportbench_output)
    if args.strict_env and not completed_records:
        _write_json(
            artifacts / "run_failed.json",
            {
                "schema_version": "hillclimbsymbolicbench.run_failed.v1",
                "reason": "no_completed_candidate_evaluations",
                "strict_env": True,
                "service_receipts": service_receipts,
            },
        )
        return 2
    return 0


def _seed_overrides(raw: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for chunk in str(raw or "").split(";"):
        if not chunk.strip() or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if key.strip() in ENV_SPECS:
            overrides[key.strip()] = value.strip()
    return overrides


def _compact_eval_ref(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "score": result.get("score"),
        "summary": result.get("summary"),
        "compile_ok": (result.get("compile") or {}).get("ok")
        if isinstance(result.get("compile"), dict)
        else None,
    }


def _directed_outcome_record(candidate_record: dict[str, Any]) -> dict[str, Any]:
    state = "completed" if candidate_record.get("accepted") else "closed"
    return {
        "directed_effort_outcome_id": f"local-{candidate_record['candidate_id']}",
        "state": state,
        "outcome_text": (
            f"{candidate_record['candidate_id']} scored {candidate_record['score']} "
            f"from {candidate_record['score_source']} and was "
            f"{'accepted' if candidate_record.get('accepted') else 'rejected'}."
        ),
        "metadata": {
            "env": candidate_record.get("env"),
            "candidate_hash": candidate_record.get("candidate_hash"),
            "parent_policy_hash": candidate_record.get("parent_policy_hash"),
            "score": candidate_record.get("score"),
            "score_source": candidate_record.get("score_source"),
            "accepted": candidate_record.get("accepted"),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _task_config() -> dict[str, Any]:
    if not TASK_TOML_PATH.is_file():
        return {}
    with TASK_TOML_PATH.open("rb") as handle:
        loaded = tomllib.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _rubric_weights() -> dict[str, float]:
    if not RUBRIC_JSON_PATH.is_file():
        return {"crafter_bundle": 1.0}
    payload = json.loads(RUBRIC_JSON_PATH.read_text(encoding="utf-8"))
    rows = payload.get("criteria") if isinstance(payload, dict) else None
    weights: dict[str, float] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            criterion_id = str(row.get("id") or "").strip()
            if criterion_id:
                weights[criterion_id] = float(row.get("weight") or 0.0)
    return weights or {"crafter_bundle": 1.0}


def _artifact_excerpt(path: Path, *, output_root: Path, limit: int = 6000) -> dict[str, Any]:
    rel = str(path.relative_to(output_root)) if path.is_relative_to(output_root) else str(path)
    if not path.exists():
        return {"path": rel, "present": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": rel,
        "present": True,
        "size_bytes": path.stat().st_size,
        "text": text[:limit],
        "truncated": len(text) > limit,
    }


def _archive_member_names(output_root: Path) -> set[str]:
    archive = output_root / "artifacts/workproduct_container.tar.gz"
    if not archive.exists():
        return set()
    try:
        with tarfile.open(archive, "r:gz") as tar:
            return set(tar.getnames())
    except tarfile.TarError:
        return set()


def _artifact_ref_exists(output_root: Path, rel_path: str, archive_members: set[str]) -> bool:
    normalized = rel_path.strip().lstrip("/")
    if not normalized:
        return False
    if (output_root / normalized).exists():
        return True
    archive_candidates = {normalized}
    if normalized.startswith("artifacts/workproduct_container/"):
        archive_candidates.add(
            "workproduct_container/" + normalized.removeprefix("artifacts/workproduct_container/")
        )
    elif normalized.startswith("artifacts/"):
        archive_candidates.add(normalized.removeprefix("artifacts/"))
    return any(candidate in archive_members for candidate in archive_candidates)


def _deterministic_verifier_scores(
    errors: list[str], weights: dict[str, float]
) -> dict[str, float]:
    score = 0.0 if errors else 1.0
    return dict.fromkeys(weights, score)


def _run_spark_verifier(
    *,
    output_root: Path,
    errors: list[str],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    task_cfg = _task_config()
    reportbench_cfg = (
        task_cfg.get("reportbench") if isinstance(task_cfg.get("reportbench"), dict) else {}
    )
    verifier_cfg = (
        reportbench_cfg.get("verifier") if isinstance(reportbench_cfg.get("verifier"), dict) else {}
    )
    weights = _rubric_weights()
    fallback_scores = _deterministic_verifier_scores(errors, weights)
    if not bool(verifier_cfg.get("enabled")):
        return {
            "score": 0.0 if errors else 1.0,
            "errors": errors,
            "summary": "valid HillclimbSymbolicBench bundle" if not errors else "; ".join(errors),
            "criteria": [
                {"id": criterion_id, "score": score, "weight": weights.get(criterion_id, 0.0)}
                for criterion_id, score in fallback_scores.items()
            ],
            "judge_source": "deterministic",
        }

    roles_cfg = (
        task_cfg.get("smr", {}).get("roles", {}) if isinstance(task_cfg.get("smr"), dict) else {}
    )
    role_cfg = roles_cfg.get("verifier", {}) if isinstance(roles_cfg, dict) else {}
    judge_cfg = verifier_cfg.get("judge", {}) if isinstance(verifier_cfg.get("judge"), dict) else {}
    model = str(judge_cfg.get("model") or role_cfg.get("model") or "gpt-5.3-codex-spark")
    api_key_env = str(role_cfg.get("api_key_env") or "OPENAI_API_KEY")
    api_key = str(os.environ.get(api_key_env) or "").strip() or None
    pass_threshold = float(verifier_cfg.get("pass_threshold") or 1.0)
    rubric_text = (
        VERIFIER_RUBRIC_PATH.read_text(encoding="utf-8") if VERIFIER_RUBRIC_PATH.is_file() else ""
    )
    system_rubric = "\n".join(
        [
            "You are the Codex Spark verifier for this ReportBench Crafter run.",
            "Use only the supplied task-owned artifacts and deterministic precheck errors.",
            "Return strict JSON only with keys: scores and notes.",
            "scores must be an object mapping each criterion id to a number in [0, 1].",
            "A missing artifact, offline fixture, or deterministic precheck error must score the affected criteria 0.",
            "",
            rubric_text,
        ]
    )
    context = {
        "deterministic_precheck_errors": errors,
        "reportbench_output": report or {},
        "manifest": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/manifest.json",
            output_root=output_root,
        ),
        "eval_summary": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/eval_summary.json",
            output_root=output_root,
        ),
        "experiment_results": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/experiment_results.json",
            output_root=output_root,
        ),
        "achievement_diversity": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/achievement_diversity.json",
            output_root=output_root,
        ),
        "candidate_ledger": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/candidate_ledger.jsonl",
            output_root=output_root,
        ),
        "reproduction": _artifact_excerpt(
            output_root / "artifacts/workproduct_container/reproduction.md",
            output_root=output_root,
        ),
    }
    configured_judge = {
        "kind": str(judge_cfg.get("kind") or "codex"),
        "profile_id": str(judge_cfg.get("profile_id") or "codex_gpt_5_3_spark"),
        "model": model,
        "api_key_env": api_key_env,
        "pass_threshold": pass_threshold,
    }
    try:
        from reportbench.llm_verifier import verify_run

        result = verify_run(
            rubric=system_rubric,
            weights=weights,
            context=context,
            deterministic_fn=lambda _ctx: fallback_scores,
            model=model,
            api_key=api_key,
        )
        score = 0.0 if errors else float(result.score)
        return {
            "score": score,
            "errors": errors,
            "summary": f"Codex Spark verifier score={score:.3f}"
            if not errors
            else "; ".join(errors),
            "criteria": result.criteria,
            "judge_source": result.judge_source,
            "judge_notes": result.judge_notes,
            "configured_judge": configured_judge,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0 if errors else 1.0,
            "errors": errors,
            "summary": "valid HillclimbSymbolicBench bundle" if not errors else "; ".join(errors),
            "criteria": [
                {"id": criterion_id, "score": score, "weight": weights.get(criterion_id, 0.0)}
                for criterion_id, score in fallback_scores.items()
            ],
            "judge_source": "deterministic",
            "judge_notes": f"Spark verifier unavailable: {exc}",
            "configured_judge": configured_judge,
        }


def score(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    errors: list[str] = []
    required = [
        output_root / "artifacts/workproduct_container/manifest.json",
        output_root / "artifacts/workproduct_container/best_policy/heuristic_policy.py",
        output_root / "artifacts/workproduct_container/starting_container.json",
        output_root / "artifacts/workproduct_container/candidate_ledger.jsonl",
        output_root / "artifacts/workproduct_container/eval_summary.json",
        output_root / "artifacts/workproduct_container/experiment_results.json",
        output_root / "artifacts/workproduct_container/achievement_diversity.json",
        output_root / "artifacts/workproduct_container/directed_effort_outcomes.json",
        output_root / "artifacts/workproduct_container/reproduction.md",
        output_root / "reports/final_report.md",
        output_root / "artifacts/reportbench_output.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing {path.relative_to(output_root)}")
    report: dict[str, Any] | None = None
    if not errors:
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        eval_summary = json.loads(required[4].read_text(encoding="utf-8"))
        experiment_results = json.loads(
            (output_root / "artifacts/workproduct_container/experiment_results.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (output_root / "artifacts/reportbench_output.json").read_text(encoding="utf-8")
        )
        if not manifest.get("best_policy", {}).get("present"):
            errors.append("manifest best_policy.present is false")
        if int(eval_summary.get("completed_candidate_count") or 0) < 2:
            errors.append("fewer than two completed candidate evaluations")
        if int(eval_summary.get("completed_non_baseline_candidate_count") or 0) < 1:
            errors.append("no completed non-baseline candidate evaluations")
        if eval_summary.get("best_source_kind") != "candidate":
            errors.append("best policy is not a non-baseline candidate")
        if float(eval_summary.get("best_score_delta") or 0.0) < 0.01:
            errors.append("best policy does not improve over baseline")
        for record in eval_summary.get("records") or []:
            if not isinstance(record, dict):
                continue
            for phase_name in ("train", "heldout"):
                phase = record.get(phase_name)
                if not isinstance(phase, dict):
                    continue
                summary = phase.get("summary")
                if not isinstance(summary, dict):
                    continue
                failure_modes = summary.get("failure_modes")
                used_fixture = (
                    summary.get("schema_version") == "hillclimbsymbolicbench.offline_fixture.v1"
                    or (
                        isinstance(failure_modes, dict)
                        and bool(failure_modes.get("offline_fixture"))
                    )
                    or phase.get("fixture") == "offline_service_unavailable"
                )
                if used_fixture:
                    errors.append(
                        f"offline fixture used for {record.get('candidate_id', 'unknown')} {phase_name}"
                    )
        archive_members = _archive_member_names(output_root)
        result_rows = experiment_results.get("results")
        if not isinstance(result_rows, list):
            errors.append("experiment_results.json has no results list")
        else:
            non_baseline_results = [
                row
                for row in result_rows
                if isinstance(row, dict) and row.get("source_kind") != "baseline"
            ]
            if not non_baseline_results:
                errors.append("experiment_results.json has no non-baseline experiment result")
            for row in non_baseline_results:
                candidate_snapshot = row.get("candidate_snapshot")
                result_summary = row.get("result_summary")
                artifact_refs = row.get("artifact_refs")
                if not isinstance(candidate_snapshot, dict) or not candidate_snapshot.get("digest"):
                    errors.append("experiment result missing candidate_snapshot digest")
                if not isinstance(result_summary, dict) or result_summary.get("value") is None:
                    errors.append("experiment result missing aggregate result_summary value")
                selected_per_seed = (
                    artifact_refs.get("selected_per_seed_path")
                    if isinstance(artifact_refs, dict)
                    else None
                )
                if not selected_per_seed or not _artifact_ref_exists(
                    output_root, str(selected_per_seed), archive_members
                ):
                    errors.append("experiment result missing selected per-seed artifact")
        if report.get("benchmark_family") != BENCHMARK_FAMILY:
            errors.append("reportbench_output benchmark_family mismatch")
    review = _run_spark_verifier(output_root=output_root, errors=errors, report=report)
    _write_json(output_root / "artifacts/verifier_review.json", review)
    if (output_root / "artifacts/reportbench_output.json").exists():
        report = json.loads(
            (output_root / "artifacts/reportbench_output.json").read_text(encoding="utf-8")
        )
        report["verifier"] = review
        _write_json(output_root / "artifacts/reportbench_output.json", report)
    pass_threshold = float(
        _task_config().get("reportbench", {}).get("verifier", {}).get("pass_threshold") or 1.0
    )
    return 0 if not errors and float(review.get("score") or 0.0) >= pass_threshold else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HillclimbSymbolicBench symbolic policy hillclimbs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument(
        "--env",
        default="all",
        choices=[
            "all",
            "nle",
            "craftax",
            "crafter",
            "dungeongrid",
            "minigrid",
            "montezuma",
            "pitfall",
        ],
    )
    run_parser.add_argument("--iterations", type=int, default=0)
    run_parser.add_argument("--candidate-root", default="")
    run_parser.add_argument(
        "--candidate-id",
        default="",
        help=(
            "Optional comma-separated candidate id filter under --candidate-root. "
            "Use this in parallel SMR workers so each worker evaluates only its assigned candidate."
        ),
    )
    run_parser.add_argument(
        "--train-seeds",
        default="",
        help="Optional semicolon map, e.g. dungeongrid=101,103;craftax=101,103",
    )
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--max-workers", type=int, default=None)
    run_parser.add_argument("--heldout-seeds-env", default=HELDOUT_ENV)
    run_parser.add_argument("--strict-env", action="store_true")
    run_parser.add_argument(
        "--no-start-services",
        action="store_true",
        help="Disable task-owned service startup and use the EnvSpec service URL directly.",
    )
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    if args.command == "score":
        return score(args)
    raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
