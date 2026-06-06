#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

TASK_ID = "nanohorizon_craftax_hello_world"
ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "runs" / TASK_ID
TASK_TOML_PATH = ROOT / "task.toml"
WORKER_SCRIPT = Path(__file__).resolve().parent / "nanohorizon_craftax_hello_world_worker.py"

ARTIFACTS = {
    "eval_summary": "artifacts/eval_summary.json",
    "rollouts": "artifacts/rollouts.jsonl",
    "result_manifest": "artifacts/result_manifest.json",
    "scorecard": "artifacts/craftax_scorecard.json",
    "rollout_media_manifest": "artifacts/craftax_rollout_media.json",
    "verifier_review": "artifacts/verifier_review.json",
    "reportbench_output": "artifacts/reportbench_output.json",
    "leaderboard_evidence": "artifacts/open_research_leaderboard_evidence.json",
    "reproduction": "reports/reproduction.md",
    "runner_stdout": "artifacts/runner_stdout.log",
    "runner_stderr": "artifacts/runner_stderr.log",
}
LEADERBOARD_EVIDENCE_SCHEMA = "synth.open_research.leaderboard_evidence.v1"
ROLLOUT_MEDIA_SCHEMA = "synth.open_research.craftax_rollout_media.v1"
SCORECARD_SCHEMA = "synth.open_research.craftax_scorecard.v1"
DEFAULT_POLICY_MODEL = "openai/gpt-5.4-nano"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _looks_like_live_smr_workspace(path: Path) -> bool:
    return (path / "starting-data").exists()


def _default_output_root() -> Path:
    cwd = Path.cwd().resolve()
    if _looks_like_live_smr_workspace(cwd):
        return cwd
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RUNS_ROOT / stamp


def _resolve_output_root(raw: str | None) -> Path:
    if not raw:
        return _default_output_root()
    candidate = Path(raw).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if _looks_like_live_smr_workspace(cwd) and not str(candidate).startswith(str(cwd)):
        return cwd
    return candidate


def _artifact_path(output_root: Path, key: str) -> Path:
    return output_root / ARTIFACTS[key]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _first_gif_data_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("data:image/gif;base64,"):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_gif_data_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _first_gif_data_url(item)
            if found:
                return found
    return None


def _write_gif_from_data_url(path: Path, data_url: str) -> bool:
    _, _, payload = data_url.partition(",")
    if not payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(payload))
    return True


def _first_existing_gif_path(value: Any) -> Path | None:
    if isinstance(value, dict):
        gif_path = value.get("gif_path")
        if isinstance(gif_path, str) and gif_path.strip():
            candidate = Path(gif_path).expanduser()
            if candidate.exists() and candidate.is_file():
                return candidate
        for item in value.values():
            found = _first_existing_gif_path(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _first_existing_gif_path(item)
            if found:
                return found
    return None


def _first_existing_media_paths(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, dict):
        for key in ("gif_path", "mp4_path", "video_path", "media_path", "path"):
            raw_path = value.get(key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path).expanduser()
            if candidate.exists() and candidate.is_file():
                suffix = candidate.suffix.lower()
                if suffix in {".gif", ".png", ".jpg", ".jpeg", ".mp4", ".webm"}:
                    paths.append(candidate)
        for item in value.values():
            paths.extend(_first_existing_media_paths(item))
    if isinstance(value, list):
        for item in value:
            paths.extend(_first_existing_media_paths(item))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _media_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".webm":
        return "video/webm"
    return "application/octet-stream"


def _media_kind(path: Path) -> str:
    return "video" if _media_content_type(path).startswith("video/") else "image"


def _copy_gif_from_path(path: Path, source: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return True


def _rollout_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata:
                return metadata[key]
    reward_info = row.get("reward_info")
    if isinstance(reward_info, dict):
        for key in keys:
            if key in reward_info:
                return reward_info[key]
    return None


def _rollout_achievements(row: dict[str, Any]) -> list[str]:
    value = _rollout_value(row, ("achievements", "unique_achievements"))
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _write_leaderboard_evidence(output_root: Path) -> None:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    rows = _load_jsonl(_artifact_path(output_root, "rollouts"))
    rollouts: list[dict[str, Any]] = []
    media_entries: list[dict[str, Any]] = []
    media_rollout_count = 0
    for index, row in enumerate(rows):
        rollout_id = str(_rollout_value(row, ("rollout_id", "id")) or f"rollout-{index + 1}")
        data_url = _first_gif_data_url(row)
        media_sources = _first_existing_media_paths(row)
        gif_source = _first_existing_gif_path(row)
        if gif_source is not None and gif_source not in media_sources:
            media_sources.insert(0, gif_source)
        media: list[dict[str, Any]] = []
        reward = _rollout_value(
            row,
            ("outcome_reward", "reward", "total_reward", "aggregate_reward"),
        )
        seed = _rollout_value(row, ("seed", "task_seed"))
        for source in media_sources:
            suffix = source.suffix.lower()
            media_path = f"artifacts/rollout_media/rollout_{index + 1:02d}{suffix}"
            target = output_root / media_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entry = {
                "path": media_path,
                "kind": _media_kind(target),
                "content_type": _media_content_type(target),
                "label": f"Rollout {index + 1} gameplay {target.suffix.upper().lstrip('.')}",
            }
            media.append(entry)
            media_entries.append(
                {
                    **entry,
                    "rollout_id": rollout_id,
                    "seed": seed,
                    "reward": reward,
                }
            )
        if not media and data_url:
            media_path = f"artifacts/rollout_media/rollout_{index + 1:02d}.gif"
            if _write_gif_from_data_url(output_root / media_path, data_url):
                entry = {
                    "path": media_path,
                    "kind": "image",
                    "content_type": "image/gif",
                    "label": f"Rollout {index + 1} gameplay GIF",
                }
                media.append(entry)
                media_entries.append(
                    {
                        **entry,
                        "rollout_id": rollout_id,
                        "seed": seed,
                        "reward": reward,
                    }
                )
        if media:
            media_rollout_count += 1
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "seed": seed,
                "reward": reward,
                "outcome_reward": reward,
                "success_status": str(row.get("status") or "observed"),
                "achievements": _rollout_achievements(row),
                "media": media,
                "media_status": "captured" if media else "not_captured",
            }
        )
    if rows and media_rollout_count < 1:
        raise RuntimeError(
            "Craftax rollout media must include at least one captured gameplay artifact; "
            "no rollout rows contained a supported media path or GIF data URL."
        )
    payload = {
        "schema": LEADERBOARD_EVIDENCE_SCHEMA,
        "application_id": "craftax",
        "track": "craftax",
        "result_metric": "aggregate_reward",
        "requested_rollouts": summary.get("requested_rollouts"),
        "completed_rollouts": summary.get("num_rollouts") or len(rows),
        "media_rollout_count": media_rollout_count,
        "aggregate_reward": summary.get("mean_outcome_reward"),
        "rollouts": rollouts,
    }
    _write_json(_artifact_path(output_root, "leaderboard_evidence"), payload)
    _write_json(
        _artifact_path(output_root, "rollout_media_manifest"),
        {
            "schema": ROLLOUT_MEDIA_SCHEMA,
            "application_id": "craftax",
            "track": "craftax",
            "requested_rollouts": summary.get("requested_rollouts"),
            "completed_rollouts": summary.get("num_rollouts") or len(rows),
            "captured_rollout_count": media_rollout_count,
            "media_count": len(media_entries),
            "media": media_entries,
        },
    )


def _write_scorecard(output_root: Path) -> None:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    rows = _load_jsonl(_artifact_path(output_root, "rollouts"))
    rollout_summary = summary.get("rollout_summary")
    if not isinstance(rollout_summary, dict):
        rollout_summary = {}
    _write_json(
        _artifact_path(output_root, "scorecard"),
        {
            "schema": SCORECARD_SCHEMA,
            "application_id": "craftax",
            "track": "craftax",
            "metric": "aggregate_reward",
            "metric_direction": "higher_is_better",
            "aggregate_reward": summary.get("mean_outcome_reward"),
            "mean_reward": summary.get("mean_outcome_reward"),
            "max_reward": summary.get("max_outcome_reward"),
            "sample_size": summary.get("num_rollouts") or len(rows),
            "requested_rollouts": summary.get("requested_rollouts"),
            "requested_total_llm_calls": summary.get("requested_total_llm_calls"),
            "max_steps_per_rollout": summary.get("requested_max_steps_per_rollout"),
            "rollout_concurrency": summary.get("requested_rollout_concurrency"),
            "num_errors": summary.get("num_errors"),
            "runner_exit_code": manifest.get("runner_exit_code"),
            "rollout_execution_mode": rollout_summary.get("rollout_execution_mode"),
        },
    )


def _read_task_config() -> dict[str, Any]:
    with TASK_TOML_PATH.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected TOML table in {TASK_TOML_PATH}")
    return payload


def _resolve_nanohorizon_root() -> Path:
    explicit = str(os.getenv("NANOHORIZON_REPO_ROOT") or "").strip()
    cwd = Path.cwd().resolve()
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        cwd,
        cwd / "project",
        Path("/Users/joshpurtell/Documents/GitHub/nanohorizon"),
        Path.home() / "Documents" / "GitHub" / "nanohorizon",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and (candidate / "src" / "nanohorizon").exists():
            return candidate.resolve()
    return Path(__file__).resolve().parent


def _run_baseline(output_root: Path) -> subprocess.CompletedProcess[str]:
    nanohorizon_root = _resolve_nanohorizon_root()
    summary_path = _artifact_path(output_root, "eval_summary")
    rollouts_path = _artifact_path(output_root, "rollouts")
    stdout_path = _artifact_path(output_root, "runner_stdout")
    stderr_path = _artifact_path(output_root, "runner_stderr")
    env = os.environ.copy()
    pythonpath_parts = [
        str(nanohorizon_root),
        str(nanohorizon_root / "src"),
    ]
    existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["NANOHORIZON_REPO_ROOT"] = str(nanohorizon_root)
    env.setdefault("NANOHORIZON_MAX_STEPS", "500")
    env.setdefault("NANOHORIZON_ROLLOUT_CONCURRENCY", "10")
    env.setdefault(
        "NANOHORIZON_VIDEO_CAPTURE_OUTPUT_DIR",
        str(output_root / "artifacts" / "craftax_runtime_media"),
    )
    env.setdefault("NANOHORIZON_VIDEO_CAPTURE_ROLLOUT_INDEX", "0")
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--summary-output",
        str(summary_path),
        "--rollouts-output",
        str(rollouts_path),
    ]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_file:
        process = subprocess.run(
            cmd,
            cwd=str(output_root),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    return subprocess.CompletedProcess(
        args=process.args,
        returncode=process.returncode,
        stdout=stdout_path.read_text(encoding="utf-8", errors="replace"),
        stderr=stderr_path.read_text(encoding="utf-8", errors="replace"),
    )


def _write_result_manifest(output_root: Path, *, process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    requested_rollouts = int(summary.get("requested_rollouts") or 0)
    completed_rollouts = int(summary.get("num_rollouts") or 0)
    num_errors = int(summary.get("num_errors") or 0)
    status = (
        "succeeded"
        if process.returncode == 0 and num_errors == 0 and completed_rollouts >= requested_rollouts
        else "failed"
    )
    manifest = {
        "task_id": TASK_ID,
        "completed_at": _utc_now(),
        "status": status,
        "runner_exit_code": int(process.returncode),
        "rollout_validation_error": (
            None
            if status == "succeeded"
            else (
                f"requested_rollouts={requested_rollouts}, "
                f"completed_rollouts={completed_rollouts}, num_errors={num_errors}"
            )
        ),
        "model": summary.get("model"),
        "mean_outcome_reward": summary.get("mean_outcome_reward"),
        "requested_rollouts": summary.get("requested_rollouts"),
        "requested_total_llm_calls": summary.get("requested_total_llm_calls"),
        "requested_max_steps_per_rollout": summary.get("requested_max_steps_per_rollout"),
        "requested_rollout_concurrency": summary.get("requested_rollout_concurrency"),
        "requested_env_batch_size": summary.get("requested_env_batch_size"),
        "env_batch_size": (summary.get("rollout_summary") or {}).get("env_batch_size")
        if isinstance(summary.get("rollout_summary"), dict)
        else None,
        "rollouts_path": ARTIFACTS["rollouts"],
        "summary_path": ARTIFACTS["eval_summary"],
        "stdout_preview": process.stdout[:2000],
        "stderr_preview": process.stderr[:2000],
    }
    _write_json(_artifact_path(output_root, "result_manifest"), manifest)
    return manifest


def _write_reproduction_report(output_root: Path) -> None:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    report = (
        "# NanoHorizon Craftax Hello World\n\n"
        f"- Completed at: `{_utc_now()}`\n"
        f"- Model: `{summary.get('model')}`\n"
        f"- Task: `{summary.get('task')}`\n"
        f"- Requested trajectories: `{summary.get('requested_rollouts')}`\n"
        f"- Requested total LLM calls: `{summary.get('requested_total_llm_calls')}`\n"
        f"- Requested LLM calls per rollout cap: `{summary.get('requested_llm_calls_per_rollout')}`\n"
        f"- Requested rollout concurrency: `{summary.get('requested_rollout_concurrency')}`\n"
        f"- Mean reward: `{summary.get('mean_outcome_reward')}`\n"
        f"- Max reward: `{summary.get('max_outcome_reward')}`\n"
        f"- Mean LLM calls per rollout: `{summary.get('mean_llm_calls_per_rollout')}`\n"
        f"- Errors: `{summary.get('num_errors')}`\n"
    )
    output_path = _artifact_path(output_root, "reproduction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _compute_verifier_review(output_root: Path, verifier_mode: str) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    report_exists = _artifact_path(output_root, "reproduction").exists()
    rollouts_exists = _artifact_path(output_root, "rollouts").exists()
    score = 1.0
    notes: list[str] = []
    if manifest.get("status") != "succeeded":
        score = 0.0
        notes.append("runner did not succeed")
    if not report_exists:
        score = min(score, 0.2)
        notes.append("report missing")
    if not rollouts_exists:
        score = min(score, 0.2)
        notes.append("rollout evidence missing")
    if int(summary.get("requested_rollouts", 0) or 0) < 1:
        score = min(score, 0.2)
        notes.append("requested_rollouts was not positive")
    if int(summary.get("requested_rollout_concurrency", 0) or 0) < 1:
        score = min(score, 0.2)
        notes.append("requested_rollout_concurrency was not positive")
    return {
        "score": round(float(score), 6),
        "summary": "Derived from the concrete NanoHorizon Craftax hello-world artifacts.",
        "criteria": [
            {"id": "artifact_completeness", "score": 1.0 if report_exists and rollouts_exists else 0.0, "weight": 0.30},
            {"id": "reward_grounding", "score": 1.0 if "mean_outcome_reward" in summary else 0.0, "weight": 0.30},
            {"id": "rollout_evidence", "score": 1.0 if rollouts_exists else 0.0, "weight": 0.20},
            {"id": "report_grounding", "score": 1.0 if report_exists else 0.0, "weight": 0.20},
        ],
        "notes": notes or ["bundle is grounded in the concrete rollout summary and rollout records"],
        "verifier_mode": verifier_mode,
    }


def _build_reportbench_output(output_root: Path, verifier_mode: str) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    verifier = _load_json(_artifact_path(output_root, "verifier_review"))
    task_cfg = _read_task_config()
    return {
        "task_id": TASK_ID,
        "state": manifest.get("status"),
        "primary_metric": "mean_outcome_reward",
        "primary_score": summary.get("mean_outcome_reward"),
        "reward": summary.get("mean_outcome_reward"),
        "verifier_score": verifier.get("score"),
        "model": summary.get("model"),
        "task": summary.get("task"),
        "requested_rollouts": summary.get("requested_rollouts"),
        "requested_total_llm_calls": summary.get("requested_total_llm_calls"),
        "requested_max_steps_per_rollout": summary.get("requested_max_steps_per_rollout"),
        "requested_rollout_concurrency": summary.get("requested_rollout_concurrency"),
        "mean_llm_calls_per_rollout": summary.get("mean_llm_calls_per_rollout"),
        "report_path": ARTIFACTS["reproduction"],
        "rollouts_path": ARTIFACTS["rollouts"],
        "result_manifest_path": ARTIFACTS["result_manifest"],
        "task_title": ((task_cfg.get("task") or {}).get("title") if isinstance(task_cfg.get("task"), dict) else None),
        "verifier_mode": verifier_mode,
        "completed_at": _utc_now(),
    }


def run(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    process = _run_baseline(output_root)
    if process.returncode != 0:
        summary_path = _artifact_path(output_root, "eval_summary")
        if not summary_path.exists():
            _write_json(
                summary_path,
                {
                    "benchmark": TASK_ID,
                    "task": "craftax",
                    "model": str(os.getenv("NANOHORIZON_MODEL") or DEFAULT_POLICY_MODEL),
                    "requested_rollouts": int(os.getenv("NANOHORIZON_ROLLOUTS") or "1"),
                    "requested_total_llm_calls": int(os.getenv("NANOHORIZON_ROLLOUTS") or "1"),
                    "requested_max_steps_per_rollout": int(os.getenv("NANOHORIZON_MAX_STEPS") or "500"),
                    "requested_llm_calls_per_rollout": 1,
                    "requested_rollout_concurrency": int(os.getenv("NANOHORIZON_ROLLOUT_CONCURRENCY") or "10"),
                    "requested_env_batch_size": int(os.getenv("NANOHORIZON_ENV_BATCH_SIZE") or "5"),
                    "mean_outcome_reward": 0.0,
                    "max_outcome_reward": 0.0,
                    "mean_llm_calls_per_rollout": 0.0,
                    "num_rollouts": 0,
                    "num_errors": int(os.getenv("NANOHORIZON_ROLLOUTS") or "1"),
                    "runner_failure": {
                        "exit_code": int(process.returncode),
                        "stdout": process.stdout[:2000],
                        "stderr": process.stderr[:2000],
                    },
                },
            )
        rollouts_path = _artifact_path(output_root, "rollouts")
        if not rollouts_path.exists():
            rollouts_path.parent.mkdir(parents=True, exist_ok=True)
            rollouts_path.write_text("", encoding="utf-8")
    manifest = _write_result_manifest(output_root, process=process)
    if manifest.get("status") != "succeeded":
        _write_scorecard(output_root)
        _write_reproduction_report(output_root)
        raise RuntimeError(
            "Craftax runner did not produce a valid full rollout set: "
            f"{manifest.get('rollout_validation_error')}"
        )
    _write_leaderboard_evidence(output_root)
    _write_scorecard(output_root)
    _write_reproduction_report(output_root)
    return int(process.returncode)


def score(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root)
    verifier = _compute_verifier_review(output_root, args.verifier_mode)
    _write_json(_artifact_path(output_root, "verifier_review"), verifier)
    output = _build_reportbench_output(output_root, args.verifier_mode)
    _write_json(_artifact_path(output_root, "reportbench_output"), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon Craftax hello-world reportbench task.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", default="")
    run_parser.set_defaults(func=run)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-root", default="")
    score_parser.add_argument("--verifier-mode", default="precheck")
    score_parser.set_defaults(func=score)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
