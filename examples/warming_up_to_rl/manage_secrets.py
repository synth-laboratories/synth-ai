#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f".env not found at {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("'").strip('"')
    return env


def write_temp_env(kv: dict[str, str]) -> Path:
    fd, p = tempfile.mkstemp(prefix="modal_secret_", suffix=".env")
    path = Path(p)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for k, v in kv.items():
            fh.write(f"{k}={v}\n")
    return path


def run(cmd: str) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return proc.returncode, proc.stdout


def ensure_secret(secret_name: str, kv: dict[str, str]) -> None:
    if not kv:
        print(f"[skip] {secret_name}: no values provided")
        return
    # Prefer passing KEY=VALUE pairs to avoid Typer --env-file bug under some shells
    kv_args = " ".join([f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in kv.items()])

    # Try plain modal first; fallback to uv run modal
    def _create() -> tuple[int, str]:
        return run(f"modal secret create {shlex.quote(secret_name)} {kv_args}")

    def _delete() -> tuple[int, str]:
        return run(f"printf 'y\n' | modal secret delete {shlex.quote(secret_name)}")

    rc, out = _create()
    if rc != 0:
        # Fallback: use uv run modal
        rc_uv, out_uv = run(f"uv run modal secret create {shlex.quote(secret_name)} {kv_args}")
        if rc_uv == 0:
            print(f"[ok] secret ready: {secret_name}")
            return
        # Try delete+create with both variants
        print(f"[info] create failed for {secret_name}, attempting delete+create…")
        _ = _delete()
        rc2, out2 = _create()
        if rc2 != 0:
            _ = run(f"printf 'y\n' | uv run modal secret delete {shlex.quote(secret_name)}")
            rc3, out3 = run(f"uv run modal secret create {shlex.quote(secret_name)} {kv_args}")
            if rc3 != 0:
                print(out3 or out2 or out_uv or out)
                raise RuntimeError(f"failed to create secret {secret_name}")
    print(f"[ok] secret ready: {secret_name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sync .env keys into Modal secret bundles for the task app"
    )
    ap.add_argument(
        "--env-path", default=str(Path(__file__).parent / ".env"), help="Path to .env with keys"
    )
    args = ap.parse_args()

    env = load_env_file(Path(args.env_path))

    # Secrets used by the task app
    groq_secret = {
        k: v
        for k, v in {
            "GROQ_API_KEY": env.get("GROQ_API_KEY", ""),
            "dev_groq_api_key": env.get("GROQ_API_KEY", ""),
        }.items()
        if v
    }

    openai_secret = {
        k: v
        for k, v in {
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY", ""),
            "dev_openai_api_key": env.get("OPENAI_API_KEY", ""),
        }.items()
        if v
    }

    # Optional: backend key (not mounted by task app today, but useful to keep consistent)
    synth_secret = (
        {"SYNTH_API_KEY": env.get("SYNTH_API_KEY", "")} if env.get("SYNTH_API_KEY") else {}
    )

    env_key = env.get("ENVIRONMENT_API_KEY", "")
    if env_key:
        print(
            "Skipping Modal secret 'crafter-environment-sdk'; the task app now expects "
            "ENVIRONMENT_API_KEY via --env-file so the CLI-minted value stays in sync."
        )
    ensure_secret("groq-api-key", groq_secret)
    ensure_secret("openai-api-key", openai_secret)
    if synth_secret:
        ensure_secret("synth-api-key", synth_secret)

    print("All requested secrets ensured. Redeploy the app if you updated any secrets:")
    print("  uv run modal deploy examples/warming_up_to_rl/task_app/grpo_crafter_task_app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        sys.exit(1)
