from __future__ import annotations

import contextlib
import os
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import click
import requests
from synth_ai.utils.cli import print_next_step
from synth_ai.utils.env import mask_str
from synth_ai.utils.modal import is_modal_public_url
from synth_ai.utils.process import popen_capture
from synth_ai.utils.user_config import USER_CONFIG_PATH, update_user_config
from synth_ai.demos import core as demo_core


class HandshakeError(Exception):
    pass


def _get_canonical_origin() -> str:
    """Resolve the dashboard origin for the browser handshake.

    Priority order:
      1. Explicit ``SYNTH_CANONICAL_ORIGIN`` override.
      2. Development flag ``SYNTH_CANONICAL_DEV`` (case-insensitive truthy) → localhost.
      3. Production dashboard at ``https://www.usesynth.ai/dashboard``.
    """

    override = (os.getenv("SYNTH_CANONICAL_ORIGIN") or "").strip()
    if override:
        return override.rstrip("/")

    dev_flag = (os.getenv("SYNTH_CANONICAL_DEV") or "").strip().lower()
    if dev_flag in { "1", "true", "yes", "on" }:
        print("USING DEV ORIGIN")
        return "http://localhost:3000"

    return "https://www.usesynth.ai/dashboard"


def _split_origin(origin: str) -> tuple[str, str]:
    parsed = urlsplit(origin)
    bare = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path.rstrip("/")
    return bare, path


def _ensure_verification_uri(data: dict[str, Any], base_with_path: str) -> None:
    uri = data.get("verification_uri")
    if not isinstance(uri, str) or not uri:
        return
    if uri.startswith("http://") or uri.startswith("https://"):
        return
    data["verification_uri"] = urljoin(base_with_path.rstrip("/") + "/", uri.lstrip("/"))


def _start_handshake_session(origin: str | None = None) -> tuple[str, str, int, int]:
    base = (origin or _get_canonical_origin()).rstrip("/")
    api_origin, _ = _split_origin(base)
    url = urljoin(api_origin.rstrip("/") + "/", "api/sdk/handshake/init")
    r = requests.post(url, timeout=10)
    if r.status_code != 200:
        raise HandshakeError(f"init failed: {r.status_code} {r.text}")
    try:
        data = r.json()
    except ValueError as exc:  # pragma: no cover - network dependent
        raise HandshakeError(f"init returned malformed JSON: {exc}") from exc
    _ensure_verification_uri(data, base)
    return (
        str(data.get("device_code")),
        str(data.get("verification_uri")),
        int(data.get("expires_in", 600)),
        int(data.get("interval", 3)),
    )


def _poll_handshake_token(
    device_code: str, origin: str | None = None, *, timeout_s: int | None = None
) -> dict[str, Any]:
    base = (origin or _get_canonical_origin()).rstrip("/")
    api_origin, _ = _split_origin(base)
    url = urljoin(api_origin.rstrip("/") + "/", "api/sdk/handshake/token")
    deadline = time.time() + (timeout_s or 600)
    while True:
        if time.time() > deadline:
            raise HandshakeError("handshake timed out")
        try:
            r = requests.post(url, json={"device_code": device_code}, timeout=10)
        except Exception:
            time.sleep(2)
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError as exc:  # pragma: no cover - network dependent
                raise HandshakeError(f"token returned malformed JSON: {exc}") from exc
            _ensure_verification_uri(data, base)
            return data
        elif r.status_code in (404, 410):
            raise HandshakeError(f"handshake failed: {r.status_code}")
        # 428 authorization_pending or others → wait and retry
        time.sleep(2)


def _run_handshake(origin: str | None = None) -> dict[str, Any]:
    device_code, verification_uri, expires_in, interval = _start_handshake_session(origin)
    with contextlib.suppress(Exception):
        webbrowser.open(verification_uri)
    return _poll_handshake_token(device_code, origin, timeout_s=expires_in)



def setup() -> int:
    # Prefer the demo directory provided in the current shell session, then fall back to persisted state
    demo_dir_env = (os.environ.get("DEMO_DIR") or "").strip()
    demo_dir: str | None = None
    if demo_dir_env:
        candidate = Path(demo_dir_env).expanduser()
        if candidate.is_dir():
            demo_dir = str(candidate.resolve())
        else:
            print(f"Warning: DEMO_DIR={demo_dir_env} does not exist; falling back to stored demo directory.")

    if demo_dir is None:
        loaded = demo_core.load_demo_dir()
        if loaded:
            demo_dir = loaded

    if demo_dir and os.path.isdir(demo_dir):
        os.chdir(demo_dir)
        print(f"Using demo directory: {demo_dir}")

    synth_key = ""
    rl_env_key = ""
    org_name = ""

    try:
        print("\n⏳ Connecting to your browser session…")
        res = _run_handshake()
        org = res.get("org") or {}
        keys = res.get("keys") or {}
        synth_key = str(keys.get("synth") or "").strip()
        rl_env_key = str(keys.get("rl_env") or "").strip()
        org_name = org.get("name") or "Unamed Organization ™️"
        print(f"✅ Connected to {org_name}!")
    except (HandshakeError, Exception) as exc:
        print(f"⚠️  Failed to fetch keys from frontend: {exc}")
        print("Falling back to manual entry...")

    if not synth_key:
        try:
            synth_key = input(
                "Failed to fetch your Synth API key. Please enter your Synth API key here:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return 1
        if not synth_key:
            print("Synth API key is required.")
            return 1

    if not rl_env_key:
        try:
            rl_env_key = input(
                "Failed to fetch your Environment API key. Please enter your Environment API key here:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return 1
        if not rl_env_key:
            print("Environment API key is required.")
            return 1

    # Persist keys to user config
    config_updates = {
        "SYNTH_API_KEY": synth_key,
        "ENVIRONMENT_API_KEY": rl_env_key,
    }
    update_user_config(config_updates)

    os.environ["SYNTH_API_KEY"] = synth_key
    os.environ["ENVIRONMENT_API_KEY"] = rl_env_key

    env = demo_core.load_env()

    def _refresh_env() -> None:
        nonlocal env
        env = demo_core.load_env()

    def _maybe_fix_task_url() -> None:
        if not env.task_app_name:
            return
        current = env.task_app_base_url
        needs_lookup = not current or not is_modal_public_url(current)
        if not needs_lookup:
            return
        code, out = popen_capture(
            [
                "uv",
                "run",
                "python",
                "-m",
                "modal",
                "app",
                "url",
                env.task_app_name,
            ]
        )
        if code != 0 or not out:
            return
        new_url = ""
        for token in out.split():
            if is_modal_public_url(token):
                new_url = token.strip().rstrip("/")
                break
        if new_url and new_url != current:
            print(f"Updating TASK_APP_BASE_URL from Modal CLI → {new_url}")
            persist_path = demo_dir or os.getcwd()
            demo_core.persist_task_url(new_url, name=env.task_app_name, path=persist_path)
            os.environ["TASK_APP_BASE_URL"] = new_url
            _refresh_env()

    modal_ok, modal_msg = demo_core.modal_auth_status()
    if modal_ok:
        print(f"✓ Modal authenticated: {modal_msg}")
    else:
        print(f"[setup] Modal authentication status: {modal_msg}")

    _maybe_fix_task_url()

    if env.dev_backend_url:
        api = env.dev_backend_url.rstrip("/") + (
            "" if env.dev_backend_url.endswith("/api") else "/api"
        )
        demo_core.assert_http_ok(api + "/health", method="GET")
    if env.task_app_base_url:
        base = env.task_app_base_url.rstrip("/")
        demo_core.assert_http_ok(
            base + "/health", method="GET"
        ) or demo_core.assert_http_ok(
            base, method="GET"
        )
    print("\nSaved keys:")
    print(f"  SYNTH_API_KEY={mask_str(synth_key)}")
    print(f"  ENVIRONMENT_API_KEY={mask_str(rl_env_key)}")
    if env.task_app_base_url:
        print(f"  TASK_APP_BASE_URL={env.task_app_base_url}")
    print(f"Configuration persisted to: {USER_CONFIG_PATH}")

    demo_core.persist_demo_dir(os.getcwd())

    print_next_step("deploy our task app", ["uvx synth-ai deploy"])
    return 0


def register(group):
    @group.command("setup")
    def demo_setup():
        code = setup()
        if code:
            raise click.exceptions.Exit(code)
