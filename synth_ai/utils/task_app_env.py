import os
import click

from .base_url import PROD_BASE_URL_DEFAULT
from .env import mask_str, resolve_env_var
from .process import ensure_local_port_available
from .task_app_state import persist_env_api_key
from .user_config import load_user_env, update_user_config

__all__ = [
    "ensure_env_credentials",
    "ensure_port_free",
    "preflight_env_key",
]


def ensure_env_credentials(*, require_synth: bool = False, prompt: bool = True) -> None:
    """Ensure required API keys are present in the process environment."""

    load_user_env(override=False)

    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if prompt and not env_key:
        resolve_env_var("ENVIRONMENT_API_KEY")
        env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()

    if env_key:
        update_user_config(
            {
                "ENVIRONMENT_API_KEY": env_key,
                "DEV_ENVIRONMENT_API_KEY": env_key,
            }
        )
        persist_env_api_key(env_key)
    elif prompt:
        raise click.ClickException("ENVIRONMENT_API_KEY is required.")

    synth_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
    if prompt and (require_synth or not synth_key):
        resolve_env_var("SYNTH_API_KEY")
        synth_key = (os.environ.get("SYNTH_API_KEY") or "").strip()

    if synth_key:
        update_user_config({"SYNTH_API_KEY": synth_key})
    elif require_synth and prompt:
        raise click.ClickException("SYNTH_API_KEY is required.")


def ensure_port_free(port: int, host: str, *, force: bool) -> None:
    """Ensure a TCP port is not in use, optionally killing processes when ``force`` is True."""

    if ensure_local_port_available(host, port, force=force):
        return

    message = f"Port {port} is still in use. Stop the running server and try again."
    if force:
        raise click.ClickException(message)
    raise click.ClickException(f"Port {port} appears to be in use. Restart with --force to terminate it.")


def preflight_env_key(*, crash_on_failure: bool = False) -> None:
    """Ensure ENVIRONMENT_API_KEY exists and attempt a backend registration."""

    ensure_env_credentials(require_synth=False, prompt=not crash_on_failure)
    load_user_env(override=False)

    raw_backend = (
        os.environ.get("BACKEND_BASE_URL")
        or os.environ.get("SYNTH_BASE_URL")
        or f"{PROD_BASE_URL_DEFAULT}/api"
    )
    backend_base = raw_backend.rstrip("/")
    if not backend_base.endswith("/api"):
        backend_base += "/api"

    synth_key = os.environ.get("SYNTH_API_KEY") or ""
    env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()

    def _mint_key() -> str | None:
        try:
            from synth_ai.learning.rl.secrets import mint_environment_api_key

            key = mint_environment_api_key()
            os.environ["ENVIRONMENT_API_KEY"] = key
            os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", key)
            update_user_config(
                {
                    "ENVIRONMENT_API_KEY": key,
                    "DEV_ENVIRONMENT_API_KEY": key,
                }
            )
            persist_env_api_key(key)
            click.echo(f"[preflight] minted ENVIRONMENT_API_KEY ({mask_str(key)})")
            return key
        except Exception as exc:  # pragma: no cover - defensive fallback
            if crash_on_failure:
                raise click.ClickException(
                    f"[CRITICAL] Failed to mint ENVIRONMENT_API_KEY: {exc}"
                ) from exc
            click.echo(
                f"[WARN] Failed to mint ENVIRONMENT_API_KEY automatically ({exc}); proceeding without upload"
            )
            return None

    minted = False
    if not env_api_key:
        env_api_key = _mint_key() or ""
        minted = bool(env_api_key)

    if not env_api_key:
        if crash_on_failure:
            raise click.ClickException(
                "[CRITICAL] ENVIRONMENT_API_KEY missing; run `synth-ai setup` to configure it."
            )
        click.echo("[preflight] ENVIRONMENT_API_KEY missing; continuing without verification.")
        return

    if minted:
        persist_env_api_key(env_api_key)

    if not synth_key.strip():
        click.echo("[preflight] SYNTH_API_KEY not set; skipping backend preflight.")
        return

    try:
        import base64

        import httpx
        from nacl.public import PublicKey, SealedBox
    except Exception:  # pragma: no cover - optional deps
        click.echo("[preflight] Optional crypto dependencies missing; skipping upload.")
        return

    try:
        with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {synth_key}"}) as client:
            click.echo(f"[preflight] backend={backend_base}")
            click.echo("[preflight] fetching public key…")
            rpk = client.get(f"{backend_base.rstrip('/')}/v1/crypto/public-key")
            if rpk.status_code != 200:
                click.echo(f"[preflight] public key fetch failed with {rpk.status_code}; skipping upload")
                return
            pk = (rpk.json() or {}).get("public_key")
            if not pk:
                click.echo("[preflight] no public key returned; skipping upload")
                return

            pk_bytes = base64.b64decode(pk, validate=True)
            sealed_box = SealedBox(PublicKey(pk_bytes))
            ciphertext = sealed_box.encrypt(env_api_key.encode("utf-8"))
            ct_b64 = base64.b64encode(ciphertext).decode()
            payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ct_b64}

            click.echo(f"[preflight] posting to {backend_base.rstrip('/')}/v1/env-keys")
            response = client.post(f"{backend_base.rstrip('/')}/v1/env-keys", json=payload)
            if 200 <= response.status_code < 300:
                click.echo(
                    f"✅ ENVIRONMENT_API_KEY uploaded successfully ({mask_str(env_api_key)})"
                )
                try:
                    ver = client.get(f"{backend_base.rstrip('/')}/v1/env-keys/verify")
                    if ver.status_code == 200 and (ver.json() or {}).get("present"):
                        click.echo("✅ Key verified in backend")
                    else:
                        click.echo(
                            f"⚠️  Verification returned {ver.status_code}, but upload succeeded - proceeding"
                        )
                except Exception as verify_err:  # pragma: no cover - verification optional
                    click.echo(
                        f"⚠️  Verification check failed ({verify_err}), but upload succeeded - proceeding"
                    )
                return

            snippet = response.text[:400] if response.text else ""
            message = (
                f"ENVIRONMENT_API_KEY upload failed with status {response.status_code}"
                + (f" body={snippet}" if snippet else "")
            )
            if crash_on_failure:
                raise click.ClickException(f"[CRITICAL] {message}")
            click.echo(f"[WARN] {message}; proceeding anyway")
    except Exception as exc:  # pragma: no cover - network failures
        message = f"Backend preflight for ENVIRONMENT_API_KEY failed: {exc}"
        if crash_on_failure:
            raise click.ClickException(f"[CRITICAL] {message}") from exc
        click.echo(f"[WARN] {message}; proceeding anyway")
