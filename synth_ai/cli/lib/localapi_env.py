import os

import click

from synth_ai.core.env import mask_str, resolve_env_var
from synth_ai.core.localapi_state import persist_localapi_key
from synth_ai.core.process import ensure_local_port_available
from synth_ai.core.urls import (
    synth_api_base,
    synth_env_keys_url,
    synth_env_keys_verify_url,
    synth_public_key_url,
)
from synth_ai.core.user_config import load_user_env, update_user_config

__all__ = [
    "ensure_localapi_credentials",
    "ensure_port_free",
    "preflight_localapi_key",
]


def ensure_localapi_credentials(*, require_synth: bool = False, prompt: bool = True) -> None:
    """Ensure required API keys are present in the process environment."""

    load_user_env(override=False)

    localapi_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if prompt and not localapi_key:
        resolve_env_var("ENVIRONMENT_API_KEY")
        localapi_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()

    if localapi_key:
        update_user_config(
            {
                "ENVIRONMENT_API_KEY": localapi_key,
                "DEV_ENVIRONMENT_API_KEY": localapi_key,
            }
        )
        persist_localapi_key(localapi_key)
    elif prompt:
        raise click.ClickException("ENVIRONMENT_API_KEY is required.")

    synth_user_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
    if prompt and (require_synth or not synth_user_key):
        resolve_env_var("SYNTH_API_KEY")
        synth_user_key = (os.environ.get("SYNTH_API_KEY") or "").strip()

    if synth_user_key:
        update_user_config({"SYNTH_API_KEY": synth_user_key})
    elif require_synth and prompt:
        raise click.ClickException("SYNTH_API_KEY is required.")


def ensure_port_free(port: int, host: str, *, force: bool) -> None:
    """Ensure a TCP port is not in use, optionally killing processes when ``force`` is True."""

    if ensure_local_port_available(host, port, force=force):
        return

    message = f"Port {port} is still in use. Stop the running server and try again."
    if force:
        raise click.ClickException(message)
    raise click.ClickException(
        f"Port {port} appears to be in use. Restart with --force to terminate it."
    )


def preflight_localapi_key(*, crash_on_failure: bool = False) -> None:
    """Ensure ENVIRONMENT_API_KEY exists and attempt a Synth registration."""

    ensure_localapi_credentials(require_synth=False, prompt=not crash_on_failure)
    load_user_env(override=False)

    synth_api_base_url = synth_api_base()

    synth_user_key = os.environ.get("SYNTH_API_KEY") or ""
    localapi_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()

    def _mint_key() -> str | None:
        try:
            from synth_ai.sdk.learning.rl.secrets import mint_environment_api_key

            key = mint_environment_api_key()
            os.environ["ENVIRONMENT_API_KEY"] = key
            os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", key)
            update_user_config(
                {
                    "ENVIRONMENT_API_KEY": key,
                    "DEV_ENVIRONMENT_API_KEY": key,
                }
            )
            persist_localapi_key(key)
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
    if not localapi_key:
        localapi_key = _mint_key() or ""
        minted = bool(localapi_key)

    if not localapi_key:
        if crash_on_failure:
            raise click.ClickException(
                "[CRITICAL] ENVIRONMENT_API_KEY missing; run `synth-ai setup` to configure it."
            )
        click.echo("[preflight] ENVIRONMENT_API_KEY missing; continuing without verification.")
        return

    if minted:
        persist_localapi_key(localapi_key)

    if not synth_user_key.strip():
        click.echo("[preflight] SYNTH_API_KEY not set; skipping Synth preflight.")
        return

    try:
        import base64

        import httpx
        from nacl.public import PublicKey, SealedBox
    except Exception:  # pragma: no cover - optional deps
        click.echo("[preflight] Optional crypto dependencies missing; skipping upload.")
        return

    try:
        with httpx.Client(
            timeout=15.0, headers={"Authorization": f"Bearer {synth_user_key}"}
        ) as client:
            click.echo(f"[preflight] synth_api_base={synth_api_base_url}")
            click.echo("[preflight] fetching public key…")
            rpk = client.get(synth_public_key_url())
            if rpk.status_code != 200:
                click.echo(
                    f"[preflight] public key fetch failed with {rpk.status_code}; skipping upload"
                )
                return
            pk = (rpk.json() or {}).get("public_key")
            if not pk:
                click.echo("[preflight] no public key returned; skipping upload")
                return

            pk_bytes = base64.b64decode(pk, validate=True)
            sealed_box = SealedBox(PublicKey(pk_bytes))
            ciphertext = sealed_box.encrypt(localapi_key.encode("utf-8"))
            ct_b64 = base64.b64encode(ciphertext).decode()
            payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ct_b64}

            click.echo(f"[preflight] posting to {synth_env_keys_url()}")
            response = client.post(synth_env_keys_url(), json=payload)
            if 200 <= response.status_code < 300:
                click.echo(
                    f"✅ ENVIRONMENT_API_KEY uploaded successfully ({mask_str(localapi_key)})"
                )
                try:
                    ver = client.get(synth_env_keys_verify_url())
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
            message = f"ENVIRONMENT_API_KEY upload failed with status {response.status_code}" + (
                f" body={snippet}" if snippet else ""
            )
            if crash_on_failure:
                raise click.ClickException(f"[CRITICAL] {message}")
            click.echo(f"[WARN] {message}; proceeding anyway")
    except Exception as exc:  # pragma: no cover - network failures
        message = f"Backend preflight for ENVIRONMENT_API_KEY failed: {exc}"
        if crash_on_failure:
            raise click.ClickException(f"[CRITICAL] {message}") from exc
        click.echo(f"[WARN] {message}; proceeding anyway")
