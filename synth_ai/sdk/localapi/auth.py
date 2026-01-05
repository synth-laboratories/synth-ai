"""Local API authentication helpers."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Iterable


from synth_ai.core.env import PROD_BASE_URL_DEFAULT, get_backend_from_env
from synth_ai.core.paths import REPO_ROOT, get_env_file_paths
from synth_ai.core.user_config import update_user_config

ENVIRONMENT_API_KEY_NAME = "ENVIRONMENT_API_KEY"
DEV_ENVIRONMENT_API_KEY_NAME = "DEV_ENVIRONMENT_API_KEY"
MAX_ENVIRONMENT_API_KEY_BYTES = 8 * 1024
_ALGORITHM = "libsodium.sealedbox.v1"
_ENV_SYNTH_PATH = Path.home() / ".env.synth"

__all__ = [
    "ENVIRONMENT_API_KEY_NAME",
    "DEV_ENVIRONMENT_API_KEY_NAME",
    "MAX_ENVIRONMENT_API_KEY_BYTES",
    "encrypt_for_backend",
    "ensure_localapi_auth",
    "mint_environment_api_key",
    "setup_environment_api_key",
]


def mint_environment_api_key() -> str:
    """Mint a random ENVIRONMENT_API_KEY value."""

    return secrets.token_hex(32)


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return value[:idx].rstrip()
    return value.rstrip()


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.lower().startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key_part, value_part = stripped.split("=", 1)
    key = key_part.strip()
    if not key:
        return None
    value_candidate = _strip_inline_comment(value_part.strip())
    if not value_candidate:
        return key, ""
    if (
        len(value_candidate) >= 2
        and value_candidate[0] in {'"', "'"}
        and value_candidate[-1] == value_candidate[0]
    ):
        value = value_candidate[1:-1]
    else:
        value = value_candidate
    return key, value


def _read_env_var_from_file(key: str, path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = _parse_env_assignment(line)
                if parsed is None:
                    continue
                parsed_key, value = parsed
                if parsed_key == key:
                    return value
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _merge_env_content(original: str, updates: dict[str, str]) -> str:
    lines = original.splitlines(keepends=True)
    seen: set[str] = set()
    newline_default = "\n"
    if lines and lines[-1].endswith("\r\n"):
        newline_default = "\r\n"

    merged: list[str] = []
    for line in lines:
        stripped = line.rstrip("\r\n")
        parsed = _parse_env_assignment(stripped)
        key = parsed[0] if parsed else None
        if key and key in updates and key not in seen:
            end = "\r\n" if line.endswith("\r\n") else "\n"
            merged.append(f"{key}={updates[key]}{end}")
            seen.add(key)
        else:
            merged.append(line)

    for key, val in updates.items():
        if key not in seen:
            merged.append(f"{key}={val}{newline_default}")

    if merged and not merged[-1].endswith(("\n", "\r\n")):
        merged[-1] = merged[-1] + newline_default

    return "".join(merged)


def _write_env_var_to_file(path: Path, key: str, value: str) -> None:
    existing = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    merged = _merge_env_content(existing, {key: value})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged, encoding="utf-8")


def _iter_env_files() -> Iterable[Path]:
    cwd = Path.cwd()
    repo_root = REPO_ROOT
    seen: set[Path] = set()

    for base in (cwd, repo_root):
        root_env = base / ".env"
        if root_env.is_file():
            resolved = root_env.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved

    for base in (cwd, repo_root):
        for path in get_env_file_paths(base):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _resolve_env_api_key_from_files() -> str | None:
    for path in _iter_env_files():
        value = _read_env_var_from_file(ENVIRONMENT_API_KEY_NAME, path)
        if value:
            return value.strip()
    return None


def _resolve_backend_base(backend_base: str | None) -> str:
    if backend_base:
        backend = backend_base.rstrip("/")
    else:
        env_value = (
            os.environ.get("SYNTH_API_BASE")
            or os.environ.get("SYNTH_BASE_URL")
            or os.environ.get("BACKEND_BASE_URL")
        )
        if env_value:
            backend = env_value.rstrip("/")
        else:
            base, _ = get_backend_from_env()
            backend = base.rstrip("/")
            if not backend:
                backend = PROD_BASE_URL_DEFAULT.rstrip("/")
    if not backend.endswith("/api"):
        backend = f"{backend}/api"
    return backend


def ensure_localapi_auth(
    backend_base: str | None = None,
    synth_api_key: str | None = None,
    *,
    upload: bool = True,
    persist: bool | None = None,
) -> str:
    """Ensure ENVIRONMENT_API_KEY is present and optionally registered."""

    key = (os.environ.get(ENVIRONMENT_API_KEY_NAME) or "").strip()
    if not key:
        key = (os.environ.get(DEV_ENVIRONMENT_API_KEY_NAME) or "").strip()
        if key:
            os.environ[ENVIRONMENT_API_KEY_NAME] = key

    if not key:
        key = _resolve_env_api_key_from_files() or ""
        if key:
            os.environ[ENVIRONMENT_API_KEY_NAME] = key

    if not key:
        key = _read_env_var_from_file(ENVIRONMENT_API_KEY_NAME, _ENV_SYNTH_PATH) or ""
        if key:
            os.environ[ENVIRONMENT_API_KEY_NAME] = key

    minted = False
    if not key:
        key = mint_environment_api_key()
        if not key:
            raise RuntimeError("Failed to mint ENVIRONMENT_API_KEY")
        os.environ[ENVIRONMENT_API_KEY_NAME] = key
        minted = True

    os.environ.setdefault(DEV_ENVIRONMENT_API_KEY_NAME, key)

    if persist is None:
        persist = os.environ.get("SYNTH_LOCALAPI_AUTH_PERSIST", "1") != "0"

    if minted and persist:
        _write_env_var_to_file(_ENV_SYNTH_PATH, ENVIRONMENT_API_KEY_NAME, key)
        update_user_config(
            {
                ENVIRONMENT_API_KEY_NAME: key,
                DEV_ENVIRONMENT_API_KEY_NAME: key,
            }
        )

    if upload:
        if synth_api_key is None:
            synth_api_key = os.environ.get("SYNTH_API_KEY")
        if synth_api_key:
            backend = _resolve_backend_base(backend_base)
            try:
                setup_environment_api_key(backend, synth_api_key, token=key)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to upload ENVIRONMENT_API_KEY: %s", exc
                )

    if not key:
        raise RuntimeError("ENVIRONMENT_API_KEY is required but missing")

    return key


def encrypt_for_backend(pubkey_b64: str, secret: str | bytes) -> str:
    from nacl.public import PublicKey, SealedBox

    if not isinstance(pubkey_b64, str) or not pubkey_b64.strip():
        raise ValueError("public key must be a non-empty base64 string")

    try:
        key_bytes = base64.b64decode(pubkey_b64, validate=True)
    except binascii.Error as exc:
        raise ValueError("public key must be base64-encoded") from exc

    if len(key_bytes) != 32:
        raise ValueError("public key must be 32 bytes for X25519")

    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        secret_bytes = secret
    else:
        raise TypeError("secret must be str or bytes")

    if not secret_bytes:
        raise ValueError("secret must not be empty")

    box = SealedBox(PublicKey(key_bytes))
    ciphertext = box.encrypt(secret_bytes)
    return base64.b64encode(ciphertext).decode("ascii")


def setup_environment_api_key(
    backend_base: str,
    synth_api_key: str,
    token: str | None = None,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    import requests

    backend = backend_base.rstrip("/")
    if backend.endswith("/api"):
        backend = backend[:-4]
    if not backend:
        raise ValueError("backend_base must be provided")
    if not synth_api_key:
        raise ValueError("synth_api_key must be provided")

    plaintext = token if token is not None else os.getenv(ENVIRONMENT_API_KEY_NAME, "").strip()
    if not plaintext:
        raise ValueError("ENVIRONMENT_API_KEY must be set (or pass token=...) to upload")
    if not isinstance(plaintext, str):
        raise TypeError("token must be a string")

    token_bytes = plaintext.encode("utf-8")
    if not token_bytes:
        raise ValueError("ENVIRONMENT_API_KEY token must not be empty")
    if len(token_bytes) > MAX_ENVIRONMENT_API_KEY_BYTES:
        raise ValueError("ENVIRONMENT_API_KEY token exceeds 8 KiB limit")

    headers = {"Authorization": f"Bearer {synth_api_key}"}
    pub_url = f"{backend}/api/v1/crypto/public-key"
    response = requests.get(pub_url, headers=headers, timeout=timeout)
    _raise_with_detail(response)

    try:
        doc = response.json()
    except ValueError as exc:
        raise RuntimeError("backend returned invalid JSON for public key") from exc

    if not isinstance(doc, dict):
        raise RuntimeError("backend public key response must be an object")

    pubkey = doc.get("public_key")
    if not isinstance(pubkey, str) or not pubkey:
        raise RuntimeError("backend response missing public_key")

    alg = doc.get("alg")
    if alg is not None and alg != _ALGORITHM:
        raise RuntimeError(f"unsupported sealed box algorithm: {alg}")

    ciphertext_b64 = encrypt_for_backend(pubkey, token_bytes)

    body = {"name": ENVIRONMENT_API_KEY_NAME, "ciphertext_b64": ciphertext_b64}
    post_url = f"{backend}/api/v1/env-keys"

    response2 = requests.post(
        post_url,
        headers={**headers, "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    _raise_with_detail(response2)

    try:
        upload_doc = response2.json()
    except ValueError:
        upload_doc = {}

    if not isinstance(upload_doc, dict):
        upload_doc = {}

    return {
        "stored": True,
        "id": upload_doc.get("id"),
        "name": upload_doc.get("name"),
        "updated_at": upload_doc.get("updated_at"),
    }


def _raise_with_detail(response: Any) -> None:
    import requests

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail_snippet: str | None = None
        try:
            detail = response.json()
            detail_snippet = json.dumps(detail, separators=(",", ":"))[:200]
        except Exception:
            body = response.text if response.text is not None else ""
            detail_snippet = body[:200] if body else None
        message = str(exc)
        if detail_snippet:
            message = f"{message} | body={detail_snippet}"
        raise requests.HTTPError(message, request=exc.request, response=exc.response) from None
