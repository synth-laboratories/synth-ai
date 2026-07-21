"""``research.account.keys`` binding tests: fake sessions only, no network.

Covers path/param construction and typed decode for list/create/rotate/delete
against the backend's /api/v1/auth/keys lifecycle routes.
"""

from __future__ import annotations

from typing import Any

import pytest
from synth_ai.research.account import (
    ApiKeyDeactivated,
    ApiKeyRedactedRow,
    MintedApiKey,
    ResearchAccountAPI,
    ResearchAccountKeysAPI,
)

FULL_KEY = "sk_synth_user_" + "ab12" * 12


class _RecordingSession:
    """Fake session client capturing every ``_request_json`` call."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = responses or {}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "params": params,
                "json_body": json_body,
            }
        )
        return self._responses.get(f"{method} {path}", {})


def _redacted_row() -> dict[str, Any]:
    return {
        "key_id": "11111111-2222-3333-4444-555555555555",
        "prefix": "sk_synth_user_",
        "last4": "b12a",
        "key_type": "synth",
        "is_active": True,
        "created_at": "2026-07-20T12:00:00Z",
        "last_used_at": None,
    }


def _minted_payload() -> dict[str, Any]:
    return {
        "key": FULL_KEY,
        "key_id": "11111111-2222-3333-4444-555555555555",
        "prefix": "sk_synth_user_",
        "last4": FULL_KEY[-4:],
        "key_type": "synth",
        "created_at": "2026-07-20T12:00:00Z",
    }


def test_list_path_and_typed_decode() -> None:
    session = _RecordingSession(
        responses={"GET /api/v1/auth/keys": {"keys": [_redacted_row()]}}
    )
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    rows = keys.list()
    assert session.calls == [
        {
            "method": "GET",
            "path": "/api/v1/auth/keys",
            "params": None,
            "json_body": None,
        }
    ]
    assert isinstance(rows, tuple) and len(rows) == 1
    row = rows[0]
    assert isinstance(row, ApiKeyRedactedRow)
    assert row.prefix == "sk_synth_user_"
    assert row.last4 == "b12a"
    assert row.created_at is not None
    # Redacted row must never carry the full secret.
    assert not hasattr(row, "key")


def test_create_path_body_and_one_time_decode() -> None:
    session = _RecordingSession(responses={"POST /api/v1/auth/keys": _minted_payload()})
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    minted = keys.create()
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["path"] == "/api/v1/auth/keys"
    assert session.calls[0]["json_body"] is None
    assert isinstance(minted, MintedApiKey)
    assert minted.key == FULL_KEY
    assert minted.last4 == FULL_KEY[-4:]


def test_create_forwards_name_when_given() -> None:
    session = _RecordingSession(responses={"POST /api/v1/auth/keys": _minted_payload()})
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    keys.create(name="prod")
    assert session.calls[0]["json_body"] == {"name": "prod"}


def test_rotate_path_and_decode() -> None:
    key_id = "11111111-2222-3333-4444-555555555555"
    session = _RecordingSession(
        responses={f"POST /api/v1/auth/keys/{key_id}/rotate": _minted_payload()}
    )
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    minted = keys.rotate(key_id)
    assert session.calls == [
        {
            "method": "POST",
            "path": f"/api/v1/auth/keys/{key_id}/rotate",
            "params": None,
            "json_body": None,
        }
    ]
    assert isinstance(minted, MintedApiKey)
    assert minted.key == FULL_KEY


def test_delete_path_and_decode() -> None:
    key_id = "11111111-2222-3333-4444-555555555555"
    session = _RecordingSession(
        responses={
            f"DELETE /api/v1/auth/keys/{key_id}": {
                "key_id": key_id,
                "status": "deactivated",
            }
        }
    )
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    result = keys.delete(key_id)
    assert session.calls[0]["method"] == "DELETE"
    assert session.calls[0]["path"] == f"/api/v1/auth/keys/{key_id}"
    assert isinstance(result, ApiKeyDeactivated)
    assert result.key_id == key_id
    assert result.status == "deactivated"


def test_account_facade_exposes_keys_namespace() -> None:
    session = _RecordingSession()
    account = ResearchAccountAPI(session)  # type: ignore[arg-type]
    assert isinstance(account.keys, ResearchAccountKeysAPI)
    assert account.keys is account.keys  # cached


def test_non_mapping_payload_is_rejected() -> None:
    session = _RecordingSession(responses={"POST /api/v1/auth/keys": ["nope"]})
    keys = ResearchAccountKeysAPI(session)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        keys.create()
