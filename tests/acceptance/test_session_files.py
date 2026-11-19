"""SYN-5737: matrix Clerk sessions re-read from 0600 token files at each use.

A Clerk session token lives about 60 s; an external refresher rewrites the file
atomically. These loopback tests rotate the file mid-matrix and require every
newly read token to pass the same principal check as the first one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from journey_loopback import api, matrix_ids, organization_accounts, problem

from scripts.acceptance import journey_main
from scripts.acceptance.journeys import JourneyContext, Secret, Step, Verdict, run_step
from scripts.acceptance.session_files import (
    FILE_SUFFIX,
    JourneyConfigError,
    LiveSecrets,
    load_secrets,
)
from scripts.acceptance.surfaces import SurfaceName

pytestmark = pytest.mark.acceptance

NAMES = ("SYNTH_API_KEY", *(name.value for name in Secret))
FILES = (Secret.SESSION.value, Secret.FOREIGN_SESSION.value)


def write_token(path: Path, token: str) -> None:
    """The refresher's write: a 0600 temp file renamed over the old one."""
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as output:
        output.write(token + "\n")
    os.replace(temporary, path)


def matrix(tmp_path, responder, files, binding="distinct"):
    secrets = LiveSecrets(
        {"SYNTH_API_KEY": "fixture-admin", Secret.FOREIGN.value: "fixture-org-b"}, files
    )
    accounts = organization_accounts(responder, binding)

    def answer(method, path, request):
        # Every rotated org A session token is bound to the same owner principal.
        if path == "/v1/account" and request["bearer"].startswith("fixture-session-a"):
            request = {**request, "bearer": "fixture-session-a"}
        return accounts(method, path, request)

    with api(answer) as (origin, requests):
        make = journey_main.surface_factory(
            SurfaceName.SDK, origin=origin, work_directory=tmp_path, mcp_url=None
        )
        with make("fixture-admin") as surface:
            context = JourneyContext(
                surface=surface,
                driver=SurfaceName.SDK,
                origin=origin,
                make=make,
                receipt_dir=tmp_path,
                secrets=secrets,
                foreign_ids=matrix_ids(),
            )
            return run_step(context, Step.CROSS_ORG_404), requests


def test_rotating_both_session_files_mid_matrix_reopens_and_rechecks(tmp_path):
    foreign, owner = tmp_path / "foreign-session", tmp_path / "owner-session"
    write_token(foreign, "fixture-session-b1")
    write_token(owner, "fixture-session-a1")
    seen = []

    def responder(method, path, request):
        bearer = request["bearer"]
        if path.startswith("/v1/api-keys"):
            seen.append(bearer)
            # The refresher runs mid-matrix, after each first use.
            if bearer == "fixture-session-b1":
                write_token(foreign, "fixture-session-b2")
                write_token(owner, "fixture-session-a2")
            if bearer.startswith("fixture-session-a"):
                return 409, problem("api_key_version_conflict", 409), {}
            if not bearer.startswith("fixture-session-b"):
                return 403, problem("session_required", 403), {}
            return 404, problem("key_not_found", 404), {}
        return 404, problem("resource_not_found", 404), {}

    result, requests = matrix(
        tmp_path,
        responder,
        {Secret.FOREIGN_SESSION.value: foreign, Secret.SESSION.value: owner},
    )
    # Both files rotated mid-matrix and the new tokens carried the later probes.
    assert {"fixture-session-b1", "fixture-session-b2", "fixture-session-a2"} <= set(seen)
    # Each token was principal-checked exactly once, before any other request used it.
    for token in ("fixture-session-a1", "fixture-session-a2", *sorted(set(seen))):
        used = [path for _, path, bearer in requests if bearer == token]
        assert used[0] == "/v1/account" and used.count("/v1/account") == 1, token
    session_rows = [
        row for row in result.evidence if row.get("actor_credential") == "clerk_session"
    ]
    assert {row["operation"] for row in session_rows} >= {
        "revokeApiKey",
        "updateApiKey",
        "rotateApiKey",
    }
    assert all(row["result"] == "pass" for row in session_rows), session_rows


def test_rotated_token_of_another_principal_fails_mid_matrix(tmp_path):
    token = tmp_path / "foreign-session"
    write_token(token, "fixture-session-b1")

    def responder(method, path, request):
        if path.startswith("/v1/api-keys"):
            write_token(token, "fixture-session-b")  # Bound to user_third ("wrong_session").
            return 404, problem("key_not_found", 404), {}
        return 404, problem("resource_not_found", 404), {}

    base = organization_accounts(responder, "wrong_session")
    with api(base) as (origin, _):
        make = journey_main.surface_factory(
            SurfaceName.SDK, origin=origin, work_directory=tmp_path, mcp_url=None
        )
        secrets = LiveSecrets(
            {"SYNTH_API_KEY": "fixture-admin", Secret.FOREIGN.value: "fixture-org-b"},
            {Secret.FOREIGN_SESSION.value: token},
        )
        with make("fixture-admin") as surface:
            context = JourneyContext(
                surface=surface,
                driver=SurfaceName.SDK,
                origin=origin,
                make=make,
                receipt_dir=tmp_path,
                secrets=secrets,
                foreign_ids=matrix_ids(),
            )
            result = run_step(context, Step.CROSS_ORG_404)
    assert result.verdict == Verdict.FAIL
    assert result.cause == "foreign_session_principal_mismatch"


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o700])
def test_token_file_must_be_0600(tmp_path, mode):
    token = tmp_path / "session"
    write_token(token, "fixture-session-a")
    token.chmod(mode)
    environ = {"SYNTH_API_KEY": "k", Secret.SESSION.value + FILE_SUFFIX: str(token)}
    with pytest.raises(JourneyConfigError) as refused:
        load_secrets(environ, NAMES, FILES)
    assert refused.value.code == "session_token_file_not_0600"
    assert "fixture-session-a" not in str(refused.value)


def test_mode_is_rechecked_at_each_read(tmp_path):
    token = tmp_path / "session"
    write_token(token, "fixture-session-a")
    environ = {"SYNTH_API_KEY": "k", Secret.SESSION.value + FILE_SUFFIX: str(token)}
    secrets = load_secrets(environ, NAMES, FILES)
    assert secrets[Secret.SESSION.value] == "fixture-session-a"
    token.chmod(0o644)
    with pytest.raises(JourneyConfigError, match="session_token_file_not_0600"):
        secrets.get(Secret.SESSION.value)


@pytest.mark.parametrize("name", FILES)
def test_file_and_value_both_set_is_refused(tmp_path, name):
    token = tmp_path / "session"
    write_token(token, "fixture-session")
    environ = {"SYNTH_API_KEY": "k", name: "fixed-value", name + FILE_SUFFIX: str(token)}
    with pytest.raises(JourneyConfigError) as refused:
        load_secrets(environ, NAMES, FILES)
    assert refused.value.code == "session_token_file_and_value_both_set"
    assert refused.value.name == name
    assert "fixed-value" not in str(refused.value)


def test_fixed_values_still_work_and_never_print(tmp_path):
    environ = {"SYNTH_API_KEY": "k", Secret.FOREIGN_SESSION.value: "fixture-session-b"}
    secrets = load_secrets(environ, NAMES, FILES)
    assert secrets.get(Secret.FOREIGN_SESSION.value) == "fixture-session-b"
    assert secrets.get(Secret.SESSION.value) is None
    assert "fixture-session-b" not in repr(secrets)


def test_entrypoint_refuses_both_set_before_the_receipt_directory(tmp_path, monkeypatch):
    token = tmp_path / "session"
    write_token(token, "fixture-session")
    monkeypatch.setenv("SYNTH_LOCAL_API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SYNTH_API_KEY", "fixture-admin")
    monkeypatch.setenv(Secret.SESSION.value, "fixed")
    monkeypatch.setenv(Secret.SESSION.value + FILE_SUFFIX, str(token))
    with pytest.raises(JourneyConfigError, match="session_token_file_and_value_both_set"):
        journey_main.main(
            [
                "--driver",
                "sdk",
                "--steps",
                Step.SKU_REFUSAL.value,
                "--receipt-dir",
                str(tmp_path / "receipts"),
            ]
        )
    assert not (tmp_path / "receipts").exists()
