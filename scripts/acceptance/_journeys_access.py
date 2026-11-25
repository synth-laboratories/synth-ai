"""J3 access, API key lifecycle, and OAuth journey steps for SYN-5695."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import select
import subprocess
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from synth_ai import auth
from synth_ai._auth_models import ClientRegistration, OAuthToken
from synth_ai.errors import SynthError

from ._journeys_common import (
    ADMIN_ROLES,
    MEMBER_ROLES,
    ROOT,
    JourneyContext,
    JourneyPrincipal,
    Secret,
    Step,
    StepReceipt,
    Verdict,
    api_key_refused,
    credential_works,
    expect_refusal,
    items,
    key,
    missing,
    receipt,
)
from .j1 import AdapterError
from .oauth_journey import HostedOAuth, OAuthJourneyError, approve, response_object
from .surfaces import Call, CLISurface, SurfaceName, obj

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from pydantic import JsonValue


def mint_key(context: JourneyContext) -> StepReceipt:
    step = Step.MINT_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    body: dict[str, JsonValue] = {
        "purpose": "automation",
        "name": "syn5695-" + secrets.token_hex(4),
        "scopes": ["account:read"],
        "expires_at": expires,
        "request_id": str(uuid.uuid4()),
    }
    with context.make(session) as surface:
        minted = surface.call(Call("createApiKey", body=body, idempotency_key=key("mint")))
    if not minted.accepted or not isinstance(minted.value, dict):
        return receipt(context, step, Verdict.FAIL, "key_not_minted", minted)
    context.state["key_id"] = str(minted.value["key_id"])
    context.state["secret"] = str(minted.value["key"])
    used = credential_works(context, context.state["secret"])
    verdict = Verdict.PASS if used.accepted else Verdict.FAIL
    return receipt(
        context, step, verdict, "" if used.accepted else "minted_key_refused", minted, used
    )


def rotate_key(context: JourneyContext) -> StepReceipt:
    step = Step.ROTATE_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if "key_id" not in context.state:
        return missing(context, step, "minted_key")
    with context.make(session) as surface:
        rotated = surface.call(
            Call(
                "rotateApiKey",
                path={"key_id": context.state["key_id"]},
                idempotency_key=key("rotate"),
            )
        )
    if not rotated.accepted or not isinstance(rotated.value, dict):
        return receipt(context, step, Verdict.FAIL, "key_not_rotated", rotated)
    old = credential_works(context, context.state["secret"])
    context.state["key_id"] = str(rotated.value["key_id"])
    context.state["secret"] = str(rotated.value["key"])
    new = credential_works(context, context.state["secret"])
    ok = new.accepted and api_key_refused(old)
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "rotation_not_enforced",
        rotated,
        old,
        new,
    )


def revoke_key(context: JourneyContext) -> StepReceipt:
    step = Step.REVOKE_KEY
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if "key_id" not in context.state:
        return missing(context, step, "minted_key")
    with context.make(session) as surface:
        revoked = surface.call(Call("revokeApiKey", path={"key_id": context.state["key_id"]}))
    after = credential_works(context, context.state["secret"])
    ok = revoked.accepted and api_key_refused(after)
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "revocation_not_enforced",
        revoked,
        after,
    )


@contextmanager
def private_auth_file(context: JourneyContext) -> Generator[Path]:
    path = context.receipt_dir / "oauth-credentials.json"
    previous = os.environ.get("SYNTH_AUTH_FILE")
    os.environ["SYNTH_AUTH_FILE"] = str(path)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)  # Live OAuth tokens never stay in receipts.
        if previous is None:
            os.environ.pop("SYNTH_AUTH_FILE", None)
        else:
            os.environ["SYNTH_AUTH_FILE"] = previous


def oauth_failure(
    context: JourneyContext, step: Step, error: Exception, notes: dict[str, JsonValue]
) -> StepReceipt:
    if isinstance(error, OAuthJourneyError):
        stage, code = error.stage, error.code
    elif isinstance(error, httpx.HTTPStatusError):
        operation = error.request.url.path.rsplit("/", 1)[-1]
        stage = (
            operation
            if operation in {"register", "token", "revoke", "oauth-authorization-server"}
            else "oauth"
        )
        notes[f"{stage}_status"] = error.response.status_code
        code = "http_refused"
    else:
        stage, code = "oauth", type(error).__name__
    notes["oauth_failure_stage"] = stage
    notes["oauth_failure_code"] = code
    return receipt(context, step, Verdict.FAIL, f"oauth_failed:{stage}:{code}", notes=notes)


def cli_oauth(context: JourneyContext) -> StepReceipt:
    step = Step.CLI_OAUTH
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    notes: dict[str, JsonValue] = {}
    oauth = auth.oauth_base(context.origin.rstrip("/"))

    def consent(url: str) -> str:
        return approve(oauth, session, url, oauth + "/mcp", notes)

    try:
        with private_auth_file(context) as path:
            if context.driver == SurfaceName.SDK:
                auth.login(context.origin, open_browser=False, redirect_source=consent)
            else:
                cli_login(context, path, consent, notes)
            record = auth.read_tokens().get(context.origin.rstrip("/"))
            if record is None:
                raise OAuthJourneyError("token_store", "oauth_credential_missing")
            try:
                used = credential_works(context, record.access_token)
            finally:
                auth.logout(context.origin)
                notes["logout_acknowledged"] = True
                after = credential_works(context, record.access_token)
                notes["revoked_token_verified"] = (
                    not after.accepted
                    and after.status == 401
                    and after.code == "oauth_token_invalid"
                )
    except (
        OAuthJourneyError,
        AdapterError,
        SynthError,
        httpx.HTTPError,
        ValueError,
        OSError,
    ) as error:
        return oauth_failure(context, step, error, notes)
    kind = obj(obj(used.value).get("principal")).get("credential_kind")
    notes["credential_kind"] = kind
    cause = ""
    if not used.accepted or kind != "oauth":
        cause = "oauth_token_refused"
    elif notes["revoked_token_verified"] is not True:
        cause = "oauth_token_revocation_unproven"
    return receipt(
        context,
        step,
        Verdict.FAIL if cause else Verdict.PASS,
        cause,
        used,
        after,
        notes=notes,
    )


def cli_login(
    context: JourneyContext,
    path: Path,
    consent: Callable[[str], str],
    notes: dict[str, JsonValue],
) -> None:
    surface = context.surface
    assert isinstance(surface, CLISurface)
    environment = {
        key_: value for key_, value in surface.environment.items() if key_ != "SYNTH_API_KEY"
    }
    environment["SYNTH_AUTH_FILE"] = str(path)
    with subprocess.Popen(
        [
            str(surface.python),
            "-m",
            "synth_ai.cli",
            "--base-url",
            context.origin,
            "login",
            "--no-browser",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stderr is not None
        try:
            readable, _, _ = select.select([child.stderr], [], [], 65)
            if not readable:
                raise OAuthJourneyError("cli_login", "cli_authorize_timeout")
            authorization_url = child.stderr.readline().strip()
            if not authorization_url:
                raise OAuthJourneyError("cli_login", "cli_authorize_url_missing")
            callback = consent(authorization_url)
            child.communicate(callback + "\n", timeout=120)
            notes["cli_login_exit"] = child.returncode
            if child.returncode != 0:
                raise OAuthJourneyError("cli_login", "cli_token_exchange_failed")
        except subprocess.TimeoutExpired as error:
            raise OAuthJourneyError("cli_login", "cli_exchange_timeout") from error
        finally:
            if child.poll() is None:
                child.kill()


def hosted_token(
    client: httpx.Client, oauth: HostedOAuth, session: str, notes: dict[str, JsonValue]
) -> tuple[str, str]:
    redirect_uri = "http://127.0.0.1:9/callback"
    registration = response_object(
        client.post(
            oauth.endpoint("registration_endpoint"),
            json={
                "client_name": "SYN-5695 journey",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        ),
        "register",
        201,
        notes,
    )
    client_id = ClientRegistration.model_validate(registration).client_id
    state = secrets.token_urlsafe(16)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    authorization = str(
        httpx.URL(
            oauth.endpoint("authorization_endpoint"),
            params={
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "resource": oauth.resource,
                # Core refuses any authorize without the required scope
                # (publicpolicy.RequiredScopes: user:org:read, alias of account:read).
                "scope": "user:org:read",
            },
        )
    )
    callback = approve(oauth.origin, session, authorization, oauth.resource, notes)
    result = auth.callback_result(callback, state=state, issuer=oauth.origin)
    if (
        urlparse(callback)._replace(query="") != urlparse(redirect_uri)
        or result is None
        or "code" not in result
        or "error" in result
    ):
        raise OAuthJourneyError("callback", "oauth_callback_mismatch")
    token = response_object(
        client.post(
            oauth.endpoint("token_endpoint"),
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": result["code"],
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": oauth.resource,
            },
        ),
        "token",
        200,
        notes,
    )
    return client_id, OAuthToken.model_validate(token).access_token


def owned_grant(
    client: httpx.Client,
    grants_url: str,
    clerk: dict[str, str],
    client_id: str,
    notes: dict[str, JsonValue],
) -> str:
    grants = response_object(client.get(grants_url, headers=clerk), "grants", 200, notes)
    matches = [
        obj(grant)
        for grant in items(grants.get("grants"))
        if obj(grant).get("client_id") == client_id
    ]
    if len(matches) != 1:
        raise OAuthJourneyError("grants", "oauth_grant_not_unique")
    grant = matches[0]
    identity = grant.get("grant_id")
    if (
        grant.get("status") != "active"
        or grant.get("owned_by_viewer") is not True
        or not isinstance(identity, str)
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity) is None
    ):
        raise OAuthJourneyError("grants", "oauth_grant_identity_invalid")
    notes["grant_bound"] = True
    return identity


def mcp_consent(context: JourneyContext) -> StepReceipt:
    """Exercise the hosted MCP OAuth grant through discovery, tool use and revoke."""
    step = Step.MCP_CONSENT
    session = context.secrets.get(Secret.SESSION)
    if session is None:
        return missing(context, step, "clerk_session")
    if context.mcp_url is None:
        return missing(context, step, "hosted_mcp_url")
    notes: dict[str, JsonValue] = {}
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            oauth = HostedOAuth.discover(client, context.mcp_url, notes)
            revocation_endpoint = oauth.endpoint("revocation_endpoint")
            client_id, access = hosted_token(client, oauth, session, notes)
            grants_url = oauth.origin + "/mcp/oauth/grants"
            clerk = {"X-Clerk-Session-Token": session}
            grant_revoked = False
            try:
                identity = owned_grant(client, grants_url, clerk, client_id, notes)
                try:
                    used = credential_works(context, access)
                finally:
                    revoked = response_object(
                        client.delete(grants_url + "/" + identity, headers=clerk),
                        "revoke",
                        200,
                        notes,
                    )
                    grant_revoked = (
                        revoked.get("grant_id") == identity
                        and revoked.get("status") == "revoked"
                        and revoked.get("outcome") == "revoked"
                    )
                    if not grant_revoked:
                        raise OAuthJourneyError("revoke", "oauth_grant_revocation_unproven")
            finally:
                if not grant_revoked:
                    oauth.cleanup_token(client, revocation_endpoint, client_id, access, notes)
            after = client.get(oauth.resource, headers={"Authorization": "Bearer " + access})
            refused = oauth.revoked(after, notes)
    except (
        OAuthJourneyError,
        AdapterError,
        SynthError,
        httpx.HTTPError,
        ValueError,
        OSError,
    ) as error:
        return oauth_failure(context, step, error, notes)
    kind = obj(obj(used.value).get("principal")).get("credential_kind")
    notes["credential_kind"] = kind
    ok = used.accepted and kind == "oauth" and refused
    return receipt(
        context,
        step,
        Verdict.PASS if ok else Verdict.FAIL,
        "" if ok else "mcp_oauth_grant_lifecycle_failed",
        used,
        notes=notes,
    )


def member_admin(context: JourneyContext) -> StepReceipt:
    step = Step.MEMBER_ADMIN
    member = context.secrets.get(Secret.MEMBER)
    if member is None:
        return missing(context, step, "member_api_key")
    admin = context.surface.call(Call("getAccount"))
    with context.make(member) as surface:
        identity = surface.call(Call("getAccount"))
        administrator = JourneyPrincipal.from_account(admin)
        participant = JourneyPrincipal.from_account(identity)
        if administrator is None or participant is None:
            return missing(context, step, "user_bound_role_principals")
        roles = {"admin": administrator.role, "member": participant.role}
        if administrator.role not in ADMIN_ROLES or participant.role not in MEMBER_ROLES:
            return receipt(
                context, step, Verdict.FAIL, "roles_unexpected", admin, identity, notes=roles
            )
        if (
            administrator.org_id != participant.org_id
            or administrator.user_id == participant.user_id
        ):
            return receipt(
                context,
                step,
                Verdict.FAIL,
                "role_principals_mismatch",
                admin,
                identity,
                notes=roles,
            )
        roles.update(
            admin_user_id=administrator.user_id,
            member_user_id=participant.user_id,
            org_id=participant.org_id,
        )
        refused = surface.call(Call("cancelSubscription", idempotency_key=key("member-cancel")))
    result = expect_refusal(
        context, step, refused, codes=frozenset({"billing_admin_required"}), status=403
    )
    return StepReceipt(
        step,
        context.driver,
        result.verdict,
        result.cause,
        (admin.evidence(), identity.evidence(), *result.evidence, roles),
    )
