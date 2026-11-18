"""SYN-5737 owner-state and fixture preconditions without staging credentials."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.acceptance import isolation_remaining as remaining
from scripts.acceptance.journeys import Secret, Verdict
from scripts.acceptance.surfaces import SurfaceName
from synth_ai._generated.models import PrincipalCredentialKind


@pytest.mark.acceptance
def test_remaining_owner_state_change_halts_other_drivers(monkeypatch, tmp_path):
    ids = {
        "owner_org_id": "org_a",
        "owner_user_id": "user_a",
        "foreign_org_id": "org_b",
        "foreign_user_id": "user_b",
        "run_id": "fixture_run",
        "output_id": "fixture_output",
        "key_id": "fixture_key",
        "upload_id": "fixture_upload",
        "request_id": "fixture_request",
        "fixture_key_name": "disposable",
        "message_id": "fixture_message",
        "grant_id": "fixture_grant",
    }
    credentials = {
        "SYNTH_API_KEY": ("a_key", "a", PrincipalCredentialKind.API_KEY),
        Secret.SESSION: ("a_session", "a", PrincipalCredentialKind.SESSION),
        Secret.FOREIGN: ("b_key", "b", PrincipalCredentialKind.API_KEY),
        Secret.FOREIGN_SESSION: ("b_session", "b", PrincipalCredentialKind.SESSION),
    }
    clients = {}
    for name, (token, side, kind) in credentials.items():
        monkeypatch.setenv(name, token)
        account = SimpleNamespace(
            org=SimpleNamespace(id=f"org_{side}"),
            principal=SimpleNamespace(
                org_id=f"org_{side}",
                user_id=f"user_{side}",
                credential_kind=kind,
                scopes=["account:write"],
            ),
        )
        clients[token] = SimpleNamespace(account=SimpleNamespace(get=lambda item=account: item))

    monkeypatch.setattr(remaining, "SynthClient", lambda token, **_: nullcontext(clients[token]))
    monkeypatch.setattr(remaining, "verify_output", lambda *_: None)
    baseline = remaining.OwnerSnapshot("key", "labels", "plan", "upload", 4, 6)
    changed = remaining.OwnerSnapshot("key", "new-labels", "plan", "upload", 4, 7)
    snapshots = iter((baseline, changed))
    monkeypatch.setattr(remaining.OwnerSnapshot, "read", lambda *_, **__: next(snapshots))
    seen = []

    def driver(surface, _directory: Path, cells, *_, **__):
        seen.append(surface)
        for cell in cells:
            cell["result"] = "pass"
        return Verdict.PASS

    monkeypatch.setattr(remaining, "run_driver", driver)
    cells = remaining.cell_rows(
        (
            remaining.Cell(SurfaceName.SDK, "updateRunLabels", "uncovered:surface_unmapped"),
            remaining.Cell(SurfaceName.CLI, "updateRunLabels", "uncovered:surface_unmapped"),
        )
    )

    with pytest.raises(remaining.FixtureBlocked, match="owner_fixture_changed"):
        remaining.run(cells, tmp_path, ids, {})

    assert seen == [SurfaceName.SDK]
    assert cells[0]["result"] == "fail:owner_state_changed"
    assert cells[0]["owner_snapshot"] == {"before": "labels", "after": "new-labels"}
    assert cells[1]["result"] == "pending:owned_fixture"


def test_remaining_cells_pin_the_current_contract():
    import tomllib

    from scripts.acceptance import isolation_remaining as remaining

    source = tomllib.loads((Path(__file__).parents[2] / "contracts/SOURCE.toml").read_text())
    assert remaining.CONTRACT_SHA256 == source["sha256"]
    assert remaining.RELEASE == "1.39.0"
    # Run messages shipped after the retained 1.35.1 receipt: they are open cells now.
    cell = remaining.Cell(remaining.SurfaceName.SDK, "sendRunMessage", "not_in_release:1.35.1")
    assert cell.disposition == "pending:owned_fixture"
