"""Verify pinned Forge compatibility and independent tree-content identity."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_ai.errors import ContractError
from synth_ai.sdk.core.commit_digest import (
    CommitDigestMismatch,
    commit_manifest_digest,
    verify_tree,
)
from synth_ai.types import Link, Producer, Ref, Relation
from synth_ai.types.refs import LinkInvalid, RefInvalid

FIXTURES = Path(__file__).parent / "fixtures"


def test_forge_wire_shapes_are_unchanged():
    assert Ref.model_json_schema() == json.loads(
        (FIXTURES / "forge/ExactReference.v1.json").read_text()
    )
    assert Producer.model_json_schema() == json.loads(
        (FIXTURES / "forge/Producer.v1.json").read_text()
    )


def test_exact_reference_factories_and_rejections():
    assert Ref.commit("repo", "b" * 40, "a" * 64).revision == "b" * 40
    assert Ref.artifact("artifact", "23", "a" * 64).revision == "23"
    assert Ref.run("run", "a" * 64).revision == "1"
    for identifier in ["", "a" * 129, "..", "../escape"]:
        with pytest.raises(RefInvalid):
            Ref.artifact(identifier, "1", "a" * 64)
    with pytest.raises(RefInvalid):
        Ref.artifact("artifact", "1", "A" * 64)
    with pytest.raises(RefInvalid):
        Ref.commit("repo", "latest", "a" * 64)


def test_link_round_trip_and_closed_relation():
    data = {
        "link_id": "link",
        "project_id": "project",
        "subject": Ref.run("run", "a" * 64).model_dump(),
        "relation": "produced",
        "object": Ref.artifact("artifact", "1", "b" * 64).model_dump(),
        "producer": {"authority": "backend", "subject_id": "swarms-runs"},
        "created_at": "2026-10-19T12:00:00Z",
    }
    link = Link.from_wire(json.dumps(data))
    assert link.relation is Relation.PRODUCED
    assert Link.model_validate_json(link.model_dump_json()) == link
    data["relation"] = "new_relation"
    with pytest.raises(LinkInvalid):
        Link.from_wire(json.dumps(data))
    del data["producer"]
    with pytest.raises(ValidationError):
        Link.model_validate_json(json.dumps(data))


def test_shared_commit_digest_vectors():
    data = (FIXTURES / "commit-digest-v1.json").read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "06d3247d9adde04453a2c46b7afacbb6e6767e04c91779a4c66bcf54ff7fe4a2"
    )
    for case in json.loads(data)["cases"]:
        assert commit_manifest_digest(reversed(case["entries"])) == case["digest_sha256"]


def test_extracted_tree_verification_and_tampering(tmp_path):
    (tmp_path / "caf\u00e9").write_bytes(b"line\r\n")
    (tmp_path / "caf\u00e9").chmod(0o755)
    (tmp_path / "link").symlink_to("caf\u00e9")
    entries = [
        {"path": "caf\u00e9", "mode": "100755", "sha256": hashlib.sha256(b"line\r\n").hexdigest()},
        {
            "path": "link",
            "mode": "120000",
            "sha256": hashlib.sha256("caf\u00e9".encode()).hexdigest(),
        },
    ]
    expected = commit_manifest_digest(entries)
    assert verify_tree(tmp_path, expected) == expected
    (tmp_path / "caf\u00e9").write_bytes(b"tampered")
    with pytest.raises(CommitDigestMismatch) as error:
        verify_tree(tmp_path, expected)
    assert error.value.expected == expected
    assert error.value.actual != expected


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a//b", "a/./b", ".git/config"])
def test_invalid_manifest_paths_fail_closed(path):
    with pytest.raises(ContractError):
        commit_manifest_digest([{"path": path, "mode": "100644", "sha256": "a" * 64}])
