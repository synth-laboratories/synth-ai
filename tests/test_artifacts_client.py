"""client.artifacts offline: get/list/upload/prepare/finalize/labels/alias/download."""

import hashlib
import json

import pytest
from synth_ai.errors import ContractError
from synth_ai.sdk.core.artifact_types import Artifact, ArtifactSource
from synth_ai.types.refs import Ref
from tests.artifact_fixtures import Fake, artifact_body, drive

pytestmark = pytest.mark.unit
MODES = pytest.mark.parametrize("async_mode", [False, True])


@MODES
def test_get_and_list_decode_tolerant_labels_but_exact_selection(async_mode):
    server = Fake()
    server.content = b"hello artifact\n"

    async def steps(client, w):
        listed = await w(client.artifacts.list("project-test", label={"tier": "gold"}, limit=5))
        got = await w(client.artifacts.get("project-test", "art-1"))
        by_ref = await w(client.artifacts.get("project-test", got.ref))
        return listed, got, by_ref

    # The server may store labels a request could never set (uppercase, spaces).
    original = server.artifact

    def labelled(*args):
        value = original(*args)
        value["labels"] = {"tier": "gold", "Legacy Key": "A" * 300}
        value["future"] = {"ignored": True}
        return value

    server.artifact = labelled
    listed, got, by_ref = drive(async_mode, server, steps)
    assert got.labels["Legacy Key"] == "A" * 300 and by_ref.ref == got.ref
    assert listed.items[0].artifact_id == "art-1"
    query = [r for r in server.requests if r.url.path.endswith("/artifacts")][0].url.params
    assert query.get_list("label") == ["tier=gold"] and query["limit"] == "5"
    exact = [r for r in server.requests if r.url.path.endswith("/art-1")][-1].url.params
    assert exact["digest_sha256"] == got.ref.digest_sha256 and exact["revision"] == "1"


def test_manifest_stays_strict_because_its_digest_covers_every_field():
    body = artifact_body()
    body["manifest"]["future_field"] = 1
    # Unknown fields are not silently trusted: the server's digest would cover
    # them, so recomputing over the known fields must fail closed.
    unsigned = {k: v for k, v in body["manifest"].items() if k != "manifest_digest"}
    body["manifest"]["manifest_digest"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical digest"):
        Artifact.model_validate_json(json.dumps(body))
    body = artifact_body(labels={"Any Key": "x"})
    assert Artifact.model_validate_json(json.dumps(body)).labels == {"Any Key": "x"}


@MODES
def test_upload_received_skips_transfer_and_finalizes_with_exact_identity(tmp_path, async_mode):
    source_path = tmp_path / "report.md"
    source_path.write_bytes(b"hello artifact\n")
    source = ArtifactSource.from_path(source_path, media_type="text/markdown")
    server = Fake()

    async def steps(client, w):
        return await w(
            client.artifacts.upload("project-test", source, kind="file", labels={"tier": "gold"})
        )

    artifact = drive(async_mode, server, steps)
    assert artifact.artifact_id == "art-1"
    paths = [r.url.path.removeprefix("/api/v1") for r in server.requests]
    assert paths == ["/projects/project-test/artifacts/uploads", "/artifacts/uploads/up-1/finalize"]
    prepare = server.bodies[0][2]
    assert prepare["labels"] == {"tier": "gold"} and "artifact_id" not in prepare
    assert prepare["manifest"]["digest_sha256"] == hashlib.sha256(b"hello artifact\n").hexdigest()


@MODES
def test_upload_prepared_transfers_once_before_finalize(tmp_path, monkeypatch, async_mode):
    import synth_ai.sdk.core.artifacts as module

    calls = []

    def fake_transfer(status, source, options, hosts):
        calls.append(status.state)
        if async_mode:

            async def done():
                return "receipt"

            return done()
        return "receipt"

    monkeypatch.setattr(module, "transfer_bytes", fake_transfer)
    monkeypatch.setattr(module, "async_transfer", fake_transfer)
    source_path = tmp_path / "report.md"
    source_path.write_bytes(b"hello artifact\n")
    source = ArtifactSource.from_path(source_path, media_type="text/markdown")
    server = Fake()
    server.upload_state = "prepared"

    async def steps(client, w):
        return await w(client.artifacts.upload("project-test", source))

    drive(async_mode, server, steps)
    assert calls == ["prepared"]
    assert server.requests[-1].url.path.endswith("/finalize")


@MODES
@pytest.mark.parametrize("state", ["refused", "expired"])
def test_upload_refused_or_expired_is_a_typed_error_and_never_finalizes(
    tmp_path, state, async_mode
):
    source_path = tmp_path / "report.md"
    source_path.write_bytes(b"hello artifact\n")
    source = ArtifactSource.from_path(source_path, media_type="text/markdown")
    server = Fake()
    server.upload_state = state

    async def steps(client, w):
        return await w(client.artifacts.upload("project-test", source))

    with pytest.raises(ContractError) as caught:
        drive(async_mode, server, steps)
    assert caught.value.code == f"artifact_upload_{state}"
    assert not any(r.url.path.endswith("/finalize") for r in server.requests)


@MODES
def test_prepare_get_finalize_status_and_reattach(async_mode):
    server = Fake()

    async def steps(client, w):
        from synth_ai.sdk.core.artifact_types import ArtifactUploadManifest

        manifest = ArtifactUploadManifest(
            filename="report.md",
            byte_size=len(server.content),
            digest_sha256=hashlib.sha256(server.content).hexdigest(),
            media_type="text/markdown",
        )
        upload = await w(
            client.artifacts.uploads.prepare("project-test", manifest=manifest, kind="file")
        )
        again = await w(client.artifacts.uploads.get("up-1"))
        status = await w(again.status())
        receipt = await w(upload.finalize())
        return upload, status, receipt

    upload, status, receipt = drive(async_mode, server, steps)
    assert upload.upload_id == status.upload_id == receipt.upload_id == "up-1"
    assert receipt.state == "finalized"


@MODES
def test_labels_alias_roundtrip_send_exact_compare_and_set(async_mode):
    server = Fake()

    async def steps(client, w):
        labelled = await w(
            client.artifacts.update_labels(
                "project-test", "art-1", set={"tier": "gold"}, remove=("old",), expected_revision=1
            )
        )
        via_namespace = await w(
            client.artifacts.labels.update(
                "project-test", "art-1", set={"a": "b"}, expected_revision=1
            )
        )
        ref = labelled.ref
        alias = await w(
            client.artifacts.set_alias("project-test", "best", ref, expected_revision=0)
        )
        fetched = await w(client.artifacts.get_alias("project-test", "best"))
        return labelled, via_namespace, alias, fetched, ref

    labelled, via, alias, fetched, ref = drive(async_mode, server, steps)
    assert server.label_calls[0] == {
        "set": {"tier": "gold"},
        "remove": ["old"],
        "expected_revision": 1,
    }
    assert server.label_calls[1]["remove"] == []
    assert alias.ref == fetched.ref == ref and alias.name == "best"
    alias_put = [b for b in server.bodies if "/aliases/" in b[1]][0]
    assert alias_put[2]["expected_revision"] == 0 and alias_put[2]["ref"]["kind"] == "artifact"


@MODES
def test_download_is_digest_verified_and_never_overwrites(tmp_path, async_mode):
    server = Fake()
    destination = tmp_path / "out.md"

    async def steps(client, w):
        return await w(client.artifacts.download("project-test", "art-1", destination))

    receipt = drive(async_mode, server, steps)
    assert destination.read_bytes() == b"hello artifact\n"
    assert receipt.digest_sha256 == hashlib.sha256(b"hello artifact\n").hexdigest()
    assert receipt.byte_size == 15 and receipt.path == "report.md"
    with pytest.raises(FileExistsError):
        drive(async_mode, Fake(), steps)


@MODES
def test_download_with_tampered_bytes_fails_and_leaves_nothing(tmp_path, async_mode):
    server = Fake()
    original = server.__call__

    def tampered(request):
        response = original(request)
        if request.url.path.endswith("/content"):
            server.content = b"tampered bytes!\n"
            return original(request)
        return response

    destination = tmp_path / "out.md"

    async def steps(client, w):
        return await w(client.artifacts.download("project-test", "art-1", destination))

    from synth_ai.sdk.core.artifact_types import ArtifactDigestMismatch

    with pytest.raises((ArtifactDigestMismatch, ContractError)):
        drive(async_mode, tampered, steps)
    assert not destination.exists()
    assert Ref  # imported for parity with the exact-ref selection used above
