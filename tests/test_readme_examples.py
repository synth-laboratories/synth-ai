"""Offline check that README and docs/swarms.md python examples execute.

Resource methods are replaced with signature-checking autospecs, so every call in a
documented example must bind to the real SDK signature. Artifact labels must also
avoid the keys the backend reserves (422 artifact_reserved_labels).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import httpx
import pytest
from synth_ai import SynthClient
from synth_ai.options import ClientOptions
from synth_ai.sdk.core.artifacts import Artifacts
from synth_ai.sdk.index.client import IndexClient
from synth_ai.sdk.swarms.client import Projects, SwarmsClient
from synth_ai.sdk.swarms.run import SwarmRun
from synth_ai.types.refs import ExactReference

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "docs/swarms.md")
RESERVED = {"role", "task_id", "logical_output_id"}


def python_blocks(name: str) -> list[tuple[str, str]]:
    text = (ROOT / name).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return [(f"{name}#{index}", block) for index, block in enumerate(blocks)]


CASES = [case for name in DOCUMENTS for case in python_blocks(name)]
LABEL_CALLS: list[dict] = []


def record_labels(**kwargs):
    LABEL_CALLS.append(kwargs.get("labels") or {})


def offline_client() -> SynthClient:
    handler = lambda request: httpx.Response(599)  # noqa: E731 - no request may leave
    network = httpx.Client(transport=httpx.MockTransport(handler))
    return SynthClient(
        api_key="fixture",
        base_url="https://backend.test",
        options=ClientOptions(http_client=network),
    )


@pytest.mark.parametrize("label,source", CASES, ids=[label for label, _ in CASES])
def test_example_compiles_and_binds_to_real_signatures(label, source, tmp_path, monkeypatch):
    (tmp_path / "context.txt").write_text("context")
    monkeypatch.chdir(tmp_path)
    compile(source, label, "exec")
    LABEL_CALLS.clear()
    client = offline_client()

    def upload(self, *args, **kwargs):
        record_labels(**kwargs)
        fake = mock.MagicMock()
        fake.ref = ExactReference(
            authority="artifacts",
            kind="artifact",
            record_id="artifact_fixture",
            revision="1",
            digest_sha256="a" * 64,
        )
        return fake

    with (
        mock.patch.object(Artifacts, "upload", autospec=True, side_effect=upload),
        mock.patch.object(IndexClient, "search", autospec=True),
        mock.patch.object(SwarmsClient, "create", autospec=True),
        mock.patch.object(Projects, "create", autospec=True),
        mock.patch.object(SwarmRun, "wait", autospec=True),
        mock.patch.object(SwarmRun, "download", autospec=True),
        mock.patch("synth_ai.SynthClient", return_value=mock.MagicMock()) as patched,
    ):
        if "with SynthClient" in source:
            # Examples that open their own client run against a recording stand-in
            # whose swarms/artifacts attributes are the real, autospecced classes.
            stand_in = patched.return_value
            stand_in.__enter__.return_value = stand_in
            stand_in.swarms = client.swarms
            stand_in.artifacts = client.artifacts
            stand_in.index = client.index
        namespace = {"client": client, "project_id": "project_fixture"}
        exec(compile(source, label, "exec"), namespace)  # noqa: S102 - repository docs only
    for labels in LABEL_CALLS:
        reserved = {key for key in labels if key in RESERVED or key.startswith("legacy_")}
        assert not reserved, f"{label}: reserved artifact label keys {sorted(reserved)}"
    client.close()


def test_no_document_uses_reserved_label_literals():
    pattern = re.compile(r"labels\s*=\s*\{\s*\"(role|task_id|logical_output_id|legacy_\w*)\"")
    for name in DOCUMENTS:
        assert not pattern.search((ROOT / name).read_text()), name
