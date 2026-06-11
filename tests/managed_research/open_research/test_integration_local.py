"""Local integration test against a Track C backend on 127.0.0.1:8000.

Skipped by default. Opt in by exporting
``MANAGED_RESEARCH_OPEN_RESEARCH_LOCAL=1`` (and optionally
``MANAGED_RESEARCH_OPEN_RESEARCH_BACKEND_BASE`` to override the host).
Covers the Test 6 read-side flow plus one anonymous submission.
"""

from __future__ import annotations

import os
import socket

import pytest
from synth_ai.managed_research.open_research import (
    OpenResearchClient,
    OpenResearchError,
    SubmitQuestionArgs,
    load_or_create_fingerprint,
)
from synth_ai.managed_research.open_research.models import MetricTarget

_LOCAL_FLAG = "MANAGED_RESEARCH_OPEN_RESEARCH_LOCAL"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


def _backend_base() -> str:
    return os.environ.get(
        "MANAGED_RESEARCH_OPEN_RESEARCH_BACKEND_BASE",
        f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}",
    )


def _local_backend_reachable() -> bool:
    if os.environ.get(_LOCAL_FLAG, "").strip() != "1":
        return False
    try:
        with socket.create_connection((_DEFAULT_HOST, _DEFAULT_PORT), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.integration_local


@pytest.mark.skipif(
    not _local_backend_reachable(),
    reason=(
        "Set MANAGED_RESEARCH_OPEN_RESEARCH_LOCAL=1 and run the Track C "
        "backend on 127.0.0.1:8000 to enable this test."
    ),
)
def test_local_open_research_read_flow_round_trip(tmp_path) -> None:
    """list_projects -> get_project -> list_queues -> submit -> get_submission -> list_experiments."""
    fingerprint = load_or_create_fingerprint(tmp_path / "fp")
    with OpenResearchClient(
        fingerprint=fingerprint,
        backend_base=_backend_base(),
    ) as client:
        projects = client.list_projects()
        assert projects.projects, "expected at least one Open Research project locally"
        first = projects.projects[0]

        detail = client.get_project(first.slug)
        assert detail.slug == first.slug
        assert detail.default_queue_id

        queues = client.list_queues(project_slug=first.slug)
        oed_queue = next(
            (q for q in queues.queues if q.unsigned_in_allowed),
            None,
        )
        if oed_queue is None:
            pytest.skip("Local backend has no unsigned-in queue to exercise.")

        try:
            submission = client.submit_question(
                SubmitQuestionArgs(
                    project_slug=first.slug,
                    queue_id=oed_queue.id,
                    prompt="Local Open Research MCP smoke submission.",
                    hypothesis="",
                    metric_target=MetricTarget(
                        name="craftax.reward.mean",
                        operator=">=",
                        value=0.0,
                    ),
                    deo_kind="open_ended_discovery",
                    rubric_acknowledged=True,
                    submitter_handle="anon-mcp-test",
                    submitter_fingerprint=fingerprint,
                )
            )
        except OpenResearchError as err:
            pytest.skip(
                f"Local backend rejected smoke submission ({err.error_class}): {err.actionable}"
            )

        assert submission.submission_id

        detail_after = client.get_submission(submission.submission_id)
        assert detail_after.submission_id == submission.submission_id
        assert detail_after.project_slug == first.slug

        experiments = client.list_experiments(project_slug=first.slug, limit=5)
        # Even if no experiments yet, the response shape must validate.
        assert isinstance(experiments.experiments, list)
