from synth_ai.sdk.optimization.job import JobStatus
from synth_ai.sdk.optimization.models import GraphJobStatus, PolicyJobStatus


def test_policy_job_status_synonyms() -> None:
    assert PolicyJobStatus.from_string("completed") == PolicyJobStatus.SUCCEEDED
    assert PolicyJobStatus.from_string("SUCCESS") == PolicyJobStatus.SUCCEEDED
    assert PolicyJobStatus.from_string("canceled") == PolicyJobStatus.CANCELLED
    assert PolicyJobStatus.from_string("error") == PolicyJobStatus.FAILED


def test_graph_job_status_synonyms() -> None:
    assert GraphJobStatus.from_string("completed") == GraphJobStatus.COMPLETED
    assert GraphJobStatus.from_string("success") == GraphJobStatus.SUCCEEDED
    assert GraphJobStatus.from_string("cancel") == GraphJobStatus.CANCELLED


def test_job_status_mapping() -> None:
    assert JobStatus.from_string("succeeded") == JobStatus.COMPLETED
    assert JobStatus.from_string("in_progress") == JobStatus.IN_PROGRESS
    assert JobStatus.from_string("queued") == JobStatus.PENDING
