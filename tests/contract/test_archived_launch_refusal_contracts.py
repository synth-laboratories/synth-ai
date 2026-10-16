"""SYN-3998/3999: stable archive conflict uses typed launch refusal semantics."""

from synth_ai.sdk.research.errors import ResearchLaunchRefusalError


def test_archive_refusal_retains_original_intent_without_retry__SYN3998_Syn3999():
    error = ResearchLaunchRefusalError(
        "Project is archived",
        status_code=409,
        detail={
            "error_code": "project_archived",
            "message": "Project is archived",
            "project_id": "project",
            "retryable": False,
            "mutation_applied": False,
        },
        operation_id="original-launch",
    )
    assert error.operation_id == "original-launch"
    assert error.detail["error_code"] == "project_archived"
    assert error.failure.retry.retryable is False
