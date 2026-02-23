from synth_ai.sdk.optimization.models import PolicyCandidate, PolicyCandidatePage
from synth_ai.sdk.optimization.policy.job import PolicyOptimizationJob


class _Delegate:
    def list_candidates_typed(self, **kwargs):
        assert kwargs["include"] == "artifact_payload"
        return PolicyCandidatePage.from_dict(
            {
                "items": [
                    {
                        "candidate_id": "cand_delegate_1",
                        "artifact_kind": "dsl_config",
                        "artifact_payload": {"alpha": 1.2},
                    }
                ],
                "job_id": "pl_delegate",
            }
        )

    def get_candidate_typed(self, candidate_id: str):
        assert candidate_id == "cand_delegate_2"
        return PolicyCandidate.from_dict(
            {
                "candidate_id": candidate_id,
                "artifact_kind": "program_code",
                "artifact_payload": {"candidate_code": "def solve(x): return x"},
            }
        )


def test_policy_optimization_job_typed_candidate_wrappers(monkeypatch) -> None:
    job = PolicyOptimizationJob.from_dict(
        config_dict={"policy_optimization": {"algorithm": "gepa", "container_url": "http://127.0.0.1:9999"}},
        backend_url="https://api.example.com",
        api_key="sk_test",
        container_api_key="env_test",
    )
    job._job_id = "pl_delegate"

    monkeypatch.setattr(job, "_get_delegate", lambda: _Delegate())

    page = job.list_candidates_typed(include="artifact_payload")
    assert page.job_id == "pl_delegate"
    assert len(page.items) == 1
    assert page.items[0].artifact_kind == "dsl_config"

    candidate = job.get_candidate_typed("cand_delegate_2")
    assert candidate.candidate_id == "cand_delegate_2"
    assert candidate.artifact_kind == "program_code"
