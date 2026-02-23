from synth_ai.sdk.optimization.models import PolicyCandidate, PolicyCandidatePage


def test_policy_candidate_from_dict_prefers_canonical_artifact_fields() -> None:
    candidate = PolicyCandidate.from_dict(
        {
            "candidate_id": "cand_1",
            "candidate_type": "program_candidate",
            "artifact_kind": "program_code",
            "artifact_payload": {"candidate_code": "def solve(x): return x + 1"},
            "artifact_preview": "def solve(x): return x + 1",
            "status": "evaluated",
            "objective": {"reward": 0.82},
            "optimization_mode": "optimize_anything",
        }
    )

    assert candidate.candidate_id == "cand_1"
    assert candidate.candidate_type == "program_candidate"
    assert candidate.artifact_kind == "program_code"
    assert isinstance(candidate.artifact_payload, dict)
    assert candidate.candidate_content == "def solve(x): return x + 1"
    assert candidate.objective == 0.82
    assert candidate.score == 0.82
    assert candidate.optimization_mode == "optimize_anything"


def test_policy_candidate_page_from_dict_parses_items_and_cursor() -> None:
    page = PolicyCandidatePage.from_dict(
        {
            "items": [
                {
                    "candidate_id": "cand_1",
                    "artifact_kind": "prompt_transformation",
                    "candidate_content": "You are a classifier",
                },
                {
                    "candidate_id": "cand_2",
                    "artifact_kind": "dsl_config",
                    "artifact_payload": {"alpha": 1.23},
                },
            ],
            "next_cursor": "2026-02-23T00:00:00Z|cand_2",
            "job_id": "pl_123",
            "algorithm": "gepa",
            "mode": "offline",
        }
    )

    assert page.job_id == "pl_123"
    assert page.algorithm == "gepa"
    assert page.mode == "offline"
    assert page.next_cursor == "2026-02-23T00:00:00Z|cand_2"
    assert len(page.items) == 2
    assert page.items[0].artifact_kind == "prompt_transformation"
    assert page.items[1].artifact_kind == "dsl_config"
    assert page.items[1].artifact_payload == {"alpha": 1.23}
