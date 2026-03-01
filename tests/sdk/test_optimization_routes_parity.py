"""Snapshot tests for Python route builder behavior.

Ensures route construction matches expected strings for all route types, v1/v2.
"""

from __future__ import annotations

import pytest

from synth_ai.core.utils.optimization_routes import (
    EVAL_API_VERSION,
    GEPA_API_VERSION,
    MIPRO_API_VERSION,
    candidate_path,
    candidate_subpath,
    normalize_api_version,
    offline_job_path,
    offline_job_subpath,
    offline_jobs_base,
    online_session_path,
    online_session_subpath,
    online_sessions_base,
    policy_system_path,
    policy_systems_base,
    system_subpath,
)


class TestApiVersionConstants:
    def test_gepa_version(self):
        assert GEPA_API_VERSION == "v2"

    def test_mipro_version(self):
        assert MIPRO_API_VERSION == "v1"

    def test_eval_version(self):
        assert EVAL_API_VERSION == "v2"


class TestNormalizeApiVersion:
    @pytest.mark.parametrize("raw,expected", [("v1", "v1"), ("v2", "v2"), ("V1", "v1"), (" v2 ", "v2")])
    def test_valid(self, raw, expected):
        assert normalize_api_version(raw) == expected

    @pytest.mark.parametrize("raw", ["v3", "", "1", "version1"])
    def test_invalid(self, raw):
        with pytest.raises(ValueError):
            normalize_api_version(raw)


class TestOfflineJobs:
    @pytest.mark.parametrize("version,expected", [
        ("v1", "/v1/offline/jobs"),
        ("v2", "/v2/offline/jobs"),
    ])
    def test_base(self, version, expected):
        assert offline_jobs_base(api_version=version) == expected

    @pytest.mark.parametrize("version,job_id,expected", [
        ("v1", "abc-123", "/v1/offline/jobs/abc-123"),
        ("v2", "xyz", "/v2/offline/jobs/xyz"),
    ])
    def test_job_path(self, version, job_id, expected):
        assert offline_job_path(job_id, api_version=version) == expected

    @pytest.mark.parametrize("version,job_id,suffix,expected", [
        ("v1", "abc", "events", "/v1/offline/jobs/abc/events"),
        ("v1", "abc", "/events", "/v1/offline/jobs/abc/events"),
        ("v2", "abc", "events/stream", "/v2/offline/jobs/abc/events/stream"),
        ("v2", "abc", "artifacts", "/v2/offline/jobs/abc/artifacts"),
        ("v1", "abc", "metrics", "/v1/offline/jobs/abc/metrics"),
    ])
    def test_job_subpath(self, version, job_id, suffix, expected):
        assert offline_job_subpath(job_id, suffix, api_version=version) == expected


class TestOnlineSessions:
    @pytest.mark.parametrize("version,expected", [
        ("v1", "/v1/online/sessions"),
        ("v2", "/v2/online/sessions"),
    ])
    def test_base(self, version, expected):
        assert online_sessions_base(api_version=version) == expected

    @pytest.mark.parametrize("version,session_id,expected", [
        ("v1", "sess-1", "/v1/online/sessions/sess-1"),
        ("v2", "sess-2", "/v2/online/sessions/sess-2"),
    ])
    def test_session_path(self, version, session_id, expected):
        assert online_session_path(session_id, api_version=version) == expected

    @pytest.mark.parametrize("version,session_id,suffix,expected", [
        ("v1", "s1", "reward", "/v1/online/sessions/s1/reward"),
        ("v2", "s1", "/events", "/v2/online/sessions/s1/events"),
    ])
    def test_session_subpath(self, version, session_id, suffix, expected):
        assert online_session_subpath(session_id, suffix, api_version=version) == expected


class TestPolicySystems:
    @pytest.mark.parametrize("version,expected", [
        ("v1", "/v1/policy-optimization/systems"),
        ("v2", "/v2/policy-optimization/systems"),
    ])
    def test_base(self, version, expected):
        assert policy_systems_base(api_version=version) == expected

    @pytest.mark.parametrize("version,system_id,expected", [
        ("v1", "sys-1", "/v1/policy-optimization/systems/sys-1"),
        ("v2", "sys-2", "/v2/policy-optimization/systems/sys-2"),
    ])
    def test_system_path(self, version, system_id, expected):
        assert policy_system_path(system_id, api_version=version) == expected


class TestSystems:
    @pytest.mark.parametrize("version,system_id,suffix,expected", [
        ("v1", "sys-1", "candidates", "/v1/systems/sys-1/candidates"),
        ("v2", "sys-2", "/candidates", "/v2/systems/sys-2/candidates"),
        ("v1", "sys-1", "seed-evals", "/v1/systems/sys-1/seed-evals"),
        ("v2", "sys-2", "/seed-evals", "/v2/systems/sys-2/seed-evals"),
    ])
    def test_system_subpath(self, version, system_id, suffix, expected):
        assert system_subpath(system_id, suffix, api_version=version) == expected


class TestCandidates:
    @pytest.mark.parametrize("version,candidate_id,expected", [
        ("v1", "cand-1", "/v1/candidates/cand-1"),
        ("v2", "cand-2", "/v2/candidates/cand-2"),
    ])
    def test_candidate_path(self, version, candidate_id, expected):
        assert candidate_path(candidate_id, api_version=version) == expected

    @pytest.mark.parametrize("version,candidate_id,suffix,expected", [
        ("v1", "cand-1", "seed-evals", "/v1/candidates/cand-1/seed-evals"),
        ("v2", "cand-2", "/seed-evals", "/v2/candidates/cand-2/seed-evals"),
    ])
    def test_candidate_subpath(self, version, candidate_id, suffix, expected):
        assert candidate_subpath(candidate_id, suffix, api_version=version) == expected
