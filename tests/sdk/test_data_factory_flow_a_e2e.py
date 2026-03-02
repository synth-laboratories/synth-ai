"""End-to-end integration test for Data Factory Flow A.

Flow A: Local MCP pair-programming -> finalized dataset
Spec reference: synth_data_factory.txt section 3, lines 115-122

This test exercises the full SDK pipeline with mocked HTTP transport:
  1. Create SMR project
  2. Upload starting data (capture bundle)
  3. Submit Data Factory finalization
  4. Poll finalization status through all 4 phases
  5. Publish finalized artifacts
  6. Verify pool binding revision incremented

The test validates contracts, request shapes, and state machine
transitions without requiring a live backend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from synth_ai.sdk.managed_research import SmrControlClient

# ---------------------------------------------------------------------------
# Fake HTTP transport
# ---------------------------------------------------------------------------

_PROJECT_ID = "proj_df_e2e_001"
_RUN_ID = "run_df_finalize_001"
_JOB_ID = _RUN_ID  # finalization_job_id == run_id per backend contract


class _FakeTransport:
    """Stateful fake that models the backend Data Factory lifecycle."""

    def __init__(self) -> None:
        self.call_log: list[dict[str, Any]] = []
        self._phase_index = 0
        self._published = False
        self._pool_binding_revision = 0

        self._phases = [
            "data_factory.normalize",
            "data_factory.finalize",
            "data_factory.quality_gate",
            "data_factory.publish_prepare",
        ]

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        self.call_log.append(
            {"method": method, "path": path, "params": params, "json_body": json_body}
        )

        # ---- Project creation ----
        if method == "POST" and path == "/smr/projects":
            return {
                "project_id": _PROJECT_ID,
                "name": json_body.get("name", "Data Factory E2E"),
                "state": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        # ---- Starting data upload URLs ----
        if method == "POST" and path.endswith("/starting-data/upload-urls"):
            files = json_body.get("files", [])
            return {
                "uploads": [
                    {
                        "path": f.get("path", "unknown"),
                        "upload_url": f"https://s3.example.com/upload/{f.get('path', 'unknown')}",
                        "content_type": f.get("content_type", "application/octet-stream"),
                    }
                    for f in files
                ]
            }

        # ---- Data Factory finalize (submit) ----
        if method == "POST" and path.endswith("/data-factory/finalize") and not path.endswith("/publish"):
            return {
                "finalization_job_id": _JOB_ID,
                "run_id": _RUN_ID,
                "project_id": _PROJECT_ID,
                "state": "queued",
                "accepted_at": datetime.now(timezone.utc).isoformat(),
            }

        # ---- Data Factory finalize status (poll) ----
        if method == "GET" and "/data-factory/finalize/" in path and not path.endswith("/publish"):
            phase_states = {}
            current_phase = None
            for i, phase in enumerate(self._phases):
                if i < self._phase_index:
                    phase_states[phase] = "success"
                elif i == self._phase_index:
                    phase_states[phase] = "executing"
                    current_phase = phase
                else:
                    phase_states[phase] = "pending"

            # Advance for next poll
            self._phase_index += 1

            all_done = self._phase_index > len(self._phases)
            state = "success" if all_done else "executing"

            if all_done:
                for p in self._phases:
                    phase_states[p] = "success"
                current_phase = self._phases[-1]

            warnings = []
            if self._phase_index == 3:
                # quality_gate adds a warning in non-strict mode
                warnings.append("quality_score=0.85 (above 0.7 threshold)")

            return {
                "finalization_job_id": _JOB_ID,
                "run_id": _RUN_ID,
                "project_id": _PROJECT_ID,
                "state": state,
                "progress": {
                    "current_phase": current_phase,
                    "phases": phase_states,
                },
                "warnings": warnings,
                "errors": [],
                "artifact_manifest_uri": (
                    f"s3://smr-artifacts/{_RUN_ID}/dataset_manifest.json"
                    if all_done
                    else None
                ),
                "status_detail": {
                    "workflow_kind": "data_factory_v1",
                    "pool_binding_revision": self._pool_binding_revision,
                    "workflow_state": {
                        "phase": current_phase,
                        "phases": phase_states,
                    },
                },
            }

        # ---- Data Factory publish ----
        if method == "POST" and path.endswith("/publish"):
            self._published = True
            self._pool_binding_revision += 1
            return {
                "run_id": _RUN_ID,
                "project_id": _PROJECT_ID,
                "pool_binding_revision": self._pool_binding_revision,
                "publish_status": "pending",
                "message_id": "msg_publish_001",
            }

        # ---- Project status (post-publish verification) ----
        if method == "GET" and path.endswith(f"/projects/{_PROJECT_ID}"):
            return {
                "project_id": _PROJECT_ID,
                "name": "Data Factory E2E",
                "state": "active",
                "pool_binding_revision": self._pool_binding_revision,
            }

        raise ValueError(f"Unhandled fake request: {method} {path}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowA:
    """Flow A: Local MCP pair-programming -> finalized dataset."""

    def setup_method(self) -> None:
        self.transport = _FakeTransport()
        self.client = SmrControlClient(
            api_key="test-key",
            backend_base="http://localhost:8000",
        )
        self.client._request_json = self.transport.request_json  # type: ignore[method-assign]

    def teardown_method(self) -> None:
        self.client.close()

    # -- Step 1: Create project --

    def test_step1_create_project(self) -> None:
        result = self.client.create_project({"name": "Data Factory E2E"})

        assert result["project_id"] == _PROJECT_ID
        assert result["state"] == "active"
        call = self.transport.call_log[-1]
        assert call["method"] == "POST"
        assert call["path"] == "/smr/projects"

    # -- Step 2: Upload starting data --

    def test_step2_upload_starting_data(self) -> None:
        result = self.client.get_starting_data_upload_urls(
            _PROJECT_ID,
            files=[
                {"path": "capture_bundle.json", "content_type": "application/json"},
                {"path": "session_trace.jsonl", "content_type": "application/jsonl"},
            ],
            dataset_ref="starting-data/demo",
        )

        assert len(result["uploads"]) == 2
        assert result["uploads"][0]["path"] == "capture_bundle.json"
        call = self.transport.call_log[-1]
        assert call["method"] == "POST"
        assert "/starting-data/upload-urls" in call["path"]
        assert call["json_body"]["dataset_ref"] == "starting-data/demo"

    # -- Step 3: Submit finalization --

    def test_step3_submit_finalization(self) -> None:
        result = self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
            target_formats=["harbor"],
            preferred_target="harbor",
            finalizer_profile="founder_default",
            source_mode="mcp_local",
            runtime_kind="sandbox_agent",
            environment_kind="harbor",
            strictness_mode="warn",
        )

        assert result["finalization_job_id"] == _JOB_ID
        assert result["state"] == "queued"
        call = self.transport.call_log[-1]
        assert call["method"] == "POST"
        assert call["path"] == f"/smr/projects/{_PROJECT_ID}/data-factory/finalize"
        body = call["json_body"]
        assert body["target_formats"] == ["harbor"]
        assert body["runtime_kind"] == "sandbox_agent"
        assert body["environment_kind"] == "harbor"
        assert body["finalizer_profile"] == "founder_default"

    # -- Step 4: Poll finalization status through all phases --

    def test_step4_poll_all_phases(self) -> None:
        # Submit first
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )

        # Poll through 4 phases + final success check
        seen_phases = []
        final_status = None

        for _ in range(10):  # safety limit
            status = self.client.data_factory_finalize_status(_PROJECT_ID, _JOB_ID)
            current = status["progress"]["current_phase"]
            if current not in seen_phases:
                seen_phases.append(current)
            final_status = status
            if status["state"] == "success":
                break

        assert final_status is not None
        assert final_status["state"] == "success"
        assert final_status["artifact_manifest_uri"] is not None
        assert "dataset_manifest.json" in final_status["artifact_manifest_uri"]

        # Should have progressed through all phases
        expected = [
            "data_factory.normalize",
            "data_factory.finalize",
            "data_factory.quality_gate",
            "data_factory.publish_prepare",
        ]
        assert seen_phases == expected

        # All phases should be "success" in final status
        for phase in expected:
            assert final_status["progress"]["phases"][phase] == "success"

    # -- Step 5: Publish finalized artifacts --

    def test_step5_publish(self) -> None:
        # Complete finalization first
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        for _ in range(10):
            status = self.client.data_factory_finalize_status(_PROJECT_ID, _JOB_ID)
            if status["state"] == "success":
                break

        result = self.client.data_factory_publish(
            _PROJECT_ID,
            _JOB_ID,
            reason="manual_publish",
        )

        assert result["pool_binding_revision"] == 1
        assert result["publish_status"] == "pending"
        assert result["message_id"] == "msg_publish_001"

        call = self.transport.call_log[-1]
        assert call["method"] == "POST"
        assert call["path"].endswith(f"/{_JOB_ID}/publish")
        assert call["json_body"]["reason"] == "manual_publish"

    # -- Step 6: Verify pool binding revision --

    def test_step6_pool_binding_revision_incremented(self) -> None:
        # Run through full pipeline
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        for _ in range(10):
            status = self.client.data_factory_finalize_status(_PROJECT_ID, _JOB_ID)
            if status["state"] == "success":
                break

        # Publish
        pub = self.client.data_factory_publish(_PROJECT_ID, _JOB_ID)
        assert pub["pool_binding_revision"] == 1

        # Verify project reflects the revision
        project = self.client.get_project(_PROJECT_ID)
        assert project["pool_binding_revision"] == 1

    # -- Full pipeline: all steps in sequence --

    def test_full_pipeline_flow_a(self) -> None:
        """Complete Flow A from spec section 3, lines 115-122."""

        # 1. Create project
        project = self.client.create_project({"name": "Data Factory E2E"})
        pid = project["project_id"]
        assert pid == _PROJECT_ID

        # 2. Upload capture bundle
        urls = self.client.get_starting_data_upload_urls(
            pid,
            files=[
                {"path": "capture_bundle.json", "content_type": "application/json"},
            ],
            dataset_ref="starting-data/demo",
        )
        assert len(urls["uploads"]) == 1

        # 3. Submit finalization with Harbor target
        finalize = self.client.data_factory_finalize(
            pid,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
            target_formats=["harbor"],
            preferred_target="harbor",
            finalizer_profile="founder_default",
            runtime_kind="sandbox_agent",
            environment_kind="harbor",
        )
        job_id = finalize["finalization_job_id"]
        assert finalize["state"] == "queued"

        # 4. Poll through all phases
        phase_log = []
        for _ in range(10):
            status = self.client.data_factory_finalize_status(pid, job_id)
            phase = status["progress"]["current_phase"]
            if phase not in phase_log:
                phase_log.append(phase)
            if status["state"] == "success":
                break

        assert status["state"] == "success"
        assert status["errors"] == []
        assert status["artifact_manifest_uri"] is not None

        # 5. Publish to pool
        pub = self.client.data_factory_publish(pid, job_id, reason="launch_demo")
        assert pub["pool_binding_revision"] >= 1
        assert pub["publish_status"] == "pending"

        # 6. Verify project state
        updated_project = self.client.get_project(pid)
        assert updated_project["pool_binding_revision"] == pub["pool_binding_revision"]

        # -- Validate call sequence --
        methods_and_paths = [
            (c["method"], c["path"]) for c in self.transport.call_log
        ]

        # Project creation
        assert ("POST", "/smr/projects") in methods_and_paths

        # Upload URL request
        assert any(
            m == "POST" and "/starting-data/upload-urls" in p
            for m, p in methods_and_paths
        )

        # Finalization submit
        assert any(
            m == "POST" and p.endswith("/data-factory/finalize")
            for m, p in methods_and_paths
        )

        # Multiple status polls
        status_polls = [
            (m, p)
            for m, p in methods_and_paths
            if m == "GET" and "/data-factory/finalize/" in p
        ]
        assert len(status_polls) >= 4  # at least one per phase

        # Publish
        assert any(
            m == "POST" and p.endswith("/publish")
            for m, p in methods_and_paths
        )

        print(f"\nFlow A complete:")
        print(f"  Project: {pid}")
        print(f"  Job: {job_id}")
        print(f"  Phases: {' -> '.join(phase_log)}")
        print(f"  Pool binding revision: {pub['pool_binding_revision']}")
        print(f"  Total API calls: {len(self.transport.call_log)}")


class TestFlowAEdgeCases:
    """Edge cases and contract validation for Flow A."""

    def setup_method(self) -> None:
        self.transport = _FakeTransport()
        self.client = SmrControlClient(
            api_key="test-key",
            backend_base="http://localhost:8000",
        )
        self.client._request_json = self.transport.request_json  # type: ignore[method-assign]

    def teardown_method(self) -> None:
        self.client.close()

    def test_finalize_defaults_target_to_harbor(self) -> None:
        """When no target_formats given, defaults to [preferred_target]."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        call = self.transport.call_log[-1]
        assert call["json_body"]["target_formats"] == ["harbor"]
        assert call["json_body"]["preferred_target"] == "harbor"

    def test_finalize_omits_runtime_env_when_none(self) -> None:
        """runtime_kind and environment_kind are omitted when None."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        call = self.transport.call_log[-1]
        assert "runtime_kind" not in call["json_body"]
        assert "environment_kind" not in call["json_body"]

    def test_finalize_includes_timebox_when_set(self) -> None:
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
            timebox_seconds=900,
        )
        call = self.transport.call_log[-1]
        assert call["json_body"]["timebox_seconds"] == 900

    def test_finalize_strictness_modes(self) -> None:
        """Both warn and strict strictness modes are forwarded."""
        for mode in ["warn", "strict"]:
            self.client.data_factory_finalize(
                _PROJECT_ID,
                dataset_ref="starting-data/demo",
                bundle_manifest_path="capture_bundle.json",
                strictness_mode=mode,
            )
            call = self.transport.call_log[-1]
            assert call["json_body"]["strictness_mode"] == mode

    def test_publish_reason_forwarded(self) -> None:
        """Custom publish reason is passed to backend."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        for _ in range(10):
            s = self.client.data_factory_finalize_status(_PROJECT_ID, _JOB_ID)
            if s["state"] == "success":
                break

        self.client.data_factory_publish(
            _PROJECT_ID, _JOB_ID, reason="automated_ci_publish"
        )
        call = self.transport.call_log[-1]
        assert call["json_body"]["reason"] == "automated_ci_publish"

    def test_status_includes_quality_gate_warning(self) -> None:
        """Quality gate phase produces a warning for non-strict mode."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
        )
        saw_warning = False
        for _ in range(10):
            s = self.client.data_factory_finalize_status(_PROJECT_ID, _JOB_ID)
            if s["warnings"]:
                saw_warning = True
            if s["state"] == "success":
                break

        assert saw_warning, "Expected quality gate warning in non-strict mode"

    def test_researcher_strict_profile_forwarded(self) -> None:
        """researcher_strict profile is correctly forwarded."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
            finalizer_profile="researcher_strict",
        )
        call = self.transport.call_log[-1]
        assert call["json_body"]["finalizer_profile"] == "researcher_strict"

    def test_multiple_target_formats(self) -> None:
        """Multiple target formats can be requested."""
        self.client.data_factory_finalize(
            _PROJECT_ID,
            dataset_ref="starting-data/demo",
            bundle_manifest_path="capture_bundle.json",
            target_formats=["harbor", "openenv", "archipelago"],
            preferred_target="harbor",
        )
        call = self.transport.call_log[-1]
        assert call["json_body"]["target_formats"] == ["harbor", "openenv", "archipelago"]
        assert call["json_body"]["preferred_target"] == "harbor"
