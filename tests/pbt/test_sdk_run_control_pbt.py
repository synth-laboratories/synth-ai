"""SDK Hypothesis RuleBasedStateMachine suite 1: Run control & idempotency races (SYN-6050)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from pydantic import JsonValue
from typing_extensions import override

from synth_ai._generated.models import RunSnapshot
from synth_ai.errors import ConflictError

from .config import SdkPbtConfig
from .mock_transport import SdkPbtHarness
from .trace import TraceRecorder, TraceSpecKind, verify_trace_conformance

__all__ = ["test_sdk_run_control_stateful_pbt"]

_ACTIVE_STATES = frozenset({"queued", "running", "pausing", "paused", "cancelling"})
_TERMINAL_STATES = frozenset({"completed", "partial", "failed", "cancelled"})


def _make_sdk_run_control_machine(
    *,
    config: SdkPbtConfig,
    trace_dir: Path,
) -> type[RuleBasedStateMachine]:
    run_counter = {"n": 0}

    class SdkRunControlMachine(RuleBasedStateMachine):
        """Drives real SynthClient.swarms controls verifying ETag, Idempotency-Key, revisions, and finality."""

        def __init__(self) -> None:
            super().__init__()
            run_counter["n"] += 1
            self._instance_id = run_counter["n"]
            self.harness = SdkPbtHarness(config)
            self.trace = TraceRecorder(TraceSpecKind.RUN_CONTROL)
            self.run_id: str | None = None
            self.create_key: str | None = None
            self.create_budget_str: str | None = None
            self.create_snapshot: RunSnapshot | None = None
            self.last_etag: str | None = None
            self.last_snapshot: RunSnapshot | None = None
            self.observed_state: str = "absent"
            self.observed_limits_revision: int = 0
            self.observed_charge_cents: int = 0
            self.execution_closed: bool = False
            self.accounting_final: bool = False
            self.cleanup_complete: bool = False
            self.message_count: int = 0
            self.idempotent_replays: int = 0
            self.idempotency_conflicts: int = 0
            self.step_seq: int = 0
            self.trace.record_init(self._tla_state())

        def _tla_state(self) -> dict[str, JsonValue]:
            return {
                "run_state": self.observed_state,
                "limits_revision": self.observed_limits_revision,
                "execution_closed": self.execution_closed,
                "accounting_final": self.accounting_final,
                "cleanup_complete": self.cleanup_complete,
                "message_count": self.message_count,
                "idempotent_replays": self.idempotent_replays,
                "idempotency_conflicts": self.idempotency_conflicts,
            }

        def _next_key(self, prefix: str) -> str:
            self.step_seq += 1
            return f"sdk-pbt-{prefix}-{self._instance_id}-{self.step_seq}"

        def _sync_snapshot(self, snap: RunSnapshot) -> None:
            new_state = snap.state
            if self.observed_state in _TERMINAL_STATES:
                assert new_state == self.observed_state
            rev = (
                snap.limits_revision
                if snap.limits_revision is not None
                else self.observed_limits_revision
            )
            assert rev >= self.observed_limits_revision
            if self.execution_closed:
                assert snap.execution_closed
            if self.accounting_final:
                assert snap.accounting_final
            if self.cleanup_complete:
                assert snap.cleanup_complete
            self.observed_state = new_state
            self.observed_limits_revision = rev
            self.execution_closed = snap.execution_closed
            self.accounting_final = snap.accounting_final
            self.cleanup_complete = snap.cleanup_complete

        @precondition(lambda self: self.run_id is None)
        @rule(budget_cents=st.sampled_from([100, 250, 500, 1000]))
        def create_run(self, budget_cents: int) -> None:
            key = self._next_key("create")
            budget_str = f"{budget_cents // 100}.{budget_cents % 100:02d}"
            snap = self.harness.client.swarms.launch(
                "SDK stateful PBT run control task",
                budget=budget_str,
                wall_time_seconds=3600,
                idempotency_key=key,
            )
            self.run_id = snap.run_id
            self.create_key = key
            self.create_budget_str = budget_str
            self.create_snapshot = snap
            self.observed_charge_cents = budget_cents
            self._sync_snapshot(snap)
            self.trace.record_step("CreateRun", self._tla_state())

        @precondition(lambda self: self.run_id is not None and self.create_key is not None)
        @rule()
        def replay_create_same_payload(self) -> None:
            assert self.create_key is not None and self.create_budget_str is not None
            snap = self.harness.client.swarms.launch(
                "SDK stateful PBT run control task",
                budget=self.create_budget_str,
                wall_time_seconds=3600,
                idempotency_key=self.create_key,
            )
            assert snap == self.create_snapshot
            if self.harness.backend is not None:
                assert self.harness.backend.wire_calls[-1].idempotent_replayed
            self.idempotent_replays += 1
            self.trace.record_step("ReplaySameKey", self._tla_state())

        @precondition(lambda self: self.run_id is not None and self.create_key is not None)
        @rule(extra_cents=st.sampled_from([100, 200, 500]))
        def replay_create_mutated_payload(self, extra_cents: int) -> None:
            assert self.create_key is not None
            mutated_cents = self.observed_charge_cents + extra_cents
            mutated_str = f"{mutated_cents // 100}.{mutated_cents % 100:02d}"
            with pytest.raises(ConflictError) as exc_info:
                self.harness.client.swarms.launch(
                    "Mutated SDK payload reusing original key",
                    budget=mutated_str,
                    wall_time_seconds=3600,
                    idempotency_key=self.create_key,
                )
            assert exc_info.value.status == 409
            self.idempotency_conflicts += 1
            self.trace.record_step("ReplayMutatedKeyRefused", self._tla_state())

        @precondition(lambda self: self.run_id is not None)
        @rule()
        def get_run_and_verify_etag(self) -> None:
            assert self.run_id is not None
            snap = self.harness.client.swarms.get(self.run_id)
            self._sync_snapshot(snap)
            if self.harness.backend is not None:
                wire = self.harness.backend.wire_calls[-1]
                assert wire.etag is not None
                expected = f'"{hashlib.sha256(wire.raw_text.encode("utf-8")).hexdigest()}"'
                assert wire.etag == expected
                if self.last_snapshot == snap and self.last_etag is not None:
                    assert wire.etag == self.last_etag
                self.last_etag = wire.etag
                self.last_snapshot = snap
                cond = self.harness.raw_conditional_get(
                    f"/v1/runs/{self.run_id}", if_none_match=wire.etag
                )
                assert cond.status_code == 304
            self.trace.record_step("ObserveSnapshot", self._tla_state())

        @precondition(lambda self: self.run_id is not None and self.observed_state == "queued")
        @rule()
        def start_queued_run(self) -> None:
            assert self.run_id is not None
            if self.harness.backend is not None:
                self.harness.backend.advance_run_state(self.run_id, "start")
                self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
                self.trace.record_step("StartRun", self._tla_state())

        @precondition(
            lambda self: (
                self.run_id is not None and self.observed_state in {"queued", "running", "pausing"}
            )
        )
        @rule()
        def pause_run(self) -> None:
            assert self.run_id is not None
            key = self._next_key("pause")
            receipt = self.harness.client.swarms.pause(self.run_id, idempotency_key=key)
            assert receipt.command == "pause"
            replay = self.harness.client.swarms.pause(self.run_id, idempotency_key=key)
            assert replay == receipt
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            self.trace.record_step("PauseRun", self._tla_state())

        @precondition(
            lambda self: self.run_id is not None and self.observed_state in {"pausing", "paused"}
        )
        @rule()
        def resume_run(self) -> None:
            assert self.run_id is not None
            receipt = self.harness.client.swarms.resume(
                self.run_id, idempotency_key=self._next_key("resume")
            )
            assert receipt.command == "resume"
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            self.trace.record_step("ResumeRun", self._tla_state())

        @precondition(
            lambda self: self.run_id is not None and self.observed_state in _ACTIVE_STATES
        )
        @rule(raise_cents=st.sampled_from([50, 100, 250]), also_resume=st.booleans())
        def update_limits_raise(self, raise_cents: int, also_resume: bool) -> None:
            assert self.run_id is not None
            prev_rev = self.observed_limits_revision
            was_paused = self.observed_state in {"pausing", "paused"}
            new_charge = self.observed_charge_cents + raise_cents
            progress = self.harness.client.swarms.update_limits(
                self.run_id,
                expected_revision=prev_rev,
                budget=f"{new_charge // 100}.{new_charge % 100:02d}",
                resume=bool(also_resume and was_paused),
                idempotency_key=self._next_key("limits"),
            )
            assert progress.revision == prev_rev + 1
            self.observed_charge_cents = new_charge
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            action = "UpdateLimitsAndResume" if (also_resume and was_paused) else "UpdateLimits"
            self.trace.record_step(action, self._tla_state())

        @precondition(lambda self: self.run_id is not None)
        @rule()
        def update_limits_stale_revision(self) -> None:
            assert self.run_id is not None
            stale = self.observed_limits_revision + 10
            with pytest.raises(ConflictError) as exc_info:
                self.harness.client.swarms.update_limits(
                    self.run_id,
                    expected_revision=stale,
                    budget="99.00",
                    idempotency_key=self._next_key("stale"),
                )
            assert exc_info.value.status == 409
            self.trace.record_step("RefuseStaleLimits", self._tla_state())

        @precondition(
            lambda self: self.run_id is not None and self.observed_state in _ACTIVE_STATES
        )
        @rule(note=st.sampled_from(["Focus on precision.", "Verify edge cases."]))
        def send_message_and_replay(self, note: str) -> None:
            assert self.run_id is not None
            key = self._next_key("msg")
            msg = self.harness.client.swarms.send_message(self.run_id, note, idempotency_key=key)
            self.message_count += 1
            assert msg.sequence == self.message_count
            self.trace.record_step("SendMessage", self._tla_state())
            replay = self.harness.client.swarms.send_message(self.run_id, note, idempotency_key=key)
            assert replay == msg
            self.idempotent_replays += 1
            self.trace.record_step("ReplaySameKey", self._tla_state())

        @precondition(
            lambda self: self.run_id is not None and self.observed_state in _ACTIVE_STATES
        )
        @rule()
        def cancel_run(self) -> None:
            assert self.run_id is not None
            snap = self.harness.client.swarms.cancel(
                self.run_id, idempotency_key=self._next_key("cancel")
            )
            self._sync_snapshot(snap)
            self.trace.record_step("CancelRun", self._tla_state())

        @precondition(
            lambda self: (
                self.run_id is not None and self.observed_state in {"running", "cancelling"}
            )
        )
        @rule()
        def close_execution(self) -> None:
            assert self.run_id is not None
            if self.harness.backend is None:
                return
            action_name = (
                "CloseExecutionCancelled"
                if self.observed_state == "cancelling"
                else "CloseExecutionCompleted"
            )
            self.harness.backend.advance_run_state(
                self.run_id, "close_cancel" if self.observed_state == "cancelling" else "complete"
            )
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            self.trace.record_step(action_name, self._tla_state())

        @precondition(
            lambda self: (
                self.run_id is not None and self.execution_closed and not self.accounting_final
            )
        )
        @rule()
        def finalize_accounting(self) -> None:
            assert self.run_id is not None
            if self.harness.backend is None:
                return
            self.harness.backend.advance_run_state(self.run_id, "finalize_accounting")
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            self.trace.record_step("FinalizeAccounting", self._tla_state())

        @precondition(
            lambda self: (
                self.run_id is not None and self.execution_closed and not self.cleanup_complete
            )
        )
        @rule()
        def complete_cleanup(self) -> None:
            assert self.run_id is not None
            if self.harness.backend is None:
                return
            self.harness.backend.advance_run_state(self.run_id, "complete_cleanup")
            self._sync_snapshot(self.harness.client.swarms.get(self.run_id))
            self.trace.record_step("CompleteCleanup", self._tla_state())

        @precondition(
            lambda self: self.run_id is not None and self.observed_state in _TERMINAL_STATES
        )
        @rule(action=st.sampled_from(["pause", "resume", "cancel"]))
        def terminal_mutations_refused(self, action: str) -> None:
            assert self.run_id is not None
            key = self._next_key(f"term-{action}")
            swarms = self.harness.client.swarms
            action_fn = {"pause": swarms.pause, "resume": swarms.resume, "cancel": swarms.cancel}[
                action
            ]
            with pytest.raises(ConflictError) as exc_info:
                action_fn(self.run_id, idempotency_key=key)
            assert exc_info.value.status == 409
            self.trace.record_step("RefuseTerminalMutation", self._tla_state())

        @invariant()
        def finality_and_revision_invariants(self) -> None:
            assert self.observed_limits_revision >= 0
            if (
                self.cleanup_complete
                or self.accounting_final
                or self.observed_state in _TERMINAL_STATES
            ):
                assert self.execution_closed

        @override
        def teardown(self) -> None:
            try:
                out_path = trace_dir / f"sdk_run_control_{self._instance_id:04d}.jsonl"
                self.trace.write_jsonl(out_path)
                result = verify_trace_conformance(
                    out_path, TraceSpecKind.RUN_CONTROL, spec_dir=config.spec_dir
                )
                assert result.ok
            finally:
                self.harness.close()

    return SdkRunControlMachine


def test_sdk_run_control_stateful_pbt(sdk_pbt_config: SdkPbtConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="sdk-pbt-run-control-") as tmp:
        trace_dir = sdk_pbt_config.trace_out if sdk_pbt_config.trace_out is not None else Path(tmp)
        machine_cls = _make_sdk_run_control_machine(config=sdk_pbt_config, trace_dir=trace_dir)
        run_state_machine_as_test(
            machine_cls,
            settings=settings(
                max_examples=sdk_pbt_config.max_examples,
                stateful_step_count=sdk_pbt_config.stateful_step_count,
                deadline=None,
                suppress_health_check=[HealthCheck.too_slow],
            ),
        )
