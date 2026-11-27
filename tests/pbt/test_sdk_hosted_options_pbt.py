"""SDK Hypothesis RuleBasedStateMachine suite 2: Hosted options & supplier-neutrality (SYN-6050)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

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

from synth_ai._generated.models import LaunchChoices, LaunchRequest, RoleAssignment
from synth_ai.errors import SwarmsAPIError

from .config import SdkPbtConfig
from .mock_transport import SdkPbtHarness
from .trace import TraceRecorder, TraceSpecKind, find_vendor_leaks, verify_trace_conformance

__all__ = ["test_sdk_hosted_options_stateful_pbt"]

DomicileChoice = Literal["us_only", "us_or_europe", "unrestricted"]


def _make_sdk_hosted_options_machine(
    *,
    config: SdkPbtConfig,
    trace_dir: Path,
) -> type[RuleBasedStateMachine]:
    run_counter = {"n": 0}

    class SdkHostedOptionsMachine(RuleBasedStateMachine):
        """Drives SynthClient.swarms launch_choices -> preflight_request -> launch_request."""

        def __init__(self) -> None:
            super().__init__()
            run_counter["n"] += 1
            self._instance_id = run_counter["n"]
            self.harness = SdkPbtHarness(config)
            self.trace = TraceRecorder(TraceSpecKind.HOSTED_OPTIONS)
            self.menu_queries = 0
            self.preflight_checks = 0
            self.launch_attempts = 0
            self.vendor_leaks_detected = 0
            self.agreement_violations = 0
            self.last_menu: LaunchChoices | None = None
            self.last_domicile: DomicileChoice = "unrestricted"
            self.last_workers: int = 1
            self.last_wall_time: int = 3600
            self.last_charge_str: str = "5.00"
            self.last_profile: str = "swarms-v2"
            self.last_request: LaunchRequest | None = None
            self.last_menu_admissible = False
            self.last_preflight_admissible = False
            self.last_launch_accepted = False
            self.step_seq = 0
            self.trace.record_init(self._tla_state())

        def _tla_state(self) -> dict[str, JsonValue]:
            return {
                "menu_queries": self.menu_queries,
                "preflight_checks": self.preflight_checks,
                "launch_attempts": self.launch_attempts,
                "vendor_leaks_detected": self.vendor_leaks_detected,
                "agreement_violations": self.agreement_violations,
                "last_menu_admissible": self.last_menu_admissible,
                "last_preflight_admissible": self.last_preflight_admissible,
                "last_launch_accepted": self.last_launch_accepted,
            }

        def _assert_last_wire_supplier_neutral(self, endpoint: str) -> None:
            if self.harness.backend is not None and self.harness.backend.wire_calls:
                raw = self.harness.backend.wire_calls[-1].raw_text
                leaks = find_vendor_leaks(raw)
                if leaks:
                    self.vendor_leaks_detected += len(leaks)
                    raise AssertionError(f"Vendor leak on {endpoint}: {leaks[:5]}")

        @rule(
            profile=st.sampled_from(["swarms-v2"]),
            domicile=st.sampled_from(["us_only", "us_or_europe", "unrestricted"]),
            parallel_workers=st.integers(min_value=1, max_value=6),
            wall_time_seconds=st.sampled_from([600, 1800, 3600, 14400]),
            maximum_charge_cents=st.sampled_from([50, 150, 500, 1500, 5000]),
        )
        def query_launch_choices(
            self,
            profile: str,
            domicile: DomicileChoice,
            parallel_workers: int,
            wall_time_seconds: int,
            maximum_charge_cents: int,
        ) -> None:
            charge_str = f"{maximum_charge_cents // 100}.{maximum_charge_cents % 100:02d}"
            menu = self.harness.client.swarms.launch_choices(
                profile=profile,
                domicile_requirement=domicile,
                parallel_workers=parallel_workers,
                wall_time_seconds=wall_time_seconds,
                budget=charge_str,
            )
            self._assert_last_wire_supplier_neutral("GET /v1/launch-choices")
            assert find_vendor_leaks(menu.model_dump_json()) == []
            assert menu.menu_revision.startswith("lcm1_")
            if domicile == "us_only":
                for sbx in menu.sandbox_profiles:
                    assert sbx.domicile == "us"

            self.menu_queries += 1
            self.last_menu = menu
            self.last_profile = profile
            self.last_domicile = domicile
            self.last_workers = parallel_workers
            self.last_wall_time = wall_time_seconds
            self.last_charge_str = charge_str
            self.last_menu_admissible = bool(menu.inference_profiles and menu.sandbox_profiles)
            self.trace.record_step("QueryLaunchChoices", self._tla_state())

        @precondition(lambda self: self.last_menu is not None)
        @rule(use_excluded_choice=st.booleans())
        def preflight_from_menu(self, use_excluded_choice: bool) -> None:
            assert self.last_menu is not None
            menu = self.last_menu
            assignments: dict[str, RoleAssignment] = {}
            expect_admissible = True

            if use_excluded_choice and menu.exclusions:
                excl = menu.exclusions[0]
                if excl.kind.value == "inference":
                    assignments["worker"] = RoleAssignment(inference_profile=excl.profile_id)
                else:
                    assignments["worker"] = RoleAssignment(sandbox_profile=excl.profile_id)
                expect_admissible = False
            elif menu.inference_profiles and menu.sandbox_profiles:
                inf = menu.inference_profiles[0]
                sbx = menu.sandbox_profiles[0]
                assignments["worker"] = RoleAssignment(
                    inference_profile=inf.profile_id,
                    sandbox_profile=sbx.profile_id,
                    sandbox_option=sbx.options[0].option_id,
                )
            elif menu.inference_profiles:
                assignments["worker"] = RoleAssignment(
                    inference_profile=menu.inference_profiles[0].profile_id,
                    planner_decides=["sandbox"],
                )
            else:
                assignments["worker"] = RoleAssignment(inference_profile="nonexistent-profile")
                expect_admissible = False

            req = self.harness.client.swarms.prepare_launch(
                "SDK hosted options PBT check",
                budget=self.last_charge_str,
                profile=self.last_profile,
                domicile_requirement=self.last_domicile,
                parallel_workers=self.last_workers,
                wall_time_seconds=self.last_wall_time,
                role_assignments=assignments,
                launch_menu_revision=menu.menu_revision,
            )
            pre = self.harness.client.swarms.preflight_request(req)
            self._assert_last_wire_supplier_neutral("POST /v1/runs/preflight")
            assert find_vendor_leaks(pre.model_dump_json()) == []
            if not self.harness.is_live and pre.admissible != expect_admissible:
                self.agreement_violations += 1
                raise AssertionError(
                    f"Menu/preflight disagreement: expected {expect_admissible}, got {pre.admissible}"
                )

            self.preflight_checks += 1
            self.last_request = req
            self.last_preflight_admissible = pre.admissible
            action = "PreflightAdmissible" if pre.admissible else "PreflightRefused"
            self.trace.record_step(action, self._tla_state())

        @precondition(lambda self: self.last_request is not None and self.preflight_checks > 0)
        @rule(corrupt_revision=st.booleans())
        def launch_matches_preflight(self, corrupt_revision: bool) -> None:
            assert self.last_request is not None
            expect_accept = self.last_preflight_admissible
            if corrupt_revision:
                raw = self.last_request.model_dump(mode="json", exclude_none=True)
                raw["launch_menu_revision"] = "lcm1_" + "f" * 64
                req = LaunchRequest.model_validate(raw)
                expect_accept = False
            else:
                req = self.last_request

            self.step_seq += 1
            key = f"sdk-hosted-{self._instance_id}-{self.step_seq}"
            self.launch_attempts += 1
            if expect_accept:
                snap = self.harness.client.swarms.launch_request(req, idempotency_key=key)
                self._assert_last_wire_supplier_neutral("POST /v1/runs")
                assert find_vendor_leaks(snap.model_dump_json()) == []
                self.last_launch_accepted = True
                self.trace.record_step("LaunchAccepted", self._tla_state())
            else:
                with pytest.raises(SwarmsAPIError) as exc_info:
                    self.harness.client.swarms.launch_request(req, idempotency_key=key)
                assert exc_info.value.status in {409, 422}
                self._assert_last_wire_supplier_neutral("POST /v1/runs (refusal)")
                self.last_launch_accepted = False
                self.trace.record_step("LaunchRefused", self._tla_state())

        @invariant()
        def zero_leaks_and_zero_disagreements(self) -> None:
            assert self.vendor_leaks_detected == 0
            assert self.agreement_violations == 0

        @override
        def teardown(self) -> None:
            try:
                out_path = trace_dir / f"sdk_hosted_options_{self._instance_id:04d}.jsonl"
                self.trace.write_jsonl(out_path)
                result = verify_trace_conformance(
                    out_path, TraceSpecKind.HOSTED_OPTIONS, spec_dir=config.spec_dir
                )
                assert result.ok
            finally:
                self.harness.close()

    return SdkHostedOptionsMachine


def test_sdk_hosted_options_stateful_pbt(sdk_pbt_config: SdkPbtConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="sdk-pbt-hosted-options-") as tmp:
        trace_dir = sdk_pbt_config.trace_out if sdk_pbt_config.trace_out is not None else Path(tmp)
        machine_cls = _make_sdk_hosted_options_machine(config=sdk_pbt_config, trace_dir=trace_dir)
        run_state_machine_as_test(
            machine_cls,
            settings=settings(
                max_examples=sdk_pbt_config.max_examples,
                stateful_step_count=sdk_pbt_config.stateful_step_count,
                deadline=None,
                suppress_health_check=[HealthCheck.too_slow],
            ),
        )
