"""Structured JSONL state-transition trace emitter and everystate conform verifier (SYN-6050)."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import JsonValue

__all__ = [
    "UPSTREAM_VENDOR_LEAK_RE",
    "EverystateConformResult",
    "EverystateStatus",
    "TraceConformanceError",
    "TraceRecorder",
    "TraceSpecKind",
    "find_everystate_binary",
    "find_vendor_leaks",
    "verify_trace_conformance",
    "verify_trace_python_oracle",
]

DEFAULT_SPEC_DIR = Path(__file__).resolve().parent / "specs"
DEFAULT_EVERYSTATE_TIMEOUT_SECONDS = 30.0

UPSTREAM_VENDOR_LEAK_RE = re.compile(
    r"(?i)\b(openai|anthropic|google|daytona|modal|e2b|e2b_self_hosted|"
    r"fireworks|together|openrouter|vertex|bedrock|azure|docker|reflection|"
    r"morph|exe\.dev|boat_dev)\b|\"providers\"|\"preset\"|hosted\.openai|hosted\.openrouter"
)


def find_vendor_leaks(raw_json_text: str) -> list[str]:
    """Return every forbidden upstream vendor/provider token found in raw public JSON text."""
    return [match.group(0) for match in UPSTREAM_VENDOR_LEAK_RE.finditer(raw_json_text)]


class TraceSpecKind(StrEnum):
    RUN_CONTROL = "RunControl"
    HOSTED_OPTIONS = "HostedOptions"
    BILLING_WALLET = "BillingWallet"


class EverystateStatus(StrEnum):
    CONFORMED = "conformed"
    INVARIANT_VIOLATED = "invariant_violated"
    NON_CONFORMING = "non_conforming"
    BAD_TRACE_FORMAT = "bad_trace_format"
    LIMIT_REACHED = "limit_reached"
    BINARY_UNAVAILABLE = "binary_unavailable"
    SPEC_UNAVAILABLE = "spec_unavailable"
    EXECUTION_FAILED = "execution_failed"


class TraceConformanceError(AssertionError):
    """Raised when a recorded JSONL trace violates TLA+ state-transition or safety invariants."""

    def __init__(self, spec_kind: TraceSpecKind, step_index: int, detail: str) -> None:
        super().__init__(f"[{spec_kind}] step {step_index}: {detail}")
        self.spec_kind = spec_kind
        self.step_index = step_index
        self.detail = detail


@dataclass(frozen=True, slots=True)
class EverystateConformResult:
    """Typed verification receipt combining the built-in Python oracle and everystate conform."""

    spec_kind: TraceSpecKind
    trace_path: Path
    step_count: int
    python_oracle_passed: bool
    everystate_status: EverystateStatus
    everystate_binary: str | None = None
    spec_path: Path | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise ValueError(f"trace must record at least an initial state, got {self.step_count}")

    @property
    def ok(self) -> bool:
        if not self.python_oracle_passed:
            return False
        return self.everystate_status in {
            EverystateStatus.CONFORMED,
            EverystateStatus.BINARY_UNAVAILABLE,
            EverystateStatus.SPEC_UNAVAILABLE,
        }


class TraceRecorder:
    """Records `everystate conform -trace <file.jsonl>` compatible state-transition traces."""

    def __init__(self, spec_kind: TraceSpecKind) -> None:
        self._spec_kind = spec_kind
        self._records: list[dict[str, JsonValue]] = []

    @property
    def spec_kind(self) -> TraceSpecKind:
        return self._spec_kind

    @property
    def records(self) -> Sequence[Mapping[str, JsonValue]]:
        return tuple(self._records)

    def record_init(self, state: Mapping[str, JsonValue]) -> None:
        if self._records:
            raise RuntimeError("record_init may only be called once as the first trace line")
        self._records.append({"state": dict(state)})

    def record_step(
        self,
        action: str,
        state: Mapping[str, JsonValue],
        *,
        params: Sequence[JsonValue] | Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not self._records:
            raise RuntimeError("record_init must be called before record_step")
        if not action or not action.strip():
            raise ValueError("action must be a non-empty TLA+ action name")
        entry: dict[str, JsonValue] = {"action": action, "state": dict(state)}
        if params is not None:
            entry["params"] = dict(params) if isinstance(params, Mapping) else list(params)
        self._records.append(entry)

    def to_jsonl(self) -> str:
        return (
            "\n".join(
                json.dumps(rec, sort_keys=True, separators=(",", ":")) for rec in self._records
            )
            + "\n"
        )

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")
        return path


def find_everystate_binary() -> str | None:
    """Locate the `everystate` binary on PATH or standard local install locations."""
    direct = shutil.which("everystate")
    if direct is not None:
        return direct
    for candidate in (
        Path("/opt/homebrew/bin/everystate"),
        Path.home() / ".cargo" / "bin" / "everystate",
        Path.home() / ".synth-dev" / "bin" / "everystate",
        Path.home() / ".local" / "bin" / "everystate",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _parse_trace_lines(trace_path: Path, spec_kind: TraceSpecKind) -> list[dict[str, JsonValue]]:
    raw_lines = [
        line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not raw_lines:
        raise TraceConformanceError(spec_kind, 0, "trace file is empty")
    parsed: list[dict[str, JsonValue]] = []
    for idx, line in enumerate(raw_lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceConformanceError(spec_kind, idx, f"invalid JSONL line: {exc}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("state"), dict):
            raise TraceConformanceError(
                spec_kind, idx, "line must be an object containing a 'state' object"
            )
        if idx == 0 and "action" in obj:
            raise TraceConformanceError(
                spec_kind, 0, "initial trace state must not declare 'action'"
            )
        if idx > 0:
            action = obj.get("action")
            if not isinstance(action, str) or not action.strip():
                raise TraceConformanceError(
                    spec_kind, idx, "non-initial step must declare non-empty string 'action'"
                )
        parsed.append(obj)
    return parsed


def _state_map(
    rec: Mapping[str, JsonValue], spec: TraceSpecKind, idx: int
) -> Mapping[str, JsonValue]:
    raw = rec.get("state")
    if not isinstance(raw, dict):
        raise TraceConformanceError(spec, idx, "missing 'state' mapping")
    return raw


def _require_bool(state: Mapping[str, JsonValue], key: str, spec: TraceSpecKind, idx: int) -> bool:
    val = state.get(key)
    if not isinstance(val, bool):
        raise TraceConformanceError(spec, idx, f"expected boolean for '{key}', got {val!r}")
    return val


def _require_int(state: Mapping[str, JsonValue], key: str, spec: TraceSpecKind, idx: int) -> int:
    val = state.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise TraceConformanceError(spec, idx, f"expected integer for '{key}', got {val!r}")
    return val


def _require_str(state: Mapping[str, JsonValue], key: str, spec: TraceSpecKind, idx: int) -> str:
    val = state.get(key)
    if not isinstance(val, str):
        raise TraceConformanceError(spec, idx, f"expected string for '{key}', got {val!r}")
    return val


_RUN_ACTIVE = frozenset({"queued", "running", "pausing", "paused", "cancelling"})
_RUN_TERMINAL = frozenset({"completed", "partial", "failed", "cancelled"})
_RUN_ALL = frozenset({"absent"}) | _RUN_ACTIVE | _RUN_TERMINAL


def _check_run_control_trace(records: Sequence[Mapping[str, JsonValue]]) -> None:
    spec = TraceSpecKind.RUN_CONTROL
    prev: Mapping[str, JsonValue] | None = None
    for idx, rec in enumerate(records):
        st = _state_map(rec, spec, idx)
        run_state = _require_str(st, "run_state", spec, idx)
        limits_rev = _require_int(st, "limits_revision", spec, idx)
        exec_closed = _require_bool(st, "execution_closed", spec, idx)
        acct_final = _require_bool(st, "accounting_final", spec, idx)
        cleanup_done = _require_bool(st, "cleanup_complete", spec, idx)
        msg_count = _require_int(st, "message_count", spec, idx)
        replays = _require_int(st, "idempotent_replays", spec, idx)
        conflicts = _require_int(st, "idempotency_conflicts", spec, idx)

        if run_state not in _RUN_ALL:
            raise TraceConformanceError(spec, idx, f"unknown run_state {run_state!r}")
        if min(limits_rev, msg_count, replays, conflicts) < 0:
            raise TraceConformanceError(spec, idx, "negative counter in RunControl state")
        if cleanup_done and not exec_closed:
            raise TraceConformanceError(
                spec, idx, "cleanup_complete=True requires execution_closed=True"
            )
        if acct_final and not exec_closed:
            raise TraceConformanceError(
                spec, idx, "accounting_final=True requires execution_closed=True"
            )
        if run_state in _RUN_TERMINAL and not exec_closed:
            raise TraceConformanceError(
                spec, idx, f"terminal state {run_state!r} requires execution_closed=True"
            )
        if run_state == "absent" and (
            limits_rev != 0 or exec_closed or acct_final or cleanup_done or msg_count != 0
        ):
            raise TraceConformanceError(
                spec, idx, "absent run must have zero revision/messages and false finality flags"
            )

        if prev is not None:
            p_state = _require_str(prev, "run_state", spec, idx - 1)
            if limits_rev < _require_int(prev, "limits_revision", spec, idx - 1):
                raise TraceConformanceError(spec, idx, "limits_revision regressed")
            for flag in ("execution_closed", "accounting_final", "cleanup_complete"):
                if _require_bool(prev, flag, spec, idx - 1) and not _require_bool(
                    st, flag, spec, idx
                ):
                    raise TraceConformanceError(
                        spec, idx, f"finality flag '{flag}' regressed True -> False"
                    )
            if p_state in _RUN_TERMINAL and run_state != p_state:
                raise TraceConformanceError(
                    spec,
                    idx,
                    f"terminal run_state {p_state!r} is absorbing, cannot transition to {run_state!r}",
                )
        prev = st


def _check_hosted_options_trace(records: Sequence[Mapping[str, JsonValue]]) -> None:
    spec = TraceSpecKind.HOSTED_OPTIONS
    prev: Mapping[str, JsonValue] | None = None
    for idx, rec in enumerate(records):
        st = _state_map(rec, spec, idx)
        menu_q = _require_int(st, "menu_queries", spec, idx)
        preflight_c = _require_int(st, "preflight_checks", spec, idx)
        launches = _require_int(st, "launch_attempts", spec, idx)
        leaks = _require_int(st, "vendor_leaks_detected", spec, idx)
        violations = _require_int(st, "agreement_violations", spec, idx)
        _require_bool(st, "last_menu_admissible", spec, idx)
        last_pre = _require_bool(st, "last_preflight_admissible", spec, idx)
        last_launch = _require_bool(st, "last_launch_accepted", spec, idx)

        if min(menu_q, preflight_c, launches, leaks, violations) < 0:
            raise TraceConformanceError(spec, idx, "negative counter in HostedOptions state")
        if leaks != 0:
            raise TraceConformanceError(spec, idx, f"upstream vendor leakage detected ({leaks})")
        if violations != 0:
            raise TraceConformanceError(
                spec, idx, f"cross-endpoint agreement violation ({violations})"
            )
        if idx > 0 and rec.get("action") == "LaunchAccepted" and (not last_launch or not last_pre):
            raise TraceConformanceError(
                spec, idx, "LaunchAccepted requires last_preflight_admissible=True"
            )
        if prev is not None and (
            menu_q < _require_int(prev, "menu_queries", spec, idx - 1)
            or preflight_c < _require_int(prev, "preflight_checks", spec, idx - 1)
            or launches < _require_int(prev, "launch_attempts", spec, idx - 1)
        ):
            raise TraceConformanceError(spec, idx, "HostedOptions step counters regressed")
        prev = st


def _expected_billing_caps(plan: str, payg_active: bool) -> tuple[int, int]:
    if plan == "free":
        return (10000, 50000) if payg_active else (500, 0)
    if plan == "standard.monthly":
        return (4000, 15000)
    if plan == "max.monthly":
        return (40000, 150000)
    raise ValueError(f"unknown plan {plan!r}")


def _check_billing_wallet_trace(records: Sequence[Mapping[str, JsonValue]]) -> None:
    spec = TraceSpecKind.BILLING_WALLET
    prev: Mapping[str, JsonValue] | None = None
    for idx, rec in enumerate(records):
        st = _state_map(rec, spec, idx)
        plan = _require_str(st, "plan", spec, idx)
        payg_active = _require_bool(st, "payg_credits_active", spec, idx)
        ledger = _require_int(st, "ledger_cents", spec, idx)
        held = _require_int(st, "held_cents", spec, idx)
        available = _require_int(st, "available_cents", spec, idx)
        purchased = _require_int(st, "purchased_cents", spec, idx)
        promo = _require_int(st, "promotional_cents", spec, idx)
        included = _require_int(st, "included_remaining_cents", spec, idx)
        daily_spend = _require_int(st, "daily_spend_cents", spec, idx)
        weekly_spend = _require_int(st, "weekly_spend_cents", spec, idx)
        daily_cap = _require_int(st, "daily_cap_cents", spec, idx)
        weekly_cap = _require_int(st, "weekly_cap_cents", spec, idx)
        _require_bool(st, "auto_reload_enabled", spec, idx)
        reload_daily = _require_int(st, "auto_reload_daily_count", spec, idx)
        reload_monthly = _require_int(st, "auto_reload_monthly_cents", spec, idx)
        replays = _require_int(st, "webhook_replays", spec, idx)

        if plan not in {"free", "standard.monthly", "max.monthly"}:
            raise TraceConformanceError(spec, idx, f"unknown plan {plan!r}")
        if (
            min(
                ledger,
                held,
                available,
                purchased,
                promo,
                included,
                daily_spend,
                weekly_spend,
                reload_daily,
                reload_monthly,
                replays,
            )
            < 0
        ):
            raise TraceConformanceError(spec, idx, "negative monetary or counter value")
        if available != ledger - held:
            raise TraceConformanceError(
                spec, idx, f"available ({available}) != ledger ({ledger}) - held ({held})"
            )
        if ledger != purchased + promo + included:
            raise TraceConformanceError(spec, idx, "ledger != purchased + promotional + included")
        expected_payg = plan == "free" and purchased > 0
        if payg_active != expected_payg:
            raise TraceConformanceError(spec, idx, "payg_credits_active mismatch")
        exp_daily, exp_weekly = _expected_billing_caps(plan, payg_active)
        if daily_cap != exp_daily or weekly_cap != exp_weekly:
            raise TraceConformanceError(spec, idx, "spend caps mismatch")
        if daily_spend > daily_cap:
            raise TraceConformanceError(spec, idx, "daily_spend > daily_cap")
        if weekly_cap > 0 and weekly_spend > weekly_cap:
            raise TraceConformanceError(spec, idx, "weekly_spend > weekly_cap")
        if reload_daily > 3 or reload_monthly > 500_000:
            raise TraceConformanceError(spec, idx, "auto-reload cap exceeded")
        if (
            prev is not None
            and rec.get("action") == "StripeWebhookReplay"
            and ledger != _require_int(prev, "ledger_cents", spec, idx - 1)
        ):
            raise TraceConformanceError(spec, idx, "StripeWebhookReplay mutated ledger")
        prev = st


def verify_trace_python_oracle(trace_path: Path, spec_kind: TraceSpecKind) -> int:
    """Verify JSONL schema and every TLA+ safety/transition invariant in pure Python."""
    records = _parse_trace_lines(trace_path, spec_kind)
    if spec_kind is TraceSpecKind.RUN_CONTROL:
        _check_run_control_trace(records)
    elif spec_kind is TraceSpecKind.HOSTED_OPTIONS:
        _check_hosted_options_trace(records)
    elif spec_kind is TraceSpecKind.BILLING_WALLET:
        _check_billing_wallet_trace(records)
    else:
        raise ValueError(f"Unsupported spec kind: {spec_kind}")
    return len(records)


def _run_bounded(
    argv: Sequence[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise TimeoutError(f"command timed out after {timeout_seconds}s: {argv!r}") from exc
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def verify_trace_conformance(
    trace_path: Path,
    spec_kind: TraceSpecKind,
    *,
    spec_dir: Path | None = None,
    everystate_bin: str | None = None,
    timeout_seconds: float = DEFAULT_EVERYSTATE_TIMEOUT_SECONDS,
) -> EverystateConformResult:
    """Verify trace conformance with the built-in Python oracle and `everystate conform` when installed."""
    step_count = verify_trace_python_oracle(trace_path, spec_kind)

    resolved_spec_dir = spec_dir if spec_dir is not None else DEFAULT_SPEC_DIR
    spec_path = resolved_spec_dir / f"{spec_kind.value}.tla"
    if not spec_path.is_file():
        return EverystateConformResult(
            spec_kind=spec_kind,
            trace_path=trace_path,
            step_count=step_count,
            python_oracle_passed=True,
            everystate_status=EverystateStatus.SPEC_UNAVAILABLE,
            spec_path=spec_path,
        )

    binary = everystate_bin if everystate_bin is not None else find_everystate_binary()
    if binary is None:
        return EverystateConformResult(
            spec_kind=spec_kind,
            trace_path=trace_path,
            step_count=step_count,
            python_oracle_passed=True,
            everystate_status=EverystateStatus.BINARY_UNAVAILABLE,
            spec_path=spec_path,
        )

    cmd = [binary, "conform"]
    cfg_path = spec_path.with_suffix(".cfg")
    if cfg_path.is_file():
        cmd.extend(["-config", str(cfg_path)])
    cmd.extend(["-trace", str(trace_path), str(spec_path)])
    completed = _run_bounded(cmd, timeout_seconds=timeout_seconds)
    code = completed.returncode
    status_by_code = {
        0: EverystateStatus.CONFORMED,
        12: EverystateStatus.INVARIANT_VIOLATED,
        20: EverystateStatus.NON_CONFORMING,
        21: EverystateStatus.BAD_TRACE_FORMAT,
        22: EverystateStatus.LIMIT_REACHED,
    }
    status = status_by_code.get(code, EverystateStatus.EXECUTION_FAILED)
    result = EverystateConformResult(
        spec_kind=spec_kind,
        trace_path=trace_path,
        step_count=step_count,
        python_oracle_passed=True,
        everystate_status=status,
        everystate_binary=binary,
        spec_path=spec_path,
        exit_code=code,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if not result.ok:
        raise TraceConformanceError(
            spec_kind,
            step_count - 1,
            f"everystate conform failed with status={status} exit={code}: {completed.stderr or completed.stdout}",
        )
    return result
