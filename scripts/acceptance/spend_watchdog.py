"""Independent J1 spend stop using the public run controls and durable local evidence."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import override

from synth_ai import SynthClient

from .launch_journal import LaunchIntent, LaunchJournal

if TYPE_CHECKING:
    from synth_ai._sync import SwarmsClient
    from synth_ai.models import RunSnapshot

LABEL = "acceptance_watchdog"
POLL_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 4
READ_FAILURES_MAX = 3
CANCEL_ATTEMPTS_MAX = 3
DISCOVERY_PAGES_MAX = 5
TERMINAL = frozenset({"completed", "partial", "failed", "cancelled"})
LEGACY_T2_CAP_USD = "0.50"


class WatchdogError(RuntimeError):
    """A local supervision refusal with no external exception or response body."""


class RunObservation(TypedDict):
    state: str
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    charge_usd: str | None
    charge_final: bool
    charge: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class WatchdogIntent:
    origin: str
    journal_directory: str
    label: str
    created_after: str
    stop_amount_usd: str
    watch_deadline: float
    settle_seconds: int
    cohort: str | None = None
    receipt_directory: str | None = None
    cohort_cap_usd: str | None = None

    def budget(self) -> CohortBudget | None:
        if self.receipt_directory is None or self.cohort is None:
            return None
        cap = self.cohort_cap_usd if self.cohort_cap_usd is not None else LEGACY_T2_CAP_USD
        return CohortBudget(Path(self.receipt_directory), self.cohort, cap)


@dataclass(slots=True)
class WatchdogState:
    outcome: str = "watching"
    run_id: str | None = None
    stop_reason: str | None = None
    stop_key: str | None = None
    cancel_attempts: int = 0
    cancel_acknowledged: bool = False
    max_seen_usd: str | None = None
    final: RunObservation | None = None
    last_error: dict[str, object] | None = None
    reconcile_deadline: float | None = None


def _write(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _amount(value: object) -> Decimal:
    try:
        amount = Decimal(str(value))
    except InvalidOperation as error:
        raise WatchdogError("watchdog_amount_invalid") from error
    if not amount.is_finite() or amount < 0:
        raise WatchdogError("watchdog_amount_invalid")
    return amount


@dataclass(slots=True)
class BudgetEntry:
    reserved_usd: str
    observed_usd: str = "0"
    final_usd: str | None = None
    run_id: str | None = None

    @property
    def charge(self) -> Decimal:
        if self.final_usd is not None:
            return _amount(self.final_usd)
        return max(_amount(self.reserved_usd), _amount(self.observed_usd))


@dataclass(frozen=True, slots=True)
class CohortBudget:
    receipt_directory: Path
    cohort: str
    maximum_spend_usd: str = LEGACY_T2_CAP_USD

    def _update(self, operation: Callable[[dict[str, BudgetEntry]], None]) -> None:
        if len(self.cohort) != 64 or any(
            character not in "0123456789abcdef" for character in self.cohort
        ):
            raise WatchdogError("watchdog_cohort_invalid")
        cap = _amount(self.maximum_spend_usd)
        if cap <= 0:
            raise WatchdogError("watchdog_cohort_cap_invalid")
        path = self.receipt_directory / f".j1-budget-{self.cohort}.json"
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a") as lock:
            for _ in range(5):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.02)
            else:
                raise WatchdogError("watchdog_budget_lock_busy")
            entries: dict[str, BudgetEntry] = {}
            if path.exists():
                retained = _read(path)
                if "entries" in retained:
                    retained_cap = _amount(retained["maximum_spend_usd"])
                    rows = retained["entries"]
                else:
                    retained_cap = _amount(LEGACY_T2_CAP_USD)
                    rows = retained
                if retained_cap != cap:
                    raise WatchdogError("watchdog_cohort_cap_changed")
                entries = {key: BudgetEntry(**entry) for key, entry in rows.items()}
            operation(entries)
            _write(
                path,
                {
                    "maximum_spend_usd": str(cap),
                    "entries": {key: asdict(entry) for key, entry in entries.items()},
                },
            )

    def reserve(self, attempt: str, maximum: str) -> None:
        def reserve(entries: dict[str, BudgetEntry]) -> None:
            if attempt in entries:
                raise WatchdogError("watchdog_budget_attempt_already_reserved")
            known = {entry.run_id for entry in entries.values() if entry.run_id is not None}
            total = sum((entry.charge for entry in entries.values()), Decimal(0))
            receipts: dict[frozenset[str], Decimal] = {}
            for path in sorted(self.receipt_directory.glob("*/receipt.json")):
                receipt = _read(path)
                if receipt["cohort"]["manifest_sha256"] != self.cohort:
                    continue
                run_ids = frozenset(receipt["run_ids"])
                if run_ids & known:
                    if len(run_ids) != 1 and receipt["spend_usd"] != "0":
                        raise WatchdogError("watchdog_budget_receipt_overlap")
                    continue
                if receipt["spend_usd"] is None:
                    raise WatchdogError("watchdog_budget_spend_unknown")
                charge = _amount(receipt["spend_usd"])
                if charge == 0:
                    continue
                if not run_ids:
                    raise WatchdogError("watchdog_budget_receipt_identity_missing")
                if run_ids in receipts:
                    if receipts[run_ids] != charge:
                        raise WatchdogError("watchdog_budget_receipt_charge_conflict")
                    continue
                if any(run_ids & previous for previous in receipts):
                    raise WatchdogError("watchdog_budget_receipt_overlap")
                receipts[run_ids] = charge
                total += charge
            reservation = _amount(maximum)
            if reservation <= 0 or total + reservation > _amount(self.maximum_spend_usd):
                raise WatchdogError("watchdog_budget_headroom_insufficient")
            entries[attempt] = BudgetEntry(maximum)

        self._update(reserve)

    def reconcile(self, attempt: str, state: WatchdogState) -> None:
        def reconcile(entries: dict[str, BudgetEntry]) -> None:
            entry = entries[attempt]
            if entry.run_id is not None and entry.run_id != state.run_id:
                raise WatchdogError("watchdog_budget_run_identity_changed")
            if state.run_id is not None and any(
                other.run_id == state.run_id for key, other in entries.items() if key != attempt
            ):
                raise WatchdogError("watchdog_budget_run_already_owned")
            entry.run_id = state.run_id
            if state.max_seen_usd is not None:
                entry.observed_usd = str(
                    max(_amount(entry.observed_usd), _amount(state.max_seen_usd))
                )
            final = None
            if state.outcome == "launch_not_dispatched":
                if state.run_id is not None:
                    raise WatchdogError("watchdog_budget_dispatch_contradiction")
                final = Decimal(0)
            elif state.outcome == "accounting_final":
                if (
                    state.run_id is None
                    or not state.final
                    or not all(
                        state.final.get(field) is True
                        for field in ("execution_closed", "accounting_final", "charge_final")
                    )
                ):
                    raise WatchdogError("watchdog_budget_accounting_not_proven")
                final = _amount(state.final["charge_usd"])
            if entry.final_usd is not None and final != _amount(entry.final_usd):
                raise WatchdogError("watchdog_budget_final_charge_changed")
            if final is not None:
                entry.final_usd = str(final)

        self._update(reconcile)


def _error(stage: str, error: Exception) -> dict[str, object]:
    return {"stage": stage, "type": type(error).__name__, "status": getattr(error, "status", None)}


def _discover(swarms: SwarmsClient, intent: WatchdogIntent) -> str | None:
    cursor = None
    matches: set[str] = set()
    for _ in range(DISCOVERY_PAGES_MAX):
        page = swarms.list(created_after=intent.created_after, cursor=cursor, limit=100)
        matches.update(
            run.run_id
            for run in page.items
            if run.labels is not None and run.labels.get(LABEL) == intent.label
        )
        if not page.has_more:
            if len(matches) > 1:
                raise WatchdogError("ambiguous_launch_label")
            return next(iter(matches), None)
        if page.next_cursor in (None, "", cursor):
            raise WatchdogError("launch_discovery_cursor_invalid")
        cursor = page.next_cursor
    raise WatchdogError("launch_discovery_incomplete")


def _observe(snapshot: RunSnapshot) -> RunObservation:
    charge = snapshot.charge
    amount = None
    if charge is not None:
        if charge.amount.currency != "USD":
            raise WatchdogError("watchdog_charge_invalid")
        amount = str(_amount(charge.amount.amount))
    return {
        "state": str(snapshot.state),
        "execution_closed": snapshot.execution_closed,
        "accounting_final": snapshot.accounting_final,
        "cleanup_complete": snapshot.cleanup_complete,
        "charge_usd": amount,
        "charge_final": charge is not None and charge.final,
        "charge": None
        if charge is None
        else {"amount": {"amount": amount, "currency": "USD"}, "final": charge.final},
    }


def supervise(
    directory: Path,
    swarms: SwarmsClient,
    *,
    parent_gone: Callable[[], bool],
    recover: bool = False,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> WatchdogState:
    """Watch or resume the same attempt. A new launch is outside this control path."""
    intent = WatchdogIntent(**_read(directory / "intent.json"))
    state_path = directory / "status.json"
    state = WatchdogState(**_read(state_path)) if state_path.is_file() else WatchdogState()
    if state.outcome in ("accounting_final", "launch_not_dispatched"):
        return state
    stop_amount = Decimal(intent.stop_amount_usd)
    if not stop_amount.is_finite() or stop_amount <= 0 or intent.settle_seconds <= 0:
        raise WatchdogError("watchdog_intent_invalid")
    journal = LaunchJournal(Path(intent.journal_directory))
    watch_end = intent.watch_deadline
    budget = intent.budget()
    failures = 0
    if recover:
        # Explicit resumption grants a bounded readback window. Keep the stop
        # key and attempted-dispatch count from the original attempt.
        state.reconcile_deadline = clock() + intent.settle_seconds
    state.outcome = "watching"

    def persist() -> None:
        _write(state_path, asdict(state))
        if budget is not None:
            try:
                budget.reconcile(intent.label, state)
            except (OSError, ValueError, TypeError, KeyError, WatchdogError) as error:
                # The reservation remains outstanding. Metering persistence must
                # never prevent an already authorized stop dispatch.
                state.last_error = _error("budget_reconcile", error)
                _write(state_path, asdict(state))

    persist()
    while clock() < (state.reconcile_deadline or watch_end + intent.settle_seconds):
        request = directory / "stop-request.json"
        requested_stop = request.is_file() or parent_gone() or clock() >= watch_end or recover
        if requested_stop and state.reconcile_deadline is None:
            state.reconcile_deadline = clock() + intent.settle_seconds
        if state.run_id is None:
            dispatched = (journal.directory / "launch-intent.json").is_file()
            if not dispatched and requested_stop:
                state.outcome = "launch_not_dispatched"
                break
            if dispatched:
                try:
                    accepted = (
                        journal.read_accepted()
                        if (journal.directory / "launch-accepted.json").is_file()
                        else None
                    )
                    if (
                        accepted is not None
                        and accepted.labels is not None
                        and LABEL in accepted.labels
                        and accepted.labels[LABEL] != intent.label
                    ):
                        raise WatchdogError("launch_receipt_label_mismatch")
                    state.run_id = (
                        accepted.run_id if accepted is not None else _discover(swarms, intent)
                    )
                except Exception as error:
                    state.last_error = _error("discover", error)
                persist()
            if state.run_id is None:
                sleep(POLL_SECONDS)
                continue

        current = None
        try:
            snapshot = swarms.get(state.run_id)
            if snapshot.run_id != state.run_id:
                raise WatchdogError("get_run_identity_mismatch")
            current = _observe(snapshot)
            amount = current["charge_usd"]
            if amount is None and not current["execution_closed"]:
                failures += 1
                state.last_error = {"stage": "get_run", "type": "charge_unobserved", "status": None}
            else:
                failures = 0
            if amount is not None and (
                state.max_seen_usd is None or Decimal(amount) > Decimal(state.max_seen_usd)
            ):
                state.max_seen_usd = amount
            state.final = current
        except Exception as error:
            failures += 1
            state.last_error = _error("get_run", error)
        last = current if current is not None else state.final
        terminal = last is not None and last["state"] in TERMINAL and last["execution_closed"]
        if terminal and last["accounting_final"] and last["charge_final"]:
            state.outcome = "accounting_final"
            break
        current_charge = current["charge_usd"] if current is not None else None
        spend_stop = current_charge is not None and Decimal(current_charge) >= stop_amount
        if (
            not terminal
            and state.stop_reason is None
            and (requested_stop or spend_stop or failures >= READ_FAILURES_MAX)
        ):
            if spend_stop:
                state.stop_reason = "spend_stop"
            elif failures >= READ_FAILURES_MAX:
                state.stop_reason = "observation_unavailable"
            elif request.is_file():
                state.stop_reason = "driver_failed"
            elif clock() >= watch_end:
                state.stop_reason = "watch_deadline"
            else:
                state.stop_reason = "driver_exited"
            state.stop_key = f"j1-watchdog-{intent.label}-{state.run_id}"
        if (terminal or state.stop_reason is not None) and state.reconcile_deadline is None:
            state.reconcile_deadline = clock() + intent.settle_seconds
        if (
            not terminal
            and state.stop_key is not None
            and not state.cancel_acknowledged
            and state.cancel_attempts < CANCEL_ATTEMPTS_MAX
        ):
            # Record the exact key and attempted dispatch before calling the
            # empty-body public control. A lost response never changes the key.
            state.cancel_attempts += 1
            persist()
            try:
                response = swarms.cancel(state.run_id, idempotency_key=state.stop_key)
                if response.run_id != state.run_id:
                    raise WatchdogError("cancel_run_identity_mismatch")
                state.cancel_acknowledged = True
            except Exception as error:
                state.last_error = _error("cancel", error)
        persist()
        sleep(POLL_SECONDS)
    else:
        state.outcome = "in_doubt"
    persist()
    return state


class SpendWatchdog:
    """Start an independent process and fail before dispatch if it cannot arm."""

    def __init__(
        self,
        directory: Path,
        *,
        journal: LaunchJournal,
        origin: str,
        key: str,
        stop_amount_usd: str,
        watch_seconds: int,
        settle_seconds: int,
        budget: CohortBudget | None = None,
    ) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(mode=0o700)
        self.label = uuid.uuid4().hex
        self.intent = WatchdogIntent(
            origin,
            str(journal.directory.resolve()),
            self.label,
            datetime.now(UTC).isoformat(),
            stop_amount_usd,
            time.time() + watch_seconds,
            settle_seconds,
            budget.cohort if budget is not None else None,
            str(budget.receipt_directory.resolve()) if budget is not None else None,
            budget.maximum_spend_usd if budget is not None else None,
        )
        _write(self.directory / "intent.json", asdict(self.intent))
        if budget is not None:
            budget.reserve(self.label, stop_amount_usd)
        source = str(Path(__file__).resolve().parents[2])
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.acceptance.spend_watchdog",
                    str(self.directory),
                    "--worker",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={
                    "PYTHONPATH": source,
                    "SYNTH_API_KEY": key,
                    "PATH": os.defpath,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                cwd=source,
                start_new_session=True,
            )
        except OSError as error:
            state = WatchdogState(
                outcome="launch_not_dispatched", last_error=_error("worker_start", error)
            )
            _write(self.directory / "status.json", asdict(state))
            if budget is not None:
                budget.reconcile(self.label, state)
            raise WatchdogError("watchdog_worker_start_failed") from error

    def __enter__(self) -> SpendWatchdog:
        deadline = time.monotonic() + 10
        while self.process.poll() is None and time.monotonic() < deadline:
            if (self.directory / "ready.json").is_file():
                return self
            time.sleep(0.05)
        self.__exit__(None, None, None)
        raise WatchdogError("watchdog_not_armed")

    def __exit__(self, *_exc: object) -> None:
        # Process separation and pipe EOF retain stop authority after a driver
        # exception, normal exit or kill. Never terminate this owned worker here.
        assert self.process.stdin is not None
        self.process.stdin.close()

    def request_stop(self) -> None:
        _write(self.directory / "stop-request.json", {"reason": "driver_failed"})

    def assert_armed(self) -> None:
        if self.process.poll() is not None or time.time() >= self.intent.watch_deadline:
            raise WatchdogError("watchdog_missing_at_dispatch")

    def launch_journal(self) -> LaunchJournal:
        return SupervisedLaunchJournal(Path(self.intent.journal_directory), self.assert_armed)

    def wait(self) -> WatchdogState:
        try:
            self.process.wait(
                timeout=max(1, self.intent.watch_deadline - time.time())
                + self.intent.settle_seconds
                + 60
            )
        except subprocess.TimeoutExpired as error:
            raise WatchdogError("watchdog_wait_timed_out") from error
        path = self.directory / "status.json"
        if not path.is_file():
            raise WatchdogError("watchdog_result_missing")
        state = WatchdogState(**_read(path))
        if state.outcome == "watching":
            raise WatchdogError("watchdog_interrupted_resume_required")
        if self.process.returncode and state.outcome in (
            "accounting_final",
            "launch_not_dispatched",
        ):
            raise WatchdogError("watchdog_budget_reconcile_required")
        return state


@dataclass(frozen=True, slots=True)
class SupervisedLaunchJournal(LaunchJournal):
    assert_armed: Callable[[], None]

    @override
    def begin(self, intent: LaunchIntent) -> None:
        self.assert_armed()
        LaunchJournal.begin(self, intent)
        self.assert_armed()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("SYNTH_API_KEY")
    if key is None or key == "":
        raise WatchdogError("watchdog_api_key_missing")
    directory = args.directory.resolve()
    intent = WatchdogIntent(**_read(directory / "intent.json"))
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with SynthClient(
            key, base_url=intent.origin, timeout=REQUEST_TIMEOUT_SECONDS, retries=0
        ) as client:
            if args.worker:
                _write(directory / "ready.json", {"armed": True})
            state = supervise(
                directory,
                client.swarms,
                recover=args.recover,
                parent_gone=lambda: args.recover or bool(select.select([sys.stdin], [], [], 0)[0]),
            )
        if (budget := intent.budget()) is not None:
            budget.reconcile(intent.label, state)
    return 0 if state.outcome in ("accounting_final", "launch_not_dispatched") else 1


if __name__ == "__main__":
    raise SystemExit(main())
