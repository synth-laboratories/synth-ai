"""J1 spend supervision controls exercise the public SDK wire with no provider."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from typing_extensions import override

from scripts.acceptance.spend_watchdog import (
    LABEL,
    CohortBudget,
    WatchdogError,
    WatchdogIntent,
    WatchdogState,
    _read,
    _write,
    supervise,
)
from synth_ai import SynthClient

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
FOREIGN = "run_AAAAAAAAAAAAAAAAAAAAAQ"
COHORT = "a" * 64
NONCE = "fixture-attempt"


def observation(contract, *, state="running", amount="0.21", final=False, run=RUN, label=NONCE):
    example = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    return {
        **example,
        "state": state,
        "run_id": run,
        "labels": {LABEL: label},
        "execution_closed": state == "cancelled",
        "accounting_final": final,
        "cleanup_complete": False,
        "charge": None
        if amount is None
        else {
            "amount": {"currency": "USD", "amount": amount},
            "final": final,
        },
    }


def attempt(tmp_path, *, accepted=None, budget=None):
    journal_path = tmp_path / "launch"
    journal_path.mkdir()
    _write(journal_path / "launch-intent.json", {"fixture": True})
    if accepted is not None:
        _write(journal_path / "launch-accepted.json", accepted)
    directory = tmp_path / "watchdog"
    directory.mkdir()
    _write(
        directory / "intent.json",
        asdict(
            WatchdogIntent(
                "http://127.0.0.1:8123",
                str(journal_path),
                NONCE,
                "2026-11-17T00:00:00Z",
                "0.20",
                60,
                40,
                COHORT if budget else None,
                str(budget.receipt_directory) if budget else None,
                budget.maximum_spend_usd if budget else None,
            )
        ),
    )
    return directory


def run_worker(directory, transport, *, parent_gone=lambda: False, recover=False):
    now = 0

    def sleep(seconds):
        nonlocal now
        now += seconds

    api = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    with SynthClient(
        "offline-fixture", base_url="http://127.0.0.1:8123", http_client=api, retries=0
    ) as client:
        return supervise(
            directory,
            client.swarms,
            parent_gone=parent_gone,
            recover=recover,
            clock=lambda: now,
            sleep=sleep,
        )


def test_transport_failure_pre_dispatch_and_lost_ack_use_same_empty_cancel(contract, tmp_path):
    running = observation(contract)
    terminal = observation(contract, state="cancelled", final=True)
    directory = attempt(tmp_path, accepted=running)
    observations = 0
    keys = []

    def transport(request):
        nonlocal observations
        if request.method == "GET":
            observations += 1
            if observations == 1:
                raise httpx.ConnectError("getRun unavailable", request=request)
            return httpx.Response(200, json=terminal if len(keys) == 3 else running)
        assert request.url.path == f"/v1/runs/{RUN}/cancel"
        assert request.content == b""
        keys.append(request.headers["idempotency-key"])
        retained = _read(directory / "status.json")
        assert retained["cancel_attempts"] == len(keys)
        assert retained["stop_key"] == keys[-1]
        assert not retained["cancel_acknowledged"]
        if len(keys) == 1:
            raise httpx.ConnectError("cancel not dispatched", request=request)
        if len(keys) == 2:
            raise httpx.ReadError("cancel dispatched, response lost", request=request)
        return httpx.Response(202, json=running)

    result = run_worker(directory, transport)
    assert len(keys) == 3 and len(set(keys)) == 1
    assert result.stop_reason == "spend_stop"
    assert result.cancel_acknowledged
    assert result.outcome == "accounting_final" and result.final["charge_usd"] == "0.21"
    assert not result.final["cleanup_complete"]


def test_lost_stop_ack_is_reconciled_without_claiming_acknowledgement(contract, tmp_path):
    running = observation(contract)
    directory = attempt(tmp_path, accepted=running)
    keys = []

    def disconnected(request):
        if request.method == "GET":
            return httpx.Response(200, json=running)
        assert request.content == b""
        keys.append(request.headers["idempotency-key"])
        raise httpx.ReadError("response lost", request=request)

    unresolved = run_worker(directory, disconnected)
    assert unresolved.outcome == "in_doubt" and unresolved.stop_reason == "spend_stop"
    assert len(keys) == 3 and len(set(keys)) == 1
    assert not unresolved.cancel_acknowledged

    def recovered(request):
        assert request.method == "GET"
        return httpx.Response(200, json=observation(contract, state="cancelled", final=True))

    final = run_worker(directory, recovered, recover=True)
    assert final.outcome == "accounting_final" and final.final["charge_usd"] == "0.21"
    assert (final.stop_key, final.cancel_attempts) == (keys[0], 3)
    assert not final.cancel_acknowledged


@pytest.mark.parametrize("missing_charge", [False, True])
def test_blind_observation_requests_stop_after_bounded_reads(contract, tmp_path, missing_charge):
    running = observation(contract, amount=None if missing_charge else "0")
    terminal = observation(contract, state="cancelled", amount="0.07", final=True)
    directory = attempt(tmp_path, accepted=running)
    reads = 0
    writes = 0

    def transport(request):
        nonlocal reads, writes
        if request.method == "POST":
            assert reads == 3 and request.content == b""
            writes += 1
            return httpx.Response(202, json=running)
        if writes:
            return httpx.Response(200, json=terminal)
        reads += 1
        if not missing_charge:
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(200, json=running)

    result = run_worker(directory, transport)
    assert writes == 1 and result.stop_reason == "observation_unavailable"
    assert result.outcome == "accounting_final"
    assert result.max_seen_usd == "0.07"


def test_driver_exit_recovers_accepted_run_from_exact_label_and_never_relaunches(
    contract, tmp_path
):
    directory = attempt(tmp_path)
    running = observation(contract, amount="0.05")
    pending = observation(contract, state="cancelled", amount="0.06")
    terminal = observation(contract, state="cancelled", amount="0.06", final=True)
    seen = []

    def transport(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/v1/runs":
            return httpx.Response(
                200,
                json={
                    "has_more": False,
                    "next_cursor": None,
                    "items": [
                        observation(contract, run=FOREIGN, label="different-attempt"),
                        running,
                    ],
                },
            )
        if request.method == "POST":
            assert request.url.path == f"/v1/runs/{RUN}/cancel"
            return httpx.Response(202, json=running)
        count = seen.count(("GET", f"/v1/runs/{RUN}"))
        return httpx.Response(200, json=[running, pending, terminal][count - 1])

    result = run_worker(directory, transport, parent_gone=lambda: True)
    assert result.run_id == RUN and result.stop_reason == "driver_exited"
    assert result.outcome == "accounting_final" and result.final["charge_usd"] == "0.06"
    assert seen.count(("POST", f"/v1/runs/{RUN}/cancel")) == 1


@pytest.mark.parametrize(("cap", "previous"), [("0.50", "0.15"), ("2.00", "1.65")])
def test_budget_reserves_concurrently_and_reconciles_once(tmp_path, cap, previous):
    receipt_dir = tmp_path / "t2"
    receipt_dir.mkdir()
    peer = receipt_dir / "cancel"
    peer.mkdir()
    _write(
        peer / "receipt.json",
        {
            "cohort": {"manifest_sha256": COHORT},
            "run_ids": [FOREIGN],
            "spend_usd": previous,
        },
    )
    budget = CohortBudget(receipt_dir, COHORT, cap)

    def reserve(key):
        try:
            budget.reserve(key, "0.20")
        except WatchdogError as error:
            if str(error) != "watchdog_budget_headroom_insufficient":
                raise
            return None
        return key

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = [key for key in pool.map(reserve, ("live", "hosted")) if key is not None]
    assert len(accepted) == 1
    budget.reconcile(accepted[0], WatchdogState(outcome="in_doubt", run_id=RUN))
    assert reserve("still_unresolved") is None

    final = WatchdogState(
        outcome="accounting_final",
        run_id=RUN,
        final={
            "state": "completed",
            "execution_closed": True,
            "accounting_final": True,
            "cleanup_complete": True,
            "charge_final": True,
            "charge_usd": "0.12",
            "charge": {"amount": {"currency": "USD", "amount": "0.12"}, "final": True},
        },
    )
    budget.reconcile(accepted[0], final)
    budget.reconcile(accepted[0], final)
    mirror = receipt_dir / "live_j1"
    mirror.mkdir()
    _write(
        mirror / "receipt.json",
        {
            "cohort": {"manifest_sha256": COHORT},
            "run_ids": [RUN],
            "spend_usd": "0.12",
        },
    )
    assert reserve("after_final") == "after_final"
    assert _read(receipt_dir / f".j1-budget-{COHORT}.json")["maximum_spend_usd"] == cap


def test_old_ledger_and_intent_keep_the_original_cap(contract, tmp_path):
    path = tmp_path / f".j1-budget-{COHORT}.json"
    _write(path, {NONCE: {"reserved_usd": "0.20"}})
    original = path.read_bytes()
    with pytest.raises(WatchdogError, match="watchdog_cohort_cap_changed"):
        CohortBudget(tmp_path, COHORT, "2.00").reserve("new", "0.20")
    assert path.read_bytes() == original
    with pytest.raises(WatchdogError, match="watchdog_budget_headroom_insufficient"):
        CohortBudget(tmp_path, COHORT).reserve("new", "0.31")

    directory = attempt(
        tmp_path, accepted=observation(contract), budget=CohortBudget(tmp_path, COHORT)
    )
    intent_path = directory / "intent.json"
    old_intent = _read(intent_path)
    del old_intent["cohort_cap_usd"]
    _write(intent_path, old_intent)
    terminal = observation(contract, state="cancelled", amount="0.12", final=True)
    state = run_worker(directory, lambda _: httpx.Response(200, json=terminal), recover=True)
    assert state.outcome == "accounting_final"
    assert _read(path)["maximum_spend_usd"] == "0.50"
    assert _read(path)["entries"][NONCE]["final_usd"] == "0.12"


def test_current_cap_boundary_is_pinned_per_cohort(tmp_path):
    current = CohortBudget(tmp_path, COHORT, "2.00")
    current.reserve("live", "1.00")
    current.reserve("hosted", "1.00")
    with pytest.raises(WatchdogError, match="watchdog_budget_headroom_insufficient"):
        current.reserve("extra", "0.01")
    with pytest.raises(WatchdogError, match="watchdog_cohort_cap_changed"):
        CohortBudget(tmp_path, COHORT, "1.99").reserve("changed", "0.01")
    assert set(_read(tmp_path / f".j1-budget-{COHORT}.json")["entries"]) == {"live", "hosted"}


def test_dispatched_without_run_id_keeps_budget_outstanding(tmp_path):
    budget = CohortBudget(tmp_path, COHORT)
    budget.reserve(NONCE, "0.20")
    directory = attempt(tmp_path, budget=budget)
    seen = []

    def transport(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"has_more": False, "next_cursor": None, "items": []})

    result = run_worker(directory, transport, parent_gone=lambda: True)
    assert result.outcome == "in_doubt" and result.run_id is None
    assert set(seen) == {("GET", "/v1/runs")}
    with pytest.raises(WatchdogError, match="watchdog_budget_headroom_insufficient"):
        budget.reserve("next", "0.31")


def test_budget_releases_reservation_when_launch_intent_is_absent(tmp_path):
    budget = CohortBudget(tmp_path, COHORT)
    budget.reserve(NONCE, "0.20")
    directory = attempt(tmp_path, budget=budget)
    (tmp_path / "launch" / "launch-intent.json").unlink()

    def transport(request):
        pytest.fail(f"unexpected {request.method} control")

    result = run_worker(directory, transport, parent_gone=lambda: True)
    assert result.outcome == "launch_not_dispatched"
    budget.reconcile(NONCE, result)
    budget.reserve("replacement", "0.50")


def test_detached_worker_outlives_failed_driver_process(contract, tmp_path):
    cancelled = threading.Event()
    requests = []
    directory = tmp_path / "watchdog"

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, document):
            payload = json.dumps(document).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            assert self.path == f"/v1/runs/{RUN}"
            if cancelled.is_set():
                self.respond(
                    200, observation(contract, state="cancelled", amount="0.08", final=True)
                )
            else:
                self.respond(200, observation(contract, amount="0.07"))

        def do_POST(self):
            requests.append(
                (
                    self.path,
                    self.headers.get("Idempotency-Key"),
                    int(self.headers.get("Content-Length", "0")),
                )
            )
            cancelled.set()
            self.respond(202, observation(contract, amount="0.07"))

        @override
        def log_message(self, format: str, *args: Any) -> None:
            return

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            command = """
import os
import sys
from pathlib import Path
from scripts.acceptance.launch_journal import LaunchIntent, LaunchJournal
from scripts.acceptance.spend_watchdog import LABEL, CohortBudget, SpendWatchdog
from synth_ai.models import RunSnapshot
out = Path(sys.argv[1])
journal = LaunchJournal(out / "launch")
watchdog = SpendWatchdog(out / "watchdog", journal=journal, origin=sys.argv[2], key="offline-fixture",
                         stop_amount_usd="0.20", watch_seconds=30, settle_seconds=15,
                         budget=CohortBudget(out, "a" * 64, "2.00"))
with watchdog:
    watchdog.launch_journal().begin(LaunchIntent("sdk", "fixture", "a" * 64, ()))
    snapshot = RunSnapshot.model_validate_json(sys.argv[3])
    journal.accepted(snapshot.model_copy(update={"labels": {LABEL: watchdog.label}}))
    os._exit(17)
"""
            parent = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    command,
                    str(tmp_path),
                    f"http://127.0.0.1:{server.server_port}",
                    json.dumps(observation(contract, amount="0.07")),
                ],
                capture_output=True,
                timeout=15,
            )
            assert parent.returncode == 17, parent.stderr.decode()
            ledger = tmp_path / f".j1-budget-{COHORT}.json"
            intent = _read(directory / "intent.json")
            assert intent["cohort_cap_usd"] == "2.00"
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                path = directory / "status.json"
                entry = _read(ledger)["entries"][intent["label"]]
                if path.is_file() and entry["final_usd"] is not None:
                    break
                time.sleep(0.10)
            status = _read(directory / "status.json")
            assert status["outcome"] == "accounting_final"
            assert status["stop_reason"] == "driver_exited"
            assert _read(ledger)["maximum_spend_usd"] == "2.00"
            assert _read(ledger)["entries"][intent["label"]]["final_usd"] == "0.08"
            assert requests == [(f"/v1/runs/{RUN}/cancel", status["stop_key"], 0)]
        finally:
            server.shutdown()
            worker.join()
