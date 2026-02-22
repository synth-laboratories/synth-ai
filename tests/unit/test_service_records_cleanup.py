from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from synth_ai.core.tunnels import service_records


class _FakeSocket:
    def __init__(self, result_code: int) -> None:
        self._result_code = result_code

    def settimeout(self, _timeout: float) -> None:
        return None

    def connect_ex(self, _addr) -> int:  # type: ignore[no-untyped-def]
        return self._result_code

    def close(self) -> None:
        return None


def _write_records(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_records(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cleanup_stale_records_removes_record_when_no_pid_and_port_closed(tmp_path, monkeypatch) -> None:
    records_path = tmp_path / "services.json"
    _write_records(
        records_path,
        {
            "8080": {
                "url": "http://127.0.0.1:8080",
                "port": 8080,
                "type": "local",
                "created_at": "2026-02-22T00:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(service_records, "_get_records_path", lambda: records_path)
    monkeypatch.setattr(
        "socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(111),  # non-zero => closed
    )

    service_records.cleanup_stale_records()

    assert _read_records(records_path) == {}


def test_cleanup_stale_records_keeps_record_when_no_pid_and_port_open(tmp_path, monkeypatch) -> None:
    records_path = tmp_path / "services.json"
    _write_records(
        records_path,
        {
            "8081": {
                "url": "http://127.0.0.1:8081",
                "port": 8081,
                "type": "local",
                "created_at": "2026-02-22T00:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(service_records, "_get_records_path", lambda: records_path)
    monkeypatch.setattr(
        "socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(0),  # zero => open
    )

    service_records.cleanup_stale_records()

    payload = _read_records(records_path)
    assert "8081" in payload


def test_cleanup_stale_records_keeps_running_pid_even_if_port_probe_fails(tmp_path, monkeypatch) -> None:
    records_path = tmp_path / "services.json"
    _write_records(
        records_path,
        {
            "8082": {
                "url": "http://127.0.0.1:8082",
                "port": 8082,
                "pid": 12345,
                "type": "local",
                "created_at": "2026-02-22T00:00:00+00:00",
            }
        },
    )
    monkeypatch.setattr(service_records, "_get_records_path", lambda: records_path)

    class _FakeProcess:
        def __init__(self, _pid: int) -> None:
            return None

        def is_running(self) -> bool:
            return True

    fake_psutil = types.SimpleNamespace(Process=_FakeProcess)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        "socket.socket",
        lambda *_args, **_kwargs: _FakeSocket(111),  # would fail if checked
    )

    service_records.cleanup_stale_records()

    payload = _read_records(records_path)
    assert "8082" in payload
