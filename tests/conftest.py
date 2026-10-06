"""Offline discrepancy acceptance: no network or provider access."""

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Discrepancy tests forbid network/provider calls")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket.socket, "connect_ex", refused)
