"""Deadline-bound HTTP transport for explicit HTTPX use.

See backend/notes/specifications/sdk/index_contract.md. The transport owns its
network loop and pool. Arbitrary non-cooperative synchronous transports cannot
be preempted; SynthClient uses this implementation internally and does not
accept synchronous client injection. An HTTPX caller must close its client.
"""

from synth_ai.core.http.blocking import DeadlineTransport


class DeadlineHTTPTransport(DeadlineTransport):
    """Use an explicit HTTPX read timeout as the total request deadline."""

    def __init__(self, maximum_connections: int = 20) -> None:
        if type(maximum_connections) is not int or not 1 <= maximum_connections <= 1000:
            raise ValueError("maximum_connections must be between 1 and 1000")
        super().__init__(maximum_connections)


__all__ = ["DeadlineHTTPTransport"]
