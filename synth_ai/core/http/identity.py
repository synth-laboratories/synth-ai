"""One client request identity per call, retained across read attempts."""

from uuid import uuid4

from synth_ai.contracts.operations import Operation
from synth_ai.errors import SynthError
from synth_ai.options import MutationOptions, ReadOptions


def request_options(operation: Operation, options: ReadOptions | None) -> ReadOptions:
    selected = options or (MutationOptions() if operation.mutation else ReadOptions())
    return (
        selected
        if selected.request_id is not None
        else selected.model_copy(update={"request_id": uuid4().hex})
    )


def annotate_request_error(error: SynthError, operation: Operation, options: ReadOptions) -> None:
    # A capability-discovery failure keeps the identity of its actual exchange.
    error.operation = error.operation or operation.id
    error.client_request_id = error.client_request_id or options.request_id
    error.request_id = error.request_id or error.client_request_id
