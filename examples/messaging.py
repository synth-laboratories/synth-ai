"""Send a question during an admitted run; see docs/quickstart.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synth_ai.models import MessageKind, RunMessage

if TYPE_CHECKING:
    from synth_ai import AsyncSynthClient

__all__ = ["ask_during_run"]


async def ask_during_run(
    client: AsyncSynthClient,
    run_id: str,
    *,
    text: str,
    idempotency_key: str,
    timeout: float,
) -> RunMessage:
    """Return the correlated reply; a timeout leaves the message unchanged."""
    question = await client.swarms.send_message(
        run_id, text, kind=MessageKind.QUESTION, idempotency_key=idempotency_key
    )
    return await client.swarms.wait_for_reply(run_id, question.message_id, timeout=timeout)
