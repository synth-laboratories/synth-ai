from synth_ai.streaming.handlers import CLIHandler
from synth_ai.streaming.types import StreamMessage, StreamType


def test_cli_handler_prints_billing_summary(capsys):
    h = CLIHandler()

    # Simulate completed event with billing fields
    msg = StreamMessage.from_event(
        "job_1",
        {
            "seq": 1,
            "type": "prompt.learning.completed",
            "message": "Prompt learning job completed",
            "data": {
                "usd_tokens": 1.23,
                "sandbox_usd": 0.77,
                "total_usd": 2.00,
            },
            "created_at": "2025-11-03T12:00:00Z",
        },
    )
    h.handle(msg)
    out = capsys.readouterr().out
    assert "billed=$2.00" in out
    assert "sandbox $0.77" in out
    assert "tokens $1.23" in out


def test_cli_handler_prints_budget_reached(capsys):
    h = CLIHandler()
    msg = StreamMessage.from_event(
        "job_1",
        {
            "seq": 2,
            "type": "prompt.learning.budget.reached",
            "message": "Budget reached",
            "data": {
                "threshold_usd": 5.0,
                "total_usd_est": 5.1,
            },
            "created_at": "2025-11-03T12:00:01Z",
        },
    )
    h.handle(msg)
    out = capsys.readouterr().out
    assert "budget: reached $5.10 (cap $5.00)" in out


