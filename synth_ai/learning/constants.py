from __future__ import annotations

# Terminal statuses normalized across FT and RL
TERMINAL_STATUSES = {
    "succeeded",
    "failed",
    "cancelled",
    "canceled",
    "error",
    "completed",
}

# Terminal event types (success/failure) across FT and RL
TERMINAL_EVENT_SUCCESS = {
    "sft.completed",
    "sft.workflow.completed",
    "rl.job.completed",
    "rl.train.completed",
    "workflow.completed",
}

TERMINAL_EVENT_FAILURE = {
    "sft.failed",
    "sft.workflow.failed",
    "rl.job.failed",
    "workflow.failed",
}
