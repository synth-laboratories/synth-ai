"""Client observation milestones; distinct from the server's open run state."""

from enum import StrEnum


class CompletionMilestone(StrEnum):
    FINISHED = "finished"
    EXECUTION_CLOSED = "execution_closed"
    RESULT_AVAILABLE = "result_available"
