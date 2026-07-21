"""Log-oriented SDK namespace."""

from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class LogsAPI(_ClientNamespace):
    def list_run_archives(self, project_id: str, run_id: str) -> list[dict[str, object]]:
        return self._client._list_run_log_archives(project_id, run_id)


__all__ = ["LogsAPI"]
