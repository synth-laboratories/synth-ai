from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class EnvHandle:
    """In-memory handle for an active environment instance."""

    env_id: str
    env: Any  # StatefulEnvironment or wrapper
    last_observation: dict[str, Any] | None
    last_info: dict[str, Any] | None
    step_idx: int
    seed: int | None
    rl_run_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PolicyHandle:
    """In-memory handle for an active policy instance."""

    policy_id: str
    policy: Any  # Policy instance
    bound_env_id: str | None
    rl_run_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RunHandle:
    """Track run status for abort support."""

    run_id: str
    status: str  # "running" | "aborted" | "completed"
    started_at: datetime
    finished_at: datetime | None = None


@dataclass
class SnapshotMeta:
    """Metadata for a stored snapshot."""

    snapshot_id: str
    kind: str  # "env" | "policy"
    rl_run_id: str
    parent_snapshot_id: str | None
    size: int
    created_at: datetime
    path: str


class Registry:
    """In-memory registries for the service."""

    def __init__(self) -> None:
        self.envs: dict[str, EnvHandle] = {}
        self.policies: dict[str, PolicyHandle] = {}
        self.runs: dict[str, RunHandle] = {}
        self.snapshots: dict[str, SnapshotMeta] = {}

    def generate_id(self) -> str:
        """Generate a UUID for unique identification."""
        return str(uuid.uuid4())

    def register_env(
        self,
        env: Any,
        seed: int | None,
        rl_run_id: str,
        last_observation: dict[str, Any] | None = None,
        last_info: dict[str, Any] | None = None,
    ) -> str:
        """Register a new environment instance."""
        env_id = self.generate_id()
        handle = EnvHandle(
            env_id=env_id,
            env=env,
            last_observation=last_observation,
            last_info=last_info,
            step_idx=0,
            seed=seed,
            rl_run_id=rl_run_id,
        )
        self.envs[env_id] = handle
        return env_id

    def register_policy(
        self,
        policy: Any,
        rl_run_id: str,
        bound_env_id: str | None = None,
    ) -> str:
        """Register a new policy instance."""
        policy_id = self.generate_id()
        handle = PolicyHandle(
            policy_id=policy_id,
            policy=policy,
            bound_env_id=bound_env_id,
            rl_run_id=rl_run_id,
        )
        self.policies[policy_id] = handle
        return policy_id

    def register_run(self, run_id: str | None = None) -> str:
        """Register a new run."""
        if run_id is None:
            run_id = self.generate_id()
        handle = RunHandle(
            run_id=run_id,
            status="running",
            started_at=datetime.utcnow(),
        )
        self.runs[run_id] = handle
        return run_id

    def abort_run(self, run_id: str) -> bool:
        """Mark a run as aborted."""
        if run_id in self.runs:
            self.runs[run_id].status = "aborted"
            self.runs[run_id].finished_at = datetime.utcnow()
            return True
        return False

    def complete_run(self, run_id: str) -> bool:
        """Mark a run as completed."""
        if run_id in self.runs:
            self.runs[run_id].status = "completed"
            self.runs[run_id].finished_at = datetime.utcnow()
            return True
        return False

    def is_run_aborted(self, run_id: str) -> bool:
        """Check if a run has been aborted."""
        return run_id in self.runs and self.runs[run_id].status == "aborted"

    def register_snapshot(
        self,
        kind: str,
        rl_run_id: str,
        size: int,
        path: str,
        parent_snapshot_id: str | None = None,
    ) -> str:
        """Register a new snapshot."""
        snapshot_id = self.generate_id()
        meta = SnapshotMeta(
            snapshot_id=snapshot_id,
            kind=kind,
            rl_run_id=rl_run_id,
            parent_snapshot_id=parent_snapshot_id,
            size=size,
            created_at=datetime.utcnow(),
            path=path,
        )
        self.snapshots[snapshot_id] = meta
        return snapshot_id

    def get_env(self, env_id: str) -> EnvHandle | None:
        """Get an environment handle by ID."""
        return self.envs.get(env_id)

    def get_policy(self, policy_id: str) -> PolicyHandle | None:
        """Get a policy handle by ID."""
        return self.policies.get(policy_id)

    def get_run(self, run_id: str) -> RunHandle | None:
        """Get a run handle by ID."""
        return self.runs.get(run_id)

    def get_snapshot(self, snapshot_id: str) -> SnapshotMeta | None:
        """Get snapshot metadata by ID."""
        return self.snapshots.get(snapshot_id)

    def remove_env(self, env_id: str) -> bool:
        """Remove an environment from the registry."""
        if env_id in self.envs:
            del self.envs[env_id]
            return True
        return False

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy from the registry."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False


# Global registry instance
registry = Registry()
