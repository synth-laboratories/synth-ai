from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .registry import registry
from .storage.volume import storage

logger = logging.getLogger(__name__)

router = APIRouter()


class BranchRequest(BaseModel):
    env_ids: Optional[List[str]] = None
    policy_ids: Optional[List[str]] = None
    num_children: int = 1
    max_branches: int = 10


class BranchResponse(BaseModel):
    env_branches: Dict[str, List[str]]
    policy_branches: Dict[str, List[str]]


@router.post("/branch", response_model=BranchResponse)
async def create_branches(request: BranchRequest) -> BranchResponse:
    """Create branches of environments and/or policies."""

    if request.num_children > request.max_branches:
        raise HTTPException(
            status_code=422,
            detail=f"num_children ({request.num_children}) exceeds max_branches ({request.max_branches})",
        )

    env_branches = {}
    policy_branches = {}

    try:
        # Branch environments
        if request.env_ids:
            for env_id in request.env_ids:
                env_handle = registry.get_env(env_id)
                if not env_handle:
                    logger.warning(f"Environment {env_id} not found, skipping")
                    continue

                child_ids = []

                for child_idx in range(request.num_children):
                    # Create snapshot of parent
                    from .environment_routes import (
                        snapshot_environment,
                        EnvSnapshotRequest,
                    )

                    snapshot_response = await snapshot_environment(
                        EnvSnapshotRequest(env_id=env_id)
                    )

                    # Restore to new environment with modified seed
                    from .environment_routes import (
                        restore_environment,
                        EnvRestoreRequest,
                    )

                    restore_response = await restore_environment(
                        EnvRestoreRequest(snapshot_id=snapshot_response.snapshot_id)
                    )

                    child_id = restore_response.env_id
                    child_handle = registry.get_env(child_id)

                    # Update child seed for determinism
                    if child_handle and child_handle.seed is not None:
                        child_handle.seed = child_handle.seed + child_idx + 1
                        child_handle.env.seed = child_handle.seed

                    child_ids.append(child_id)

                    # Track parent relationship in snapshot metadata
                    snapshot_meta = registry.get_snapshot(snapshot_response.snapshot_id)
                    if snapshot_meta:
                        snapshot_meta.parent_snapshot_id = env_id

                env_branches[env_id] = child_ids

        # Branch policies
        if request.policy_ids:
            for policy_id in request.policy_ids:
                policy_handle = registry.get_policy(policy_id)
                if not policy_handle:
                    logger.warning(f"Policy {policy_id} not found, skipping")
                    continue

                child_ids = []

                for child_idx in range(request.num_children):
                    # Create snapshot of parent
                    from .policy_routes import snapshot_policy, PolicySnapshotRequest

                    snapshot_response = await snapshot_policy(
                        PolicySnapshotRequest(policy_id=policy_id)
                    )

                    # Restore to new policy
                    from .policy_routes import restore_policy, PolicyRestoreRequest

                    restore_response = await restore_policy(
                        PolicyRestoreRequest(snapshot_id=snapshot_response.snapshot_id)
                    )

                    child_id = restore_response.policy_id
                    child_ids.append(child_id)

                    # Copy bound environment if parent had one
                    child_handle = registry.get_policy(child_id)
                    if child_handle and policy_handle.bound_env_id:
                        # If we also branched the env, bind to corresponding child
                        if policy_handle.bound_env_id in env_branches:
                            child_envs = env_branches[policy_handle.bound_env_id]
                            if child_idx < len(child_envs):
                                child_handle.bound_env_id = child_envs[child_idx]
                        else:
                            # Otherwise keep same env binding
                            child_handle.bound_env_id = policy_handle.bound_env_id

                    # Track parent relationship
                    snapshot_meta = registry.get_snapshot(snapshot_response.snapshot_id)
                    if snapshot_meta:
                        snapshot_meta.parent_snapshot_id = policy_id

                policy_branches[policy_id] = child_ids

        return BranchResponse(
            env_branches=env_branches,
            policy_branches=policy_branches,
        )

    except Exception as e:
        logger.error(f"Failed to create branches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
