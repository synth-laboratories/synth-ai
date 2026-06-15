"""Resolve a ManagedResearchClient from a local synth-dev slot contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _resolve_evals_root() -> Path:
    env_root = os.environ.get("EVALS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    workspace = (
        Path(os.environ.get("SYNTH_WORKSPACE_ROOT") or Path(__file__).resolve().parents[1])
        .expanduser()
        .resolve()
    )
    return (workspace / "evals").resolve()


def ensure_evals_importable() -> None:
    """Match readme_smoke path setup for ``evals.*`` and ``reportbench`` imports."""
    evals_root = _resolve_evals_root()
    if not evals_root.is_dir():
        raise FileNotFoundError(
            "evals checkout not found. Set EVALS_ROOT or SYNTH_WORKSPACE_ROOT "
            f"(looked for {evals_root})."
        )
    workspace_root = (
        Path(os.environ.get("SYNTH_WORKSPACE_ROOT") or evals_root.parent).expanduser().resolve()
    )
    synth_ai_root = Path(__file__).resolve().parents[1]
    for path in (synth_ai_root, evals_root, workspace_root):
        root_text = str(path)
        if path.is_dir() and root_text not in sys.path:
            sys.path.insert(0, root_text)


def build_managed_research_client_for_slot(
    slot: str,
    *,
    slot_mode: str | None = "local-dockerized",
    api_key: str | None = None,
    backend_base: str | None = None,
    timeout_seconds: float = 120.0,
) -> ManagedResearchClient:
    """Build an SMR client using the slot launch-target contract (slot1, etc.)."""
    if api_key and backend_base:
        return ManagedResearchClient(
            api_key=str(api_key).strip(),
            backend_base=str(backend_base).strip(),
            timeout_seconds=timeout_seconds,
        )

    ensure_evals_importable()
    from evals.launch_target_contract import (  # noqa: PLC0415
        env_from_launch_target_contract,
        load_local_launch_target_contract,
    )
    from standard.shared.core.evals_core.local_contract import (  # noqa: PLC0415
        expected_contract_path as expected_local_eval_contract_path,
    )
    from standard.shared.core.evals_core.local_contract import (
        load_local_eval_contract,
    )

    contract, contract_path = load_local_launch_target_contract(
        slot,
        target=slot_mode,
    )
    contract_env = env_from_launch_target_contract(
        contract,
        contract_path=contract_path,
    )
    local_eval = load_local_eval_contract(expected_local_eval_contract_path(slot))
    resolved_backend = (
        str(getattr(local_eval, "backend_url", "") or "").strip() or contract.network.backend_url
    )
    resolved_api_key = str(api_key or contract_env.get("SYNTH_API_KEY") or "").strip()
    if not resolved_backend or not resolved_api_key:
        raise RuntimeError(f"slot {slot!r} contract missing backend_url or SYNTH_API_KEY")
    return ManagedResearchClient(
        api_key=resolved_api_key,
        backend_base=resolved_backend,
        timeout_seconds=timeout_seconds,
    )


def actor_trace_key(actor: dict[str, Any]) -> str:
    """Prefer explicit actor_key; fall back to actor_id for trace routes."""
    for field in ("actor_key", "key"):
        value = str(actor.get(field) or "").strip()
        if value:
            return value
    return str(actor.get("actor_id") or "").strip()
