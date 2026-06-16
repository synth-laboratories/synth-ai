"""Validate vendored synth-ai managed-research contract artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_public_models() -> list[str]:
    path = _repo_root() / "synth_ai" / "managed_research" / "schemas" / "public_models.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit(f"Invalid public model snapshot: {path}")
    return [
        str(item.get("id") or "").strip()
        for item in models
        if isinstance(item, dict) and bool(item.get("public", True))
    ]


def _load_sdk_model_values() -> list[str]:
    namespace: dict[str, object] = {}
    model_module = (
        _repo_root() / "synth_ai" / "managed_research" / "models" / "smr_agent_models.py"
    )
    exec(model_module.read_text(encoding="utf-8"), namespace)
    values = namespace.get("SMR_AGENT_MODEL_VALUES")
    if not isinstance(values, tuple):
        raise SystemExit(f"Could not load SMR_AGENT_MODEL_VALUES from {model_module}")
    return [str(item) for item in values]


def main() -> None:
    schema_dir = _repo_root() / "synth_ai" / "managed_research" / "schemas"
    openapi_snapshot = schema_dir / "smr_openapi.yaml"
    if not openapi_snapshot.exists() or openapi_snapshot.stat().st_size == 0:
        raise SystemExit(f"Missing vendored OpenAPI snapshot: {openapi_snapshot}")

    vendored_models = _load_public_models()
    sdk_models = _load_sdk_model_values()
    if vendored_models != sdk_models:
        raise SystemExit(
            "Public model drift detected between vendored snapshot and SDK enum:\n"
            f"  snapshot: {vendored_models}\n"
            f"  sdk: {sdk_models}"
        )

    print("synth-ai managed-research contract artifacts validated")


if __name__ == "__main__":
    main()
