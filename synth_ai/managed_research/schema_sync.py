"""Schema and enum sync helpers for the synth-ai Managed Research surface."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path


def sync_public_schemas(
    *,
    source_dir: Path | None = None,
    destination_dir: Path | None = None,
) -> list[Path]:
    """Copy quarantined legacy schemas into the new generated schema folder."""

    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parents[1]
    source = source_dir or (repo_root / "old" / "schemas" / "generated")
    destination = destination_dir or (package_root / "models" / "generated")
    copied: list[Path] = []
    if not source.exists():
        return copied
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _enum_member_name(model_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").upper()
    if not name:
        raise ValueError(f"Unable to derive enum name for '{model_id}'")
    if name[0].isdigit():
        name = f"MODEL_{name}"
    return name


def _default_backend_manifest_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "backend" / "config" / "smr_supported_models.json"


def _default_backend_supported_models_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return (
        workspace_root / "backend" / "packages" / "smr" / "config" / "supported_models_catalog.py"
    )


def _default_backend_public_models_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "backend" / "config" / "smr_public_models.json"


def _default_backend_actor_policy_path() -> Path:
    """Legacy JSON bridge path (deleted when policy moved code-first).

    Prefer `_load_actor_policy_manifest_from_backend_python()`. The JSON path remains
    only as an explicit override for offline sync when a exported manifest is provided.
    """

    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "backend" / "config" / "smr_actor_model_policy.json"


def _default_backend_actor_role_gates_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return (
        workspace_root
        / "backend"
        / "packages"
        / "smr"
        / "config"
        / "actor_configurations"
        / "actor_role_gates.py"
    )


def _backend_python_import_paths() -> tuple[Path, ...]:
    workspace_root = Path(__file__).resolve().parents[3]
    backend_root = workspace_root / "backend"
    return (backend_root / "packages", backend_root)


def _load_actor_policy_manifest_from_backend_python() -> dict[str, object]:
    """Load the live actor-policy manifest from backend code-first registries.

    Authority: `backend/packages/smr/config/actor_configurations/actor_role_gates.py`
    via `smr.config.actor_model_policy.load_smr_actor_model_policy_entries`.
    """

    import sys

    import_paths = _backend_python_import_paths()
    missing = [str(path) for path in import_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Managed Research actor policy sync requires a local backend checkout; "
            f"missing: {', '.join(missing)}"
        )
    inserted: list[str] = []
    for path in reversed(import_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted.append(path_str)
    try:
        from smr.config.actor_model_policy import (  # type: ignore[import-not-found]
            SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES,
            load_smr_actor_model_policy_entries,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent
        for path_str in inserted:
            if path_str in sys.path:
                sys.path.remove(path_str)
        raise RuntimeError(
            "Failed to import backend smr.config.actor_model_policy for actor policy sync. "
            "Ensure backend/packages is importable (PYTHONPATH=backend/packages:backend)."
        ) from exc

    policies: list[dict[str, object]] = []
    for entry in load_smr_actor_model_policy_entries():
        policies.append(
            {
                "actor_type": entry.actor_type.value,
                "actor_subtype": entry.actor_subtype.value,
                "public": bool(entry.public),
                "permitted_models": [model.value for model in entry.permitted_models],
            }
        )
    if not policies:
        raise ValueError("Backend actor model policy produced an empty policies list")
    return {
        "schema_version": "smr.actor_model_policy.v1",
        "source": "backend/packages/smr/config/actor_configurations/actor_role_gates.py",
        "shared_top_level_agent_models": list(SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES),
        "policies": policies,
    }


def _load_actor_policy_manifest(source: Path | None) -> dict[str, object]:
    if source is not None:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Actor model policy manifest must be a JSON object")
        return raw
    try:
        return _load_actor_policy_manifest_from_backend_python()
    except Exception as primary_exc:
        legacy = _default_backend_actor_policy_path()
        if legacy.is_file():
            raw = json.loads(legacy.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Actor model policy manifest must be a JSON object") from primary_exc
            return raw
        raise RuntimeError(
            "Unable to sync Managed Research actor model policy from backend Python "
            f"({_default_backend_actor_role_gates_path()}) and no legacy JSON exists at "
            f"{legacy}. Original error: {primary_exc}"
        ) from primary_exc


def _shared_top_level_model_ids(policies: list[dict[str, object]]) -> tuple[str, ...]:
    """Mirror backend/packages/smr/config/actor_model_policy.py intersection logic."""

    public_sets: list[set[str]] = []
    for item in policies:
        if not isinstance(item, dict) or not bool(item.get("public", True)):
            continue
        models = item.get("permitted_models")
        if not isinstance(models, list) or not models:
            continue
        normalized = {str(model_id).strip() for model_id in models if str(model_id).strip()}
        if normalized:
            public_sets.append(normalized)
    if not public_sets:
        return ()
    return tuple(sorted(set.intersection(*public_sets)))


def _normalize_actor_policy_entries(
    policies: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in policies:
        if not isinstance(item, dict):
            raise ValueError("Each actor model policy entry must be an object")
        actor_type = str(item.get("actor_type") or "").strip()
        actor_subtype = str(item.get("actor_subtype") or "").strip()
        if not actor_type or not actor_subtype:
            raise ValueError("Actor model policy entries require actor_type and actor_subtype")
        key = f"{actor_type}:{actor_subtype}"
        if key in seen:
            raise ValueError(f"Duplicate actor model policy entry '{key}'")
        seen.add(key)
        models = item.get("permitted_models")
        if not isinstance(models, list) or not models:
            raise ValueError(f"Actor model policy entry '{key}' must define permitted_models")
        permitted_models = list(
            dict.fromkeys(str(model_id).strip() for model_id in models if str(model_id).strip())
        )
        if not permitted_models:
            raise ValueError(f"Actor model policy entry '{key}' must define permitted_models")
        entries.append(
            {
                "actor_type": actor_type,
                "actor_subtype": actor_subtype,
                "permitted_models": permitted_models,
            }
        )
    return tuple(entries)


_STATIC_ENUM_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "smr_agent_kinds.py",
        "SmrAgentKind",
        "agent_kind",
        ("codex", "opencode_sdk"),
    ),
    (
        "smr_funding_sources.py",
        "SmrFundingSource",
        "funding_source",
        ("synth_managed", "customer_byok", "user_connected"),
    ),
    (
        "smr_credential_providers.py",
        "SmrCredentialProvider",
        "provider",
        ("deepseek", "openai", "openrouter", "xai", "tinker"),
    ),
    (
        "smr_inference_providers.py",
        "SmrInferenceProvider",
        "inference_provider",
        ("deepseek", "openai", "google", "openrouter", "xai"),
    ),
    (
        "smr_tool_providers.py",
        "SmrToolProvider",
        "tool_provider",
        ("tinker", "sublinear", "linear"),
    ),
    (
        "smr_work_modes.py",
        "SmrWorkMode",
        "work_mode",
        ("general", "open_ended_discovery", "directed_effort"),
    ),
    (
        "smr_resource_providers.py",
        "SmrResourceProvider",
        "resource_provider",
        ("modal",),
    ),
    (
        "smr_resource_kinds.py",
        "SmrResourceKind",
        "resource_kind",
        ("pod", "sandbox", "app"),
    ),
)

_SMR_AGENT_MODEL_COMPATIBILITY_VALUES: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "cursor/grok-4.5",
    "gpt-5.4",
)


def _render_static_enum_module(
    *,
    class_name: str,
    field_name: str,
    values: tuple[str, ...],
    docstring: str,
) -> str:
    values_constant_name = re.sub(r"[^A-Za-z0-9]+", "_", class_name).strip("_").upper()
    values_constant_name = f"{values_constant_name}_VALUES"
    coerce_name = "coerce_" + re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    lines = [
        f'"""{docstring}."""',
        "",
        "from __future__ import annotations",
        "",
        "from enum import StrEnum",
        "",
        "",
        f"class {class_name}(StrEnum):",
    ]
    for value in values:
        lines.append(f'    {_enum_member_name(value)} = "{value}"')
    lines.extend(
        [
            "",
            "",
            f"{values_constant_name}: tuple[str, ...] = tuple(value.value for value in {class_name})",
            "",
            "",
            f"def {coerce_name}(",
            f"    value: {class_name} | str | None,",
            "    *,",
            f'    field_name: str = "{field_name}",',
            f") -> {class_name} | None:",
            "    if value is None:",
            "        return None",
            f"    if isinstance(value, {class_name}):",
            "        return value",
            "    normalized = str(value).strip()",
            "    if not normalized:",
            "        return None",
            "    try:",
            f"        return {class_name}(normalized)",
            "    except ValueError as exc:",
            "        raise ValueError(",
            f"            f\"{{field_name}} must be one of: {{', '.join({values_constant_name})}}\"",
            "        ) from exc",
            "",
            "",
            f'__all__ = ["{values_constant_name}", "{class_name}", "{coerce_name}"]',
            "",
        ]
    )
    return "\n".join(lines)


def sync_smr_layered_enums(
    *,
    destination_dir: Path | None = None,
) -> list[Path]:
    target_dir = destination_dir or (Path(__file__).resolve().parent / "models")
    target_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for filename, class_name, field_name, values in _STATIC_ENUM_SPECS:
        target = target_dir / filename
        target.write_text(
            _render_static_enum_module(
                class_name=class_name,
                field_name=field_name,
                values=values,
                docstring=f"Public {field_name.replace('_', '-')} enum",
            ),
            encoding="utf-8",
        )
        generated.append(target)
    return generated


def sync_smr_agent_models(
    *,
    source_manifest: Path | None = None,
    destination_file: Path | None = None,
) -> Path:
    """Generate the Managed Research model enum from the backend-supported catalog."""

    source = source_manifest or _default_backend_supported_models_path()
    destination = destination_file or (
        Path(__file__).resolve().parent / "models" / "smr_agent_models.py"
    )
    if source.suffix == ".py":
        module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        models = None
        for node in module.body:
            if isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id == "SUPPORTED_MODEL_ENTRIES"
                    and node.value is not None
                ):
                    models = ast.literal_eval(node.value)
                    break
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SUPPORTED_MODEL_ENTRIES"
                for target in node.targets
            ):
                models = ast.literal_eval(node.value)
                break
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        models = raw.get("models")
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("Managed Research supported-model source must contain non-empty models")

    model_ids = [str(item.get("id") or "").strip() for item in models if isinstance(item, dict)]
    if not model_ids or any(not model_id for model_id in model_ids):
        raise ValueError("Managed Research supported-model entries require non-empty ids")
    model_ids.extend(
        model_id for model_id in _SMR_AGENT_MODEL_COMPATIBILITY_VALUES if model_id not in model_ids
    )

    lines = [
        '"""Generated Managed Research agent model enum.',
        "",
        "Source of truth: backend/packages/smr/config/supported_models_catalog.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from enum import StrEnum",
        "",
        "",
        "class SmrAgentModel(StrEnum):",
    ]
    for model_id in model_ids:
        lines.append(f'    {_enum_member_name(model_id)} = "{model_id}"')
    lines.extend(
        [
            "",
            "",
            "SMR_AGENT_MODEL_VALUES: tuple[str, ...] = tuple(model.value for model in SmrAgentModel)",
            "",
            "",
            "def coerce_smr_agent_model(",
            "    value: SmrAgentModel | str | None,",
            "    *,",
            '    field_name: str = "agent_model",',
            ") -> SmrAgentModel | None:",
            "    if value is None:",
            "        return None",
            "    if isinstance(value, SmrAgentModel):",
            "        return value",
            "    normalized = str(value).strip()",
            "    if not normalized:",
            "        return None",
            "    try:",
            "        return SmrAgentModel(normalized)",
            "    except ValueError as exc:",
            "        raise ValueError(",
            "            f\"{field_name} must be one of: {', '.join(SMR_AGENT_MODEL_VALUES)}. \"",
            '            "Backend preflight remains authoritative for model availability."',
            "        ) from exc",
            "",
            "",
            '__all__ = ["SMR_AGENT_MODEL_VALUES", "SmrAgentModel", "coerce_smr_agent_model"]',
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def sync_smr_actor_model_policy(
    *,
    source_manifest: Path | None = None,
    destination_file: Path | None = None,
) -> Path:
    """Generate actor policy constants from backend code-first actor role gates.

    Default source: import `smr.config.actor_model_policy` from a sibling backend
    checkout (gates live in actor_role_gates.py). Optional `source_manifest` may
    point at an exported JSON for offline sync; the historical
    `backend/config/smr_actor_model_policy.json` path is no longer authoritative.
    """

    destination = destination_file or (
        Path(__file__).resolve().parent / "models" / "smr_actor_policy_data.py"
    )
    raw = _load_actor_policy_manifest(source_manifest)
    policies = raw.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError(
            "Managed Research actor model policy manifest must contain a non-empty policies list"
        )

    policy_entries = _normalize_actor_policy_entries(policies)
    shared_raw = raw.get("shared_top_level_agent_models")
    if isinstance(shared_raw, list) and shared_raw:
        shared_top_level = tuple(
            dict.fromkeys(
                str(model_id).strip() for model_id in shared_raw if str(model_id).strip()
            )
        )
    else:
        shared_top_level = _shared_top_level_model_ids(policies)

    source_label = str(raw.get("source") or "").strip() or (
        str(source_manifest)
        if source_manifest is not None
        else "backend/packages/smr/config/actor_configurations/actor_role_gates.py"
    )

    lines = [
        '"""Generated Managed Research actor model policy constants.',
        "",
        f"Source of truth: {source_label}",
        "",
        "Regenerate: python -m synth_ai.managed_research.schema_sync",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "",
        "SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES: tuple[str, ...] = (",
    ]
    for model_id in shared_top_level:
        lines.append(f'    "{model_id}",')
    lines.extend(
        [
            ")",
            "",
            "",
            "SMR_ACTOR_MODEL_POLICY: tuple[dict[str, Any], ...] = (",
        ]
    )
    for entry in policy_entries:
        lines.append("    {")
        lines.append(f'        "actor_type": "{entry["actor_type"]}",')
        lines.append(f'        "actor_subtype": "{entry["actor_subtype"]}",')
        lines.append('        "permitted_models": [')
        for model_id in entry["permitted_models"]:
            lines.append(f'            "{model_id}",')
        lines.append("        ],")
        lines.append("    },")
    lines.extend(
        [
            ")",
            "",
            "",
            '__all__ = ["SMR_ACTOR_MODEL_POLICY", "SMR_SHARED_TOP_LEVEL_AGENT_MODEL_VALUES"]',
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def sync_smr_public_models_snapshot(
    *,
    source_manifest: Path | None = None,
    destination_file: Path | None = None,
) -> Path:
    """Sync the vendored public-model snapshot from the backend public-model export."""

    source = source_manifest or _default_backend_public_models_path()
    destination = destination_file or (
        Path(__file__).resolve().parent / "schemas" / "public_models.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Managed Research public model export must contain models")
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Managed Research public model export entries must be objects")
        if not str(item.get("id") or "").strip():
            raise ValueError("Managed Research public model export entries require non-empty ids")
    destination.write_text(
        json.dumps(raw, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """CLI entrypoint: sync tracked generated artifacts."""
    copied = sync_public_schemas()
    static_enums = sync_smr_layered_enums()
    generated = sync_smr_agent_models()
    actor_policy = sync_smr_actor_model_policy()
    public_models = sync_smr_public_models_snapshot()
    for path in copied:
        print(path)
    for path in static_enums:
        print(path)
    print(generated)
    print(actor_policy)
    print(public_models)


__all__ = [
    "main",
    "sync_public_schemas",
    "sync_smr_actor_model_policy",
    "sync_smr_agent_models",
    "sync_smr_layered_enums",
    "sync_smr_public_models_snapshot",
]
