"""Schema and enum sync helpers for the synth-ai Managed Research surface."""

from __future__ import annotations

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
        ("openai", "openrouter", "tinker"),
    ),
    (
        "smr_inference_providers.py",
        "SmrInferenceProvider",
        "inference_provider",
        ("openai", "google", "groq"),
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
        ("runpod", "modal"),
    ),
    (
        "smr_resource_kinds.py",
        "SmrResourceKind",
        "resource_kind",
        ("pod", "sandbox", "app"),
    ),
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
    """Generate the public Managed Research model enum from the backend agent-model catalog."""

    source = source_manifest or _default_backend_manifest_path()
    destination = destination_file or (
        Path(__file__).resolve().parent / "models" / "smr_agent_models.py"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(
            "Managed Research public model manifest must contain a non-empty models list"
        )

    model_ids = [
        str(item.get("id") or "").strip()
        for item in models
        if isinstance(item, dict) and bool(item.get("public", True))
    ]
    if not model_ids or any(not model_id for model_id in model_ids):
        raise ValueError("Managed Research public model manifest entries require non-empty ids")

    lines = [
        '"""Generated public Managed Research agent model enum.',
        "",
        "Source of truth: backend/config/smr_supported_models.json",
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


def sync_smr_public_models_snapshot(
    *,
    source_manifest: Path | None = None,
    destination_file: Path | None = None,
) -> Path:
    """Generate the vendored public-model snapshot from the backend catalog."""

    source = source_manifest or _default_backend_manifest_path()
    destination = destination_file or (
        Path(__file__).resolve().parent / "schemas" / "public_models.json"
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Managed Research supported model manifest must contain models")
    public_models = []
    for item in models:
        if not isinstance(item, dict) or not bool(item.get("public", True)):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            raise ValueError("Managed Research public model manifest entries require non-empty ids")
        public_models.append(
            {
                "id": model_id,
                "display_group": str(item.get("display_group") or "additional_first_launch"),
                "launch_lane": str(item.get("auth_route") or "").strip(),
                "public": True,
            }
        )
    destination.write_text(
        json.dumps({"version": 1, "models": public_models}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """CLI entrypoint: sync tracked generated artifacts."""
    copied = sync_public_schemas()
    static_enums = sync_smr_layered_enums()
    generated = sync_smr_agent_models()
    public_models = sync_smr_public_models_snapshot()
    for path in copied:
        print(path)
    for path in static_enums:
        print(path)
    print(generated)
    print(public_models)


__all__ = [
    "main",
    "sync_public_schemas",
    "sync_smr_agent_models",
    "sync_smr_layered_enums",
    "sync_smr_public_models_snapshot",
]
