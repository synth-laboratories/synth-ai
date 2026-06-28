#!/usr/bin/env python3
"""Generate Mintlify SDK reference under synth-ai/docs/ from docstrings.

See: specifications/sdk/docstrings.md
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sdk_docs_postprocess import postprocess_mdx_files  # noqa: E402

DOCS_DIR = ROOT / "docs"
OUTPUT_DIR = DOCS_DIR / "reference" / "sdk"
DOCS_JSON = DOCS_DIR / "docs.json"
MANIFEST_PATH = ROOT / "specifications" / "sdk" / "public_api_manifest.json"
REPO_URL = "https://github.com/synth-laboratories/synth-ai"

GENERATED_RENAMES: dict[str, str] = {
    "synth_ai-client.mdx": "synth-client.mdx",
    "synth_ai-sdk-containers.mdx": "containers-client.mdx",
    "synth_ai-sdk-tunnels.mdx": "tunnels-client.mdx",
    "synth_ai-sdk-pools.mdx": "pools-client.mdx",
}

INFRA_NAV_PAGES: list[str] = [
    "reference/sdk/index",
    "reference/sdk/synth-client",
    "reference/sdk/containers-client",
    "reference/sdk/tunnels-client",
    "reference/sdk/pools-client",
]


def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _run_mdxify(
    root_module: str,
    output_dir: Path,
    *,
    all_modules: bool,
    excludes: list[str],
) -> None:
    manifest = _load_manifest()
    exclude_patterns = list(manifest.get("exclude_patterns") or [])

    cmd = [sys.executable, "-m", "mdxify"]
    if all_modules:
        cmd.extend(["--all", "--root-module", root_module])
    else:
        cmd.append(root_module)

    cmd.extend(
        [
            "--output-dir",
            str(output_dir),
            "--no-update-nav",
            "--repo-url",
            REPO_URL,
            "--branch",
            "main",
            "--format",
            "mdx",
            "--docstring-style",
            str(manifest.get("docstring_style") or "google"),
        ]
    )
    for pattern in exclude_patterns:
        cmd.extend(["--exclude", pattern])
    for module in excludes:
        cmd.extend(["--exclude", module])

    subprocess.run(cmd, cwd=ROOT, check=True)


def _rename_generated_pages() -> None:
    for source_name, dest_name in GENERATED_RENAMES.items():
        source = OUTPUT_DIR / source_name
        dest = OUTPUT_DIR / dest_name
        if source.exists():
            if dest.exists():
                dest.unlink()
            source.rename(dest)


def _write_index_mdx() -> None:
    content = """---
title: SDK Reference
description: Auto-generated Python SDK reference for SynthClient and SynthClient().research.
sidebarTitle: SDK Reference
---

This reference is **auto-generated from Python docstrings** in
[`synth-ai`](https://github.com/synth-laboratories/synth-ai).

Regenerate locally:

```bash
make docs-gen
```

For narrative quickstarts, see [Managed Research SDK](https://docs.usesynth.ai/managed-research/sdk).

## Client entrypoint

| Surface | Description | Reference |
| --- | --- | --- |
| `SynthClient` | Front-door client (infra + research) | [SynthClient](/reference/sdk/synth-client) |

## Infrastructure

| Surface | Description | Reference |
| --- | --- | --- |
| Containers | Build and run environment containers | [ContainersClient](/reference/sdk/containers-client) |
| Tunnels | Expose local containers to Synth | [TunnelsClient](/reference/sdk/tunnels-client) |
| Pools | Container pools and rollouts | [ContainerPoolsClient](/reference/sdk/pools-client) |

## Managed Research (`SynthClient().research`)

Entrypoint: **`client.research`**

### Core

| Namespace | Description | Reference |
| --- | --- | --- |
| `research` | Root research client | [ResearchClient](/reference/sdk/research/synth_ai-research-client) |
| `research.factories` | Factory Tag sessions | [Factories](/reference/sdk/research/synth_ai-research-factories) |
| `research.limits` | Org limits | [Limits](/reference/sdk/research/synth_ai-research-limits) |
| `research.secrets` | Project secrets | [Secrets](/reference/sdk/research/synth_ai-research-secrets) |

### Projects

| Namespace | Description | Reference |
| --- | --- | --- |
| `research.projects` | Project CRUD | [Projects](/reference/sdk/research/synth_ai-research-projects) |
| `research.projects.*` | Setup, workspace, git, objectives | [Project namespaces](/reference/sdk/research/synth_ai-research-project_namespaces) |

### Runs

| Namespace | Description | Reference |
| --- | --- | --- |
| `research.runs` | Launch, wait, lifecycle | [Runs](/reference/sdk/research/synth_ai-research-runs) |
| `handle.*` | Usage, progress, queue, artifacts | [Run readouts](/reference/sdk/research/synth_ai-research-run_readouts) |

### Types

| Page | Reference |
| --- | --- |
| Models | [Models](/reference/sdk/research/synth_ai-research-models) |
| Enums | [Enums](/reference/sdk/research/synth_ai-research-enums) |
| Errors | [Errors](/reference/sdk/research/synth_ai-research-errors) |
| Hosted artifacts | [Hosted artifacts](/reference/sdk/research/synth_ai-research-hosted-artifacts) |
"""
    (OUTPUT_DIR / "index.mdx").write_text(content, encoding="utf-8")


def _write_overview_mdx() -> None:
    content = """---
title: Overview
description: Python SDK for Synth containers, tunnels, pools, and Managed Research.
sidebarTitle: Overview
---

# Synth AI SDK

Install:

```bash
pip install "synth-ai[research]"
```

Hero entrypoint:

```python
from synth_ai import SynthClient

client = SynthClient()
research = client.research
limits = research.limits.get()
```

## Documentation layers

| Layer | Where |
| --- | --- |
| **Guides** | [docs.usesynth.ai](https://docs.usesynth.ai/managed-research/sdk) — quickstarts and concepts |
| **Reference** | Auto-generated from docstrings — [SDK Reference](/reference/sdk/index) |

## Open Research hosted artifacts (alpha)

Workers with subtype `artifact_builder` publish HTML proof pages; operators promote public slugs for the Open Research index. See [Hosted artifacts](/reference/sdk/research/synth_ai-research-hosted-artifacts).

## Local preview

```bash
make docs-gen
make docs-dev
```

Open the URL printed by `mint dev` (usually http://localhost:3000/overview).
"""
    (DOCS_DIR / "overview.mdx").write_text(content, encoding="utf-8")


def _write_hosted_artifacts_mdx() -> None:
    content = """---
title: Hosted artifacts
sidebarTitle: Hosted artifacts
tag: "ALPHA"
---

# Open Research hosted artifacts

<Badge color="yellow" icon="triangle-exclamation">Alpha</Badge>

SMR **`artifact_builder`** workers publish HTML hosted artifacts during a run. Operators promote a public slug for the Open Research index at `/openresearch/artifacts/{slug}`.

## Worker subtypes (launch contract)

Set `actor_subtype` on a worker task in the kickoff contract:

| Subtype | Role |
| --- | --- |
| `artifact_builder` | Build HTML and call `publish_hosted_artifact` |
| `artifact_reviewer` | Review hosted artifact before public promote (orchestrator dispatch) |

Python enums (SDK):

```python
from synth_ai.managed_research.models.smr_actor_models import (
    SmrWorkerSubtype,
    SmrReviewerSubtype,
)

SmrWorkerSubtype.ARTIFACT_BUILDER  # "artifact_builder"
SmrReviewerSubtype.ARTIFACT_REVIEWER  # "artifact_reviewer"
```

Example kickoff task snippet:

```python
{
    "task_key": "build_hosted_artifact",
    "kind": "worker_task",
    "actor_subtype": "artifact_builder",
    "instructions": "Build index.html and call publish_hosted_artifact.",
}
```

## Operator HTTP surface (backend)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/smr/runs/{run_id}/hosted-artifact` | Receipt: hosted URL, WorkProduct id, lineage |
| `GET` | `/smr/hosted-artifacts/{id}/content` | Serve HTML |
| `POST` | `/smr/hosted-artifacts/{id}/publish-public` | Promote public slug |
| `POST` | `/smr/hosted-artifacts/{id}/assign-reviewer` | Dispatch `artifact_reviewer` |
| `GET` | `/api/open-research/v1/artifacts` | Public index JSON |
| `GET` | `/api/open-research/v1/artifacts/{slug}` | Public slug bundle |

## Local smoke

```bash
cd ~/Documents/GitHub/backend
./.venv/bin/python scripts/run_artifact_builder_smoke.py \\
  --base-url http://127.0.0.1:8001 \\
  --pool-id slot2
```

Requires slot2 backend-api + smr-runtime with `artifact_builder_minimal` profile.

## Stack operator

- `stack_get_run_artifact_status` → same fields as `GET /smr/runs/{run_id}/hosted-artifact`
- `stack_open_hosted_artifact` → open hosted or public shell URL in the system browser
"""
    (OUTPUT_DIR / "research" / "synth_ai-research-hosted-artifacts.mdx").write_text(
        content, encoding="utf-8"
    )


def _write_docs_json() -> None:
    manifest = _load_manifest()
    nav_groups = manifest.get("nav_groups") or []

    research_pages: list[str] = []
    for section in nav_groups:
        for page in section.get("pages", []):
            if isinstance(page, str):
                research_pages.append(page)

    config = {
        "$schema": "https://mintlify.com/docs.json",
        "name": "Synth AI SDK",
        "theme": "mint",
        "colors": {
            "primary": "#f97316",
            "light": "#ffffff",
            "dark": "#000000",
        },
        "appearance": {"default": "dark", "strict": True},
        "redirects": [
            {"source": "/", "destination": "/overview", "permanent": False},
        ],
        "navigation": {
            "groups": [
                {
                    "group": "Get started",
                    "pages": ["overview"],
                },
                {
                    "group": "SDK Reference",
                    "pages": INFRA_NAV_PAGES,
                },
                {
                    "group": "Research API",
                    "pages": research_pages,
                },
            ],
        },
        "footerSocials": {
            "github": REPO_URL,
        },
    }
    DOCS_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _sync_docs_repo() -> None:
    docs_repo = ROOT.parent / "docs"
    generator = docs_repo / "scripts" / "generate_sdk_reference.py"
    if not generator.is_file():
        print("Skipping docs repo sync (../docs not found)")
        return
    subprocess.run(
        [sys.executable, str(generator), str(ROOT)],
        cwd=docs_repo,
        check=True,
    )
    print(f"Synced production docs at {docs_repo / 'docs' / 'reference' / 'sdk'}")


def main() -> None:
    sync_prod = "--sync-docs-repo" in sys.argv
    manifest = _load_manifest()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("Generating SDK reference into docs/reference/sdk/ ...")
    for root in manifest.get("roots") or []:
        if not isinstance(root, dict):
            continue
        module = str(root["module"])
        output = OUTPUT_DIR
        rel_output = str(root.get("output") or "")
        if rel_output.endswith("research"):
            output = OUTPUT_DIR / "research"
        output.mkdir(parents=True, exist_ok=True)
        excludes = list(root.get("exclude_modules") or [])
        if module == "synth_ai.research":
            excludes.append("synth_ai.research.control")
        _run_mdxify(
            module,
            output,
            all_modules=bool(root.get("all_modules")),
            excludes=excludes,
        )

    _rename_generated_pages()
    modified = postprocess_mdx_files(OUTPUT_DIR)
    print(f"Post-processed {modified} MDX files")
    _write_index_mdx()
    _write_overview_mdx()
    _write_hosted_artifacts_mdx()
    _write_docs_json()

    file_count = len(list(OUTPUT_DIR.rglob("*.mdx")))
    print(f"Generated {file_count} reference pages under docs/reference/sdk/")

    if sync_prod:
        _sync_docs_repo()


if __name__ == "__main__":
    main()
