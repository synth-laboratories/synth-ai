# Local SDK docs preview

This directory is a **local Mintlify site** for previewing auto-generated SDK
reference pages. Production docs live in the sibling [`docs`](https://github.com/synth-laboratories/docs) repo.

Generated output (`reference/sdk/`, `docs.json`, `overview.mdx`) is produced by
`scripts/generate_sdk_docs.py` from Python docstrings. See
`specifications/sdk/docstrings.md` for the full spec.

## Prerequisites

```bash
uv sync --group dev
brew install mintlify  # or: npm i -g mintlify
```

## Generate reference

From the repo root:

```bash
make docs-gen      # write docs/reference/sdk/ + docs/docs.json
make docs-check    # docs-gen + docstring gate (CI runs this)
```

Sync to the production docs repo:

```bash
uv run python scripts/generate_sdk_docs.py --sync-docs-repo
```

## Local preview

**Always run `make docs-gen` before starting the preview.** Do not run
`make docs-gen` (or `--sync-docs-repo`) while `mint dev` is running.

Regeneration deletes and recreates MDX files under `reference/sdk/`. If Mintlify
is watching that tree, the file watcher thrashes, nav briefly points at missing
pages, and the dev server can crash with:

```text
TypeError: controller[kState].transformAlgorithm is not a function
```

Workflow:

```bash
make docs-gen
make docs-dev          # cd docs && mint dev
# or explicitly:
cd docs && mint dev --port 3000
```

Open **http://localhost:3000/overview** (redirects from `/`).

If the preview keeps crashing, stop `mint dev`, run `make docs-gen` again, then
restart. Consider `mint update` if Mintlify warns about an available update.

## What not to edit by hand

| Path | Notes |
| --- | --- |
| `reference/sdk/**` | Auto-generated MDX — edit Python docstrings instead |
| `docs.json` | Written by `generate_sdk_docs.py` |
| `overview.mdx` | Written by `generate_sdk_docs.py` |

Vendored third-party references under `references/` are ignored (see `.gitignore`).

## Research recipes

- [First bounded research run](recipes/first-bounded-research-run.md)
- [Async handoff](recipes/async-handoff.md)

These hand-written recipes are not generated or removed by `make docs-gen`.
