# Local SDK docs preview

This directory holds the auto-generated SDK reference pages. The published site
is Blume, in the sibling [`docs`](https://github.com/synth-laboratories/docs)
repo; preview there with `npm run dev`.

Generated output (`reference/sdk/`, `docs.json`, `overview.mdx`) is produced by
`scripts/generate_sdk_docs.py` from Python docstrings. See
`specifications/sdk/docstrings.md` for the full spec.

## Prerequisites

```bash
uv sync --group dev
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

Preview in the docs repo, which owns the Blume site:

```bash
make docs-gen
uv run python scripts/generate_sdk_docs.py --sync-docs-repo
cd ../docs/docs && npm run dev
```

Do not run `make docs-gen` or `--sync-docs-repo` while the preview server is
running: regeneration deletes and recreates the MDX tree under `reference/sdk/`,
so the file watcher thrashes and navigation briefly points at missing pages.

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
