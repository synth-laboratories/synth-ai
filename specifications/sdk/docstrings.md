# SDK docstrings and reference generation

Public Python SDK reference is generated from Google-style docstrings via
[mdxify](https://github.com/modal-labs/mdxify).

## Source of truth

| Artifact | Role |
| --- | --- |
| `specifications/sdk/public_api_manifest.json` | Modules to generate, nav groups, docstring gate list |
| `scripts/generate_sdk_docs.py` | Local Mintlify preview under `docs/reference/sdk/` |
| `scripts/check_sdk_docstrings.py` | CI gate: required modules must have docstrings |
| `scripts/sdk_docs_postprocess.py` | Shared MDX postprocess (examples, JSX escapes, titles) |

Production site generation lives in the sibling `docs` repo:
`scripts/generate_sdk_reference.py` imports postprocess from synth-ai.

## Commands

```bash
make docs-gen      # regenerate docs/reference/sdk/ + docs/docs.json
make docs-check    # docs-gen + docstring gate
make docs-dev      # mint dev (after docs-gen)
uv run python scripts/generate_sdk_docs.py --sync-docs-repo
```

## Docstring rules

- Google style (`Args:`, `Returns:`, `Raises:`, `Example:`).
- Every public class and method in manifest `docstring_coverage_required` files.
- Hero methods include a runnable `Example:` block.
- No `TODO` lines in generated MDX (gate scans output).
