Exact references identify immutable content. Resolve a mutable alias before
constructing a Ref; references never select a moving “latest” revision.

| Content | Authority | Kind | Record | Revision | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| Artifact | `artifacts` | `artifact` | Artifact ID | Positive revision number | Artifact manifest |
| Commit | `git` | `commit` | Repository ID | Full commit SHA | Canonical commit manifest |
| Run | `swarms` | `run` | Run ID | `1` | Frozen launch request |
| Future record | `forge` | Record kind | Record ID | Exact revision | Record document |

`Producer` preserves the Forge producer stamp; absent references are omitted.
`Link` records append-only provenance using a closed relation vocabulary.

Commit integrity uses `synth.git-commit-manifest.v1`: entries sort by raw UTF-8
path bytes and contain the Git mode and SHA-256 of each blob. Symlinks hash
their target bytes; gitlinks hash `gitlink:<commit-sha>`. Commit identity is
separate from tree identity. The Git service owns the digest; SDK tree checks
verify downloaded content against it.
