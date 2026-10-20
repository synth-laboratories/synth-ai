# Exact repository reads

Repository and provenance operations use `core.v1`. A Ref identifies an immutable
revision and its canonical manifest digest; authorization is checked on every
request. Public repository IDs are opaque Forge identifiers. The backend derives
them as `repo_` followed by the SHA-256 of the internal immutable repository ID.

```python
from pathlib import Path
from synth_ai import SynthClient
from synth_ai.sdk.core import RepositoryArchiveOptions

with SynthClient() as synth:
    run = synth.swarms.get("run-id")
    produced = run.produced()
    repo = synth.repos.get("project-id")
    commit = repo.commit("0123456789012345678901234567890123456789")
    receipt = repo.archive(
        commit.ref, Path("checkout"), options=RepositoryArchiveOptions(timeout_seconds=300)
    )
```

`AsyncSynthClient` has the same operations, awaited. `run.ref` comes from the
frozen launch request reported by the backend. The SDK never infers it from a
mutable configuration. Handles constructed manually or recovered without a
snapshot need `status()` before provenance reads. An older backend omitting the
Ref remains readable; provenance then raises `run_ref_unavailable`.

`links.list(project_id, subject=ref, relation="produced")` selects exactly one
subject or object. The SDK binds both the four-part selector and its digest,
and checks each returned endpoint. Pages are limited to 200 links. `produced()`
and `consumed()` return these links with the run as subject. Follow `next_cursor`
when a page is incomplete. Provenance history alone does not select a final
worker branch tip.

Repository handles expose `refs(prefix=..., limit=50)`, `commit(full_sha)`,
`diff(base, head)`, `archive(commit, destination)` and `clone_url(ttl=300)`.
Diff inputs accept exact Refs or full lowercase commit SHAs; returned Refs are
checked against those selections. Diff patches are bounded to 8 MiB and 2,000
files and report truncation. Ref pages are bounded to 50 entries.
Repository ref cursors permit8,192 characters to carry a4,096byte ref-name
position and its scoped envelope. Ref pagination observes current branch names;
each returned commit Ref remains exact if a branch subsequently advances.

`clone_url()` is a keyed mutation returning a descriptor: `clone_url`,
`auth_header`, `expires_at` and `scope="project"`. It mints a read-only,
project-scoped capability for at most 600 seconds. Supply the bearer header to
Git via an appropriate private credential mechanism. URLs contain no credentials;
the header is hidden from descriptor repr. A lost mutation response is recovered
through its mutation identity before minting another capability.

Archive downloads perform one streaming request with no retries or redirects.
The overall deadline includes an optional commit-descriptor lookup, transfer,
extraction and verification. Defaults and hard maxima are 900 seconds, 512 MiB
and 100,000 entries. Implicit directories have a separate configurable bound
(100,000 by default). The uncompressed `synth.git-raw-tar.v1` protocol preserves
raw files, executable modes, symlink targets and gitlink identities, including
files that Git export attributes would hide or rewrite.

The SDK first verifies complete tar bytes, count and SHA-256 in private scratch
beside the destination, then safely extracts and hashes the actual tree against
the exact Ref. Traversal, escaping symlinks, duplicate/colliding names, hardlinks,
device nodes, sparse encodings and oversized extension metadata are refused.
Verified directories are published with a native exclusive rename. The parent
directory must exist; an existing destination, including an empty directory or
symlink, is never replaced. Cancellation and client close retain cleanup
ownership. Failed downloads leave no partial destination. Local directory
publication requires Darwin, Linux with `renameat2`, or Windows support.

The contract snapshots define required identity and delivery headers.
`CommitDigestMismatch` identifies a materialized tree mismatch; other malformed
archive receipts or framing raise typed contract errors. Error messages include
the operation and bounded request identity without exposing response bodies.
