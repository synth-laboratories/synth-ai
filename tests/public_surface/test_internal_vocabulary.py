"""Backend orchestration vocabulary must not be reachable from the public API.

`synth-ai` is the customer-facing SDK.  A customer never provisions a Daytona snapshot,
claims a worker-pool slot, or resolves an internal Railway URL — so those words appearing in
a module a customer can import means backend internals leaked across the boundary.

This is a ratchet, not a clean bill of health.  Every offender that exists today is listed in
`internal_vocabulary_allowlist.txt`; the list may shrink and never grow.  Removing the last
entry for a term is what finishing the job looks like.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from import_graph import PUBLIC_ROOTS, REPO_ROOT, reachable_from, relative_path

# Concepts that belong to backend orchestration and to nothing a customer constructs.
# `smr_runtime_kinds` is excluded: it is the launch enum a customer picks from
# (`codex`, `react`, `container_http`, …), not the internal worker runtime.
INTERNAL_VOCABULARY: dict[str, re.Pattern[str]] = {
    term: re.compile(pattern)
    for term, pattern in (
        ("daytona", r"daytona"),
        ("worker_pool", r"worker_pool"),
        ("worker_claim", r"worker_claim"),
        ("smr_runtime", r"smr_runtime(?!_kind)"),
        ("slot_id", r"slot_id"),
        ("railway", r"railway"),
    )
}

ALLOWLIST_PATH = Path(__file__).parent / "internal_vocabulary_allowlist.json"


def _allowlist() -> frozenset[str]:
    return frozenset(json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))["offenders"])


def _offenders() -> frozenset[str]:
    """`path::term` for every internal term appearing in a publicly reachable module."""
    found = set()
    for module in reachable_from(PUBLIC_ROOTS):
        path = relative_path(module)
        text = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore").lower()
        found.update(
            f"{path}::{term}"
            for term, pattern in INTERNAL_VOCABULARY.items()
            if pattern.search(text)
        )
    return frozenset(found)


def test_no_new_internal_vocabulary() -> None:
    new = sorted(_offenders() - _allowlist())
    assert not new, (
        "backend-internal vocabulary reached the public API in modules that were clean:\n  "
        + "\n  ".join(new)
        + f"\n\nIf a customer would never legitimately construct it, it does not belong in "
        f"synth-ai. Move it out rather than adding it to {ALLOWLIST_PATH.name}."
    )


def test_allowlist_does_not_go_stale() -> None:
    """The ratchet only tightens if fixed entries are struck from the list."""
    fixed = sorted(_allowlist() - _offenders())
    assert not fixed, (
        "these allowlist entries no longer describe a real offender — delete them so the "
        "ratchet holds:\n  " + "\n  ".join(fixed)
    )
