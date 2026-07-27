"""`__all__` is the contract, and the package ships nothing it cannot reach.

Two failure modes this catches.  A name leaks into `synth_ai` that nobody meant to publish —
an import alias, a helper, a re-export — and a customer starts depending on it.  Or a module
stays in the wheel after the last thing that reached it is gone, and the SDK carries dead
weight that still has to be read, reviewed, and kept compiling.

Reachability here is the lazy-aware static walk in `import_graph`, not `sys.modules` after an
import: `ResearchClient.account` pulls in `core/research/advanced.py` on first access, and a
runtime walk would call that module — and the eight behind it — an orphan.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from import_graph import (
    CONSOLE_ENTRY_POINTS,
    DUNDER_MAIN_ROOTS,
    PACKAGE,
    PUBLIC_ROOTS,
    REPO_ROOT,
    reachable_from,
    relative_path,
    shipped_modules,
)

# Orphans by design: nothing in the SDK's own import graph reaches these, but code outside
# this repo imports them by path.  Each needs a named consumer to stay on the list.
DECLARED_ORPHANS = {
    # evals/reportbench/synth_client.py, evals/scripts/smr_sdk_harness/readme_smoke.py
    "synth_ai.config",
    # evals/scripts/run_readme_smoke_via_managed_research.py
    "synth_ai.core.research.enums",
    # backend/services/container_pools/containers/templates/engine_bench/container.py
    "synth_ai.sdk.container",
    "synth_ai.sdk.container.auth",
}

# `from __future__ import annotations` binds this name in every module that uses it.
_FUTURE_FEATURE = "annotations"

_NAMESPACE_PROBE = """
import json, types
import synth_ai

public = {
    name: type(getattr(synth_ai, name)).__name__
    for name in vars(synth_ai)
    if not name.startswith("_")
}
print(json.dumps({
    "public": public,
    "declared": list(synth_ai.__all__),
    "deprecated": list(synth_ai._DEPRECATED_ALIASES),
    "exports": {k: v[0] for k, v in synth_ai._EXPORTS.items()},
}))
"""


def _namespace() -> dict[str, object]:
    """What a bare `import synth_ai` exposes, measured in a clean interpreter."""
    completed = subprocess.run(
        [sys.executable, "-c", _NAMESPACE_PROBE],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"namespace probe failed:\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_import_synth_ai_exposes_exactly_all() -> None:
    """No name reaches a customer without being declared.

    Submodules are exempt: `import synth_ai.cli` binds `synth_ai.cli` whether or not anyone
    asked for it, which is Python, not a curation decision.
    """
    namespace = _namespace()
    leaked = {
        name
        for name, kind in namespace["public"].items()
        if kind != "module" and name != _FUTURE_FEATURE
    } - set(namespace["declared"])
    assert not leaked, (
        f"names reachable from `import synth_ai` but absent from __all__: {sorted(leaked)}. "
        "Add them to __all__ if they are public, or give them a leading underscore."
    )


def test_all_is_resolvable_and_has_no_duplicates() -> None:
    namespace = _namespace()
    declared = namespace["declared"]
    assert sorted(declared) == sorted(set(declared)), "__all__ contains duplicates"
    missing = set(declared) - set(namespace["exports"])
    assert not missing, f"__all__ names nothing can resolve: {sorted(missing)}"


def test_deprecated_aliases_are_undeclared_and_warn() -> None:
    """A rename gets a window, not a silent removal — and never a place in the contract."""
    namespace = _namespace()
    overlap = set(namespace["deprecated"]) & set(namespace["declared"])
    assert not overlap, f"deprecated aliases must not be in __all__: {sorted(overlap)}"

    import synth_ai

    for alias in namespace["deprecated"]:
        with pytest.warns(DeprecationWarning, match=alias):
            getattr(synth_ai, alias)


def test_public_names_resolve_into_public_modules() -> None:
    """A customer's object should not report a private module as its home.

    Private helpers *behind* a public module are ordinary Python; a public name whose own
    defining module is private is a boundary that leaked.
    """
    namespace = _namespace()
    private = {
        name: module
        for name, module in namespace["exports"].items()
        if any(part.startswith("_") for part in module.split("."))
    }
    assert not private, f"public names defined in internal modules: {private}"


def test_shipped_modules_are_reachable() -> None:
    orphans = shipped_modules() - reachable_from(PUBLIC_ROOTS) - DECLARED_ORPHANS
    assert not orphans, (
        "modules ship but nothing reaches them:\n  "
        + "\n  ".join(sorted(relative_path(module) for module in orphans))
        + "\n\nDelete them, or add them to DECLARED_ORPHANS with the consumer that imports "
        "them by path."
    )


def test_declared_orphans_are_still_orphans() -> None:
    """Ratchet: an orphan that something now reaches stops needing the exemption."""
    reached = DECLARED_ORPHANS & reachable_from(PUBLIC_ROOTS)
    assert not reached, (
        f"these are reachable now — drop them from DECLARED_ORPHANS: {sorted(reached)}"
    )


@pytest.mark.parametrize("root", (PACKAGE, *CONSOLE_ENTRY_POINTS, *DUNDER_MAIN_ROOTS))
def test_declared_roots_exist(root: str) -> None:
    """Roots are the contract's other half; a stale one silently widens the orphan set."""
    assert root in shipped_modules(), f"declared root {root} is not a shipped module"
