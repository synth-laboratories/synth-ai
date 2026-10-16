"""Extract Sublinear owner-origin HTTP routes from axum `.route(...)` registrations.

Usage: python extract_sublinear_routes.py <sublinear_checkout> <git_ref> <out_json>

Reads `crates/sublinear-server/src/**/*.rs` at <git_ref> (never the working tree) so the
fixture is pinned to an exact producer revision. Output rows mirror backend_all_routes.json:
{"methods": [...], "path": "...", "source": "<file>"} plus a header with the resolved commit.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SERVER_SOURCE = "crates/sublinear-server/src"
ROUTE_RE = re.compile(r"\.route\(\s*\"(/[^\"]+)\"\s*,")
METHOD_RE = re.compile(r"(?:^|[\s(.])(get|post|put|patch|delete)\(")
SEGMENT_END_RE = re.compile(r"\.route\(|\.merge\(|\.nest\(|\.with_state\(|\.layer\(Extension")


@dataclass(frozen=True)
class OwnerRoute:
    methods: tuple[str, ...]
    path: str
    source: str


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True
    ).stdout


def routes_in(text: str, source: str) -> list[OwnerRoute]:
    found = []
    for match in ROUTE_RE.finditer(text):
        tail = text[match.end() :]
        end = SEGMENT_END_RE.search(tail)
        segment = tail[: end.start()] if end else tail
        methods = tuple(sorted({m.upper() for m in METHOD_RE.findall(segment)}))
        if not methods:
            raise ValueError(f"route without a method handler: {match.group(1)} in {source}")
        found.append(OwnerRoute(methods=methods, path=match.group(1), source=source))
    return found


def main(checkout: Path, ref: str, out: Path) -> None:
    commit = git(checkout, "rev-parse", ref).strip()
    files = [
        name
        for name in git(checkout, "ls-tree", "-r", "--name-only", commit, SERVER_SOURCE).split()
        if name.endswith(".rs")
    ]
    routes = sorted(
        (
            route
            for name in files
            for route in routes_in(git(checkout, "show", f"{commit}:{name}"), name)
        ),
        key=lambda route: (route.path, route.methods),
    )
    out.write_text(
        json.dumps(
            {
                "producer": "sublinear",
                "source_commit": commit,
                "source_tree": SERVER_SOURCE,
                "routes": [
                    {"methods": list(r.methods), "path": r.path, "source": r.source} for r in routes
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(len(routes), "routes at", commit)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
