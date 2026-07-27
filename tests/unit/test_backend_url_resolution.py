"""An unconfigured SDK reaches production, and a local backend has to be asked for.

`SynthClient()` with no `base_url` is the pip-install path: someone read the README, set
`SYNTH_API_KEY`, and called it.  Resolving that to `http://localhost:8000` fails with a
connection error against a machine that runs no backend, and the SDK is the only thing that
could have known better.

The inverse matters just as much.  A developer pointing at a local or dev backend must say
so — via `SYNTH_BACKEND_URL`, `SYNTH_BACKEND_URL_OVERRIDE`, or `ENVIRONMENT` — because the
alternative is a default that silently decides which deployment gets the traffic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from synth_ai.core.utils.urls import (
    REQUIRE_EXPLICIT_BACKEND_ENV,
    _resolve_backend_url,
    default_backend_base,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

PROD = "https://api.usesynth.ai"
DEV = "https://api-dev.usesynth.ai"
LOCAL = "http://localhost:8000"

# Every variable that can move the answer.  Cleared before each case so a developer's shell
# cannot make this suite agree with itself for the wrong reason.
BACKEND_ENV_VARS = (
    "SYNTH_BACKEND_URL_OVERRIDE",
    "SYNTH_BACKEND_URL",
    "SYNTH_API_URL",
    "BACKEND_URL",
    "PROD_SYNTH_BACKEND_URL",
    "PROD_BACKEND_URL",
    "DEV_SYNTH_BACKEND_URL",
    "DEV_BACKEND_URL",
    "LOCAL_BACKEND_URL",
    "ENVIRONMENT",
    "APP_ENVIRONMENT",
    "ENV",
)

CASES = (
    ("nothing configured resolves to prod", {}, PROD),
    ("explicit url wins", {"SYNTH_BACKEND_URL": "https://example.test"}, "https://example.test"),
    ("override names local", {"SYNTH_BACKEND_URL_OVERRIDE": "local"}, LOCAL),
    ("override names dev", {"SYNTH_BACKEND_URL_OVERRIDE": "dev"}, DEV),
    ("override names prod", {"SYNTH_BACKEND_URL_OVERRIDE": "prod"}, PROD),
    ("ENVIRONMENT=dev opts into the dev chain", {"ENVIRONMENT": "dev"}, LOCAL),
    ("ENVIRONMENT=prod stays prod", {"ENVIRONMENT": "prod"}, PROD),
    ("DEV_BACKEND_URL implies dev", {"DEV_BACKEND_URL": "https://dev.test"}, "https://dev.test"),
    ("PROD_BACKEND_URL implies prod", {"PROD_BACKEND_URL": "https://p.test"}, "https://p.test"),
)


@pytest.mark.parametrize(
    ("env", "expected"),
    [pytest.param(env, expected, id=name) for name, env, expected in CASES],
)
def test_backend_url_resolution(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    for name in BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _resolve_backend_url() == expected


def test_unconfigured_client_targets_prod() -> None:
    """The import-time constant, measured with the environment scrubbed.

    `BACKEND_URL_BASE` is fixed when the module loads, so mutating the environment after
    import cannot move it — only a fresh interpreter answers this honestly.
    """
    probe = (
        "import json; from synth_ai import SynthClient;"
        "from synth_ai.core.utils.urls import BACKEND_URL_BASE;"
        "print(json.dumps({'constant': BACKEND_URL_BASE,"
        " 'client': SynthClient(api_key='probe').base_url}))"
    )
    environment = {k: v for k, v in os.environ.items() if k not in BACKEND_ENV_VARS}
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    resolved = json.loads(completed.stdout.strip().splitlines()[-1])
    assert resolved == {"constant": PROD, "client": PROD}


# Every door onto the package default. `default_backend_base` is the only one that computes
# it; the rest must delegate, or SYNTH_REQUIRE_EXPLICIT_BACKEND covers some callers and not
# others — which is worse than no guard, because it reads as protection.
DEFAULT_BACKEND_DOORS = (
    (
        "urls.default_backend_base",
        "from synth_ai.core.utils.urls import default_backend_base; default_backend_base()",
    ),
    (
        "urls.resolve_synth_backend_url",
        "from synth_ai.core.utils.urls import resolve_synth_backend_url;"
        " resolve_synth_backend_url()",
    ),
    (
        "config.default_backend_base",
        "from synth_ai.config import default_backend_base; default_backend_base()",
    ),
    (
        "env.get_backend_url",
        "from synth_ai.core.utils.env import get_backend_url; get_backend_url()",
    ),
    (
        "sdk.base.resolve_backend_base",
        "from synth_ai.sdk.base import resolve_backend_base; resolve_backend_base(None)",
    ),
    (
        "session.config.resolve_backend_base",
        "from synth_ai.core.research.session.config import resolve_backend_base;"
        " resolve_backend_base(None)",
    ),
    (
        "client.SynthClient",
        "from synth_ai.client import SynthClient; SynthClient(api_key='k')",
    ),
    (
        "research.client.Client",
        "from synth_ai.core.research.client import Client; Client(api_key='k')",
    ),
    (
        "cli._resolve_backend_url",
        "from synth_ai.cli.research import _resolve_backend_url; _resolve_backend_url(None)",
    ),
    (
        "research._internal.urls",
        "from synth_ai.core.research._internal.urls import default_backend_base;"
        " default_backend_base()",
    ),
)


@pytest.mark.parametrize(
    "probe", [pytest.param(source, id=name) for name, source in DEFAULT_BACKEND_DOORS]
)
def test_require_explicit_backend_covers_every_door(probe: str) -> None:
    """With the flag set, no unnamed backend resolves — through any entry point."""
    environment = {k: v for k, v in os.environ.items() if k not in BACKEND_ENV_VARS}
    environment[REQUIRE_EXPLICIT_BACKEND_ENV] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    assert completed.returncode != 0, "resolved a backend nobody named"
    assert "synth_backend_base_unspecified" in completed.stderr, completed.stderr


def test_named_backend_still_resolves_under_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard objects to silence, not to strictness — naming a backend still works."""
    monkeypatch.setenv(REQUIRE_EXPLICIT_BACKEND_ENV, "1")
    from synth_ai.core.research.session.config import resolve_backend_base

    assert resolve_backend_base("https://api-dev.usesynth.ai") == "https://api-dev.usesynth.ai"

    monkeypatch.setenv("SYNTH_BACKEND_URL", "https://named.test")
    assert resolve_backend_base(None) == "https://named.test"


def test_guard_is_off_by_default() -> None:
    assert default_backend_base() == PROD
