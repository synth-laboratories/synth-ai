"""CLI subcommands for Synth AI.

This package hosts modular commands (watch, traces, recent, calc, status)
and exposes a top-level Click group named `cli` compatible with the
pyproject entry point `synth_ai.cli:cli`.
"""

from __future__ import annotations

# Load environment variables from a local .env if present (repo root)
try:
    from dotenv import find_dotenv, load_dotenv

    # Source .env early so CLI subcommands inherit config; do not override shell
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # dotenv is optional at runtime; proceed if unavailable
    pass


from .root import cli  # new canonical CLI entrypoint

# Register subcommands from this package onto the group
try:
    from . import watch as _watch

    _watch.register(cli)
except Exception:
    pass
try:
    from . import balance as _balance

    _balance.register(cli)
except Exception:
    pass
try:
    from . import man as _man

    _man.register(cli)
except Exception:
    pass
try:
    from . import traces as _traces

    _traces.register(cli)
except Exception:
    pass
try:
    from . import recent as _recent

    _recent.register(cli)
except Exception:
    pass
try:
    from . import calc as _calc

    _calc.register(cli)
except Exception:
    pass
try:
    from . import status as _status

    _status.register(cli)
except Exception:
    pass
try:
    from . import demo as _demo

    _demo.register(cli)
except Exception:
    pass
try:
    from . import rl_demo as _rl_demo

    _rl_demo.register(cli)
except Exception:
    pass
