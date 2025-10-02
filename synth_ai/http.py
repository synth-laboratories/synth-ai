"""
Compatibility shim to avoid shadowing Python's stdlib `http` module.
This re-exports the actual client implementation from http_client.py and
supports both package and script execution contexts.
"""

try:
    from synth_ai.http_client import *  # type: ignore F401,F403
except Exception:
    try:
        from .http_client import *  # type: ignore F401,F403
    except Exception:
        import importlib.util as _ilu
        import sys as _sys
        from pathlib import Path as _Path

        _here = _Path(__file__).resolve()
        _client_path = _here.parent / "http_client.py"
        _spec = _ilu.spec_from_file_location("http_client", str(_client_path))
        if not _spec or not _spec.loader:
            raise ImportError("Could not load http_client module")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _sys.modules["synth_ai.http_client"] = _mod
        for _name in ("HTTPError", "AsyncHttpClient", "sleep"):
            globals()[_name] = getattr(_mod, _name)
