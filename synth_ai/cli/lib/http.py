from __future__ import annotations

from typing import Any

__all__ = ["http_request"]


def http_request(
    method: str, url: str, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any] | str]:
    import json as _json
    import os
    import ssl
    import urllib.error
    import urllib.request

    data = None
    if body is not None:
        data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        ctx = ssl._create_unverified_context()
        if os.getenv("SYNTH_SSL_VERIFY", "0") == "1":
            ctx = None
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            code = getattr(resp, "status", 200)
            txt = resp.read().decode("utf-8", errors="ignore")
            try:
                return int(code), _json.loads(txt)
            except Exception:
                return int(code), txt
    except urllib.error.HTTPError as exc:  # Capture 4xx/5xx bodies
        txt = exc.read().decode("utf-8", errors="ignore")
        try:
            return int(exc.code or 0), _json.loads(txt)
        except Exception:
            return int(exc.code or 0), txt
    except Exception as exc:
        return 0, str(exc)

