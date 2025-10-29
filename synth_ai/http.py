"""
Backward-compatible HTTP client exports.

Historically, some modules imported ``synth_ai.http``. The canonical location
is ``synth_ai.http_client``; this module simply re-exports the same symbols so
legacy imports keep working.
"""


from synth_ai.http_client import AsyncHttpClient, HTTPError, sleep

__all__ = ["AsyncHttpClient", "HTTPError", "sleep"]
