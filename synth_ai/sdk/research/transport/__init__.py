"""Research HTTP transport.

Import from the owning submodule; this package re-exports nothing. Every
consumer already deep-imports, and the aggregator was the only thing keeping a
second ``RetryPolicy`` alive alongside the real one in ``core/http/retry.py``.
"""
