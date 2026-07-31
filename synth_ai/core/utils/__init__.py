"""Shared utilities. Import from the submodule that owns the helper.

This package deliberately re-exports nothing. The aggregator it used to be had
no consumer -- every call site deep-imports -- and its only real effect was to
keep a tree of dead helpers reachable, and so alive.
"""
