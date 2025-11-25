from __future__ import annotations

"""Minimal helpers for the math task app.

This module provides a local fallback for install_problem_bank_into_shared so
the modal task app can import it without requiring an external math_rl package.
"""


def install_problem_bank_into_shared() -> None:
    """No-op placeholder for installing the Hendrycks MATH problem bank.

    In production deployments, this can download or unpack the problem bank
    into a shared directory. For the demo scaffold, it is a no-op.
    """
    return None
