"""Which products the public client, CLI and MCP catalog expose.

This is a code constant, not an environment switch. Hidden products keep their
modules in the package; they are only left out of the public surface.
"""

from __future__ import annotations

PUBLIC_PRODUCTS: tuple[str, ...] = ("swarms", "account", "billing")

HIDDEN_PRODUCTS: tuple[str, ...] = ("index", "forge", "mnemosyne", "menagerie", "tunnels")

# MCP tool names that belong to the account product without an ``account_`` prefix.
_ACCOUNT_TOOL_NAMES = frozenset({"auth_context", "mutation_resolve"})


def tool_product(name: str) -> str:
    """Product that owns an MCP tool, derived from its stable name."""
    if name in _ACCOUNT_TOOL_NAMES:
        return "account"
    prefix = name.split("_", 1)[0]
    return prefix


def is_public(product: str) -> bool:
    return product in PUBLIC_PRODUCTS
