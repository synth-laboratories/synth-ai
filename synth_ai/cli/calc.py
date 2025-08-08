#!/usr/bin/env python3
"""
CLI: basic calculator for quick math in terminal.
Safe evaluation of arithmetic expressions.
"""

import ast
import operator as op
import click
from rich.console import Console


# Supported operators
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):  # 3.8 and earlier
            return n.n
        if isinstance(n, ast.Constant):  # 3.8+
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants are allowed")
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Expr):
            return _eval(n.value)
        raise ValueError("Unsupported expression")

    return _eval(node)


def register(cli):
    @cli.command(name="calc")
    @click.argument("expr", nargs=-1)
    def calc(expr: tuple[str, ...]):
        """Evaluate a basic math expression, e.g., "(12_345 + 6789) / 100".

        Supports + - * / // % ** and parentheses. No variables or functions.
        """
        console = Console()
        expression = " ".join(expr).strip()
        if not expression:
            console.print("[dim]Usage:[/dim] synth-ai calc '2*(3+4)' or: uvx . calc 2 + 2")
            return
        try:
            # Allow underscores in numbers for readability
            expression = expression.replace("_", "")
            result = _safe_eval(expression)
            console.print(f"= [bold]{result}[/bold]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

