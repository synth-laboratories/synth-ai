#!/usr/bin/env python3
"""
CLI: basic calculator for quick math in terminal.
Safe evaluation of arithmetic expressions.
"""

import ast
import operator
from collections.abc import Callable

import click
from rich.console import Console

# Supported operators
_BINARY_OPS: dict[type[ast.AST], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.AST], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Only numeric constants are allowed")
        num_node = getattr(ast, "Num", None)
        if num_node is not None and isinstance(n, num_node):  # pragma: no cover
            numeric_value = getattr(n, "n", None)
            if isinstance(numeric_value, (int, float)):
                return float(numeric_value)
            raise ValueError("Only numeric constants are allowed")
        if isinstance(n, ast.BinOp):
            op_type = type(n.op)
            func = _BINARY_OPS.get(op_type)
            if func:
                return func(_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            op_type = type(n.op)
            func = _UNARY_OPS.get(op_type)
            if func:
                return func(_eval(n.operand))
        if isinstance(n, ast.Expr):
            return _eval(n.value)
        raise ValueError("Unsupported expression")

    result = _eval(node)
    return float(result)


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
