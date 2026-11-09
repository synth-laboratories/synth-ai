#!/usr/bin/env python3
"""Static SDK boundary checks. Inspect source only; never import the checked package."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    line: int
    column: int
    code: str
    message: str

    @override
    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.code} {self.message}"


CONTAINERS = {
    "dict",
    "list",
    "set",
    "tuple",
    "Dict",
    "List",
    "Set",
    "Tuple",
    "Mapping",
    "MutableMapping",
    "Sequence",
}
MAPPINGS = {"dict", "Dict", "Mapping", "MutableMapping"}
VALIDATORS = {"model_validate", "model_validate_json", "validate_python", "validate_json"}


class Checker(ast.NodeVisitor):
    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.public_module = not path.stem.startswith("_") or path.stem == "_sync"
        self.bindings: dict[str, str] = {}
        self.aliases: dict[str, ast.expr] = {}
        self.models: dict[str, dict[str, bool]] = {}
        self.function_depth = 0
        self.class_names: list[str] = []
        self.diagnostics: set[Diagnostic] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.bindings[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname is not None else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                self.aliases[node.targets[0].id] = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                self.aliases[node.target.id] = node.value

    def name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return self.name(node.value) + "." + node.attr
        return ""

    def report(self, node: ast.expr | ast.stmt | ast.arg, code: str, message: str) -> None:
        self.diagnostics.add(
            Diagnostic(str(self.path), node.lineno, node.col_offset + 1, code, message)
        )

    def annotation_nodes(
        self, node: ast.AST, seen: frozenset[str] = frozenset()
    ) -> Iterator[ast.AST]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                yield from self.annotation_nodes(ast.parse(node.value, mode="eval").body, seen)
            except SyntaxError:
                pass  # Python and ty report invalid annotations.
            return
        if isinstance(node, ast.Name) and node.id in self.aliases and node.id not in seen:
            yield from self.annotation_nodes(self.aliases[node.id], seen | {node.id})
            return
        yield node
        if isinstance(node, ast.Subscript):
            kind = self.name(node.value).split(".")[-1]
            if kind == "Literal":
                return  # Literal strings are values, never forward type references.
            if kind == "Annotated" and isinstance(node.slice, ast.Tuple):
                yield from self.annotation_nodes(node.slice.elts[0], seen)
                return  # Metadata is not an annotation.
        for child in ast.iter_child_nodes(node):
            yield from self.annotation_nodes(child, seen)

    def annotation(self, node: ast.expr, *, boundary: bool) -> None:
        nodes = list(self.annotation_nodes(node))
        subscript_bases = {id(item.value) for item in nodes if isinstance(item, ast.Subscript)}
        for item in nodes:
            name = self.name(item)
            if name in {"Any", "typing.Any", "typing_extensions.Any"}:
                self.report(
                    node,
                    "SYN001",
                    "Any is forbidden; use a named type or JsonValue at a JSON boundary",
                )
            if not boundary:
                continue
            if name.split(".")[-1] in CONTAINERS and id(item) not in subscript_bases:
                self.report(node, "SYN003", "boundary collections must declare their element types")
            if isinstance(item, ast.Subscript) and self.name(item.value).split(".")[-1] in MAPPINGS:
                if any(
                    self.name(child) in {"object", "builtins.object"}
                    for child in self.annotation_nodes(item.slice)
                ):
                    self.report(
                        node,
                        "SYN003",
                        "unstructured object maps are forbidden at domain boundaries; use a record or JsonValue",
                    )

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"typing", "typing_extensions"}:
            for alias in node.names:
                if alias.name in {"Any", "*"}:
                    self.report(node, "SYN001", "do not import Any or wildcard typing names")

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.name(node) in {"typing.Any", "typing_extensions.Any"}:
            self.report(node, "SYN001", "Any is forbidden, including qualified aliases")
        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.name(node.annotation).split(".")[-1] == "TypeAlias" and node.value is not None:
            self.annotation(node.value, boundary=False)
        self.annotation(
            node.annotation, boundary=bool(self.class_names) and self.function_depth == 0
        )
        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        public = (
            self.public_module
            and self.function_depth == 0
            and all(not name.startswith("_") for name in self.class_names)
            and (not node.name.startswith("_") or node.name == "__init__")
        )
        if public and node.args.kwarg is not None:
            self.report(
                node.args.kwarg,
                "SYN002",
                "public APIs must name their parameters; do not accept **kwargs bags",
            )
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        args += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
        for arg in args:
            if arg.annotation is not None:
                self.annotation(arg.annotation, boundary=public)
        if node.returns is not None:
            self.annotation(node.returns, boundary=public)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        config: dict[str, bool] | None = None
        for base in node.bases:
            name = self.name(base)
            if name == "pydantic.BaseModel":
                config = {"strict": False, "frozen": False}
            elif name.endswith("_wire.WireModel"):
                config = {"strict": True, "frozen": True}
            elif isinstance(base, ast.Name) and base.id in self.models:
                config = self.models[base.id].copy()
        if config is not None:
            for statement in node.body:
                value = None
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "model_config"
                    for target in statement.targets
                ):
                    value = statement.value
                elif (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == "model_config"
                ):
                    value = statement.value
                if value is None:
                    continue
                if isinstance(value, ast.Call) and self.name(value.func) in {
                    "pydantic.ConfigDict",
                    "dict",
                }:
                    settings = {kw.arg: kw.value for kw in value.keywords}
                elif isinstance(value, ast.Dict):
                    settings = {
                        key.value: item
                        for key, item in zip(value.keys, value.values, strict=True)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
                else:
                    settings = {"strict": value, "frozen": value}
                for flag in ("strict", "frozen"):
                    if flag in settings:
                        setting = settings[flag]
                        config[flag] = isinstance(setting, ast.Constant) and setting.value is True
                if None in settings:  # **options cannot prove either invariant.
                    config = {"strict": False, "frozen": False}
            # Pydantic also accepts configuration as class keywords.
            for keyword in node.keywords:
                if keyword.arg in config:
                    config[keyword.arg] = (
                        isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    )
            self.models[node.name] = config
            for flag in ("strict", "frozen"):
                if not config[flag]:
                    self.report(
                        node,
                        "SYN004",
                        f"Pydantic records must enable {flag}=True (directly or through a checked base)",
                    )
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    @override
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in {"model_construct", "construct"} or (
                method in {"model_copy", "copy"}
                and any(kw.arg in {"update", None} for kw in node.keywords)
            ):
                self.report(
                    node, "SYN005", "do not bypass record validation; validate a new record instead"
                )
            if method in VALIDATORS and any(
                kw.arg == "strict"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value is True)
                for kw in node.keywords
            ):
                self.report(node, "SYN005", "explicit strict validation must be True")
        self.generic_visit(node)


def check_source(source: str, path: Path) -> list[Diagnostic]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [
            Diagnostic(
                str(path),
                error.lineno or 1,
                error.offset or 1,
                "SYN000",
                "cannot parse Python source",
            )
        ]
    checker = Checker(path, tree)
    checker.visit(tree)
    return sorted(checker.diagnostics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("synth_ai")])
    args = parser.parse_args()
    files: set[Path] = set()
    for path in args.paths:
        if not path.exists():
            parser.error(f"path does not exist: {path}")
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    if not files:
        parser.error("no Python files found")
    diagnostics = [item for path in sorted(files) for item in check_source(path.read_text(), path)]
    for diagnostic in diagnostics:
        print(diagnostic)
    if diagnostics:
        print(f"{len(diagnostics)} standard violation(s)")
        return 1
    print(f"SDK standards passed ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
