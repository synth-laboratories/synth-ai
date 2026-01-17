#!/usr/bin/env python3
"""Detect API drift between demos/, benchmarks/ and synth_ai/.

This script performs static analysis to catch breaking changes:
- Renamed/removed functions or classes
- Changed function signatures (added/removed/renamed parameters)
- Moved modules
- Missing required parameters

Usage:
    python scripts/check_api_drift.py
    python scripts/check_api_drift.py --verbose
    python scripts/check_api_drift.py demos/gepa_banking77/run_demo.py
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
CONSUMER_DIRS = [
    REPO_ROOT / "demos",
    REPO_ROOT / "benchmarks",
]


@dataclass
class ImportedName:
    """A name imported from synth_ai."""

    module: str  # e.g., "synth_ai.sdk.task"
    name: str  # e.g., "RolloutRequest"
    alias: str | None  # e.g., "RR" if `import X as RR`
    file: Path
    line: int


@dataclass
class FunctionCall:
    """A call to a synth_ai function/class."""

    name: str  # The name used in code (might be alias)
    resolved_module: str | None  # The actual synth_ai module
    resolved_name: str | None  # The actual name in that module
    resolved_class: str | None  # If this is a class method, the class name
    args: list[str]  # Positional arg representations
    kwargs: list[str]  # Keyword argument names used
    file: Path
    line: int


@dataclass
class SignatureMismatch:
    """A detected signature mismatch."""

    call: FunctionCall
    actual_params: list[str]
    issue: str
    severity: str = "error"  # error, warning


@dataclass
class AnalysisResult:
    """Result of analyzing a file."""

    file: Path
    imports: list[ImportedName] = field(default_factory=list)
    calls: list[FunctionCall] = field(default_factory=list)
    import_errors: list[str] = field(default_factory=list)
    signature_mismatches: list[SignatureMismatch] = field(default_factory=list)


class SynthAIUsageVisitor(ast.NodeVisitor):
    """AST visitor that extracts synth_ai imports and usages."""

    def __init__(self, file_path: Path) -> None:
        self.file = file_path
        self.imports: list[ImportedName] = []
        self.calls: list[FunctionCall] = []

        # Map from local name -> (module, actual_name)
        self.name_map: dict[str, tuple[str, str]] = {}
        # Map from module alias -> module
        self.module_map: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        """Handle `import synth_ai.X` or `import synth_ai.X as Y`."""
        for alias in node.names:
            if alias.name.startswith("synth_ai"):
                local_name = alias.asname or alias.name
                self.module_map[local_name] = alias.name
                self.imports.append(
                    ImportedName(
                        module=alias.name,
                        name="<module>",
                        alias=alias.asname,
                        file=self.file,
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle `from synth_ai.X import Y` or `from synth_ai.X import Y as Z`."""
        if node.module and node.module.startswith("synth_ai"):
            for alias in node.names:
                local_name = alias.asname or alias.name
                self.name_map[local_name] = (node.module, alias.name)
                self.imports.append(
                    ImportedName(
                        module=node.module,
                        name=alias.name,
                        alias=alias.asname,
                        file=self.file,
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function/class calls."""
        resolved_module = None
        resolved_name = None
        call_name = None

        # Try to resolve the call target
        if isinstance(node.func, ast.Name):
            # Direct call: RolloutRequest(...)
            call_name = node.func.id
            if call_name in self.name_map:
                resolved_module, resolved_name = self.name_map[call_name]

        resolved_class = None

        if isinstance(node.func, ast.Attribute):
            # Attribute call: synth_ai.sdk.task.RolloutRequest(...)
            # or: PromptLearningJob.from_dict(...)
            # or: client.rollout(...)
            parts = self._get_attribute_parts(node.func)
            if parts:
                call_name = ".".join(parts)
                # Check if first part is a known module alias
                if parts[0] in self.module_map:
                    full_module = self.module_map[parts[0]]
                    if len(parts) > 1:
                        resolved_module = (
                            full_module + "." + ".".join(parts[1:-1])
                            if len(parts) > 2
                            else full_module
                        )
                        resolved_name = parts[-1]
                elif parts[0] in self.name_map:
                    # It's a call on an imported class (e.g., PromptLearningJob.from_dict)
                    resolved_module, class_or_func = self.name_map[parts[0]]
                    if len(parts) > 1:
                        # This is a method/classmethod call on the class
                        resolved_class = class_or_func
                        resolved_name = parts[-1]  # Method name
                    else:
                        # Direct call to the imported name (constructor)
                        resolved_name = class_or_func

        # Only track if it's a synth_ai call
        if resolved_module and resolved_module.startswith("synth_ai"):
            # Extract argument info
            args = [self._node_repr(arg) for arg in node.args]
            kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]

            self.calls.append(
                FunctionCall(
                    name=call_name or "<unknown>",
                    resolved_module=resolved_module,
                    resolved_name=resolved_name,
                    resolved_class=resolved_class,
                    args=args,
                    kwargs=kwargs,
                    file=self.file,
                    line=node.lineno,
                )
            )

        self.generic_visit(node)

    def _get_attribute_parts(self, node: ast.Attribute | ast.Name) -> list[str]:
        """Recursively get parts of an attribute access like a.b.c."""
        parts = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return parts

    def _node_repr(self, node: ast.expr) -> str:
        """Get a string representation of an AST node (for args)."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Starred):
            return f"*{self._node_repr(node.value)}"
        else:
            return f"<{type(node).__name__}>"


def parse_file(file_path: Path) -> AnalysisResult:
    """Parse a Python file and extract synth_ai usages."""
    result = AnalysisResult(file=file_path)

    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        result.import_errors.append(f"Syntax error: {e}")
        return result

    visitor = SynthAIUsageVisitor(file_path)
    visitor.visit(tree)

    result.imports = visitor.imports
    result.calls = visitor.calls

    return result


def get_actual_signature(module_name: str, name: str) -> inspect.Signature | None:
    """Get the actual signature of a function/class from synth_ai."""
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, name, None)
        if obj is None:
            return None
        return inspect.signature(obj)
    except Exception:
        return None


def get_actual_object(module_name: str, name: str, class_name: str | None = None) -> Any | None:
    """Get the actual object from synth_ai.

    If class_name is provided, first get the class from the module,
    then get the attribute from the class (for class/static methods).
    """
    try:
        module = importlib.import_module(module_name)
        if class_name:
            cls = getattr(module, class_name, None)
            if cls is None:
                return None
            return getattr(cls, name, None)
        return getattr(module, name, None)
    except Exception:
        return None


def check_import_exists(imp: ImportedName) -> str | None:
    """Check if an import actually exists in synth_ai. Returns error message or None."""
    try:
        module = importlib.import_module(imp.module)
        if imp.name != "<module>" and not hasattr(module, imp.name):
            return f"'{imp.name}' not found in '{imp.module}'"
        return None
    except ImportError as e:
        return f"Cannot import '{imp.module}': {e}"


def check_call_signature(call: FunctionCall) -> SignatureMismatch | None:
    """Check if a function call matches the actual signature."""
    if not call.resolved_module or not call.resolved_name:
        return None

    obj = get_actual_object(call.resolved_module, call.resolved_name, call.resolved_class)
    if obj is None:
        location = (
            f"'{call.resolved_class}.{call.resolved_name}'"
            if call.resolved_class
            else f"'{call.resolved_name}'"
        )
        return SignatureMismatch(
            call=call,
            actual_params=[],
            issue=f"{location} not found in '{call.resolved_module}'",
            severity="error",
        )

    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        # Can't get signature (builtin, etc.)
        return None

    params = sig.parameters
    param_names = list(params.keys())
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())

    # Check kwargs - do they exist as parameters?
    for kwarg in call.kwargs:
        if kwarg not in params and not has_var_keyword:
            return SignatureMismatch(
                call=call,
                actual_params=param_names,
                issue=f"Unknown keyword argument '{kwarg}'",
                severity="error",
            )

    # Find required parameters (no default, not self, not *args/**kwargs)
    required_params = [
        p.name
        for p in params.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name not in ("self", "cls")
    ]

    # Check that all required params are satisfied by either positional args or kwargs
    num_positional = len(call.args)
    kwargs_set = set(call.kwargs)

    # Positional args fill params in order (excluding self)
    positional_params = [
        p.name
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.name not in ("self", "cls")
    ]

    # Params satisfied by positional args
    satisfied_by_positional = set(positional_params[:num_positional])
    # Params satisfied by kwargs
    satisfied_by_kwargs = kwargs_set & set(param_names)
    # All satisfied params
    satisfied = satisfied_by_positional | satisfied_by_kwargs

    # Check for missing required params
    missing = [p for p in required_params if p not in satisfied]
    if missing:
        return SignatureMismatch(
            call=call,
            actual_params=param_names,
            issue=f"Missing required parameter(s): {missing}",
            severity="error",
        )

    # Check for too many positional args
    max_positional = len(positional_params)
    if not has_var_positional and num_positional > max_positional:
        return SignatureMismatch(
            call=call,
            actual_params=param_names,
            issue=f"Too many positional arguments: got {num_positional}, max {max_positional}",
            severity="error",
        )

    return None


def analyze_directory(directory: Path, verbose: bool = False) -> list[AnalysisResult]:
    """Analyze all Python files in a directory."""
    results = []

    if not directory.exists():
        return results

    for py_file in sorted(directory.rglob("*.py")):
        if verbose:
            print(f"  Analyzing {py_file.relative_to(REPO_ROOT)}", file=sys.stderr)

        result = parse_file(py_file)

        # Check imports exist
        for imp in result.imports:
            error = check_import_exists(imp)
            if error:
                result.import_errors.append(f"Line {imp.line}: {error}")

        # Check call signatures
        for call in result.calls:
            mismatch = check_call_signature(call)
            if mismatch:
                result.signature_mismatches.append(mismatch)

        results.append(result)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for API drift between demos/benchmarks and synth_ai"
    )
    parser.add_argument(
        "files", nargs="*", help="Specific files to check (default: demos/ and benchmarks/)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Ensure synth_ai is importable from repo
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    all_results: list[AnalysisResult] = []

    if args.files:
        # If any file is in synth_ai/, the API changed - check ALL consumers
        synth_ai_changed = any(
            "synth_ai" in Path(f).parts
            and "demos" not in Path(f).parts
            and "benchmarks" not in Path(f).parts
            for f in args.files
        )

        if synth_ai_changed:
            if args.verbose:
                print("synth_ai changed - checking all consumers...", file=sys.stderr)
            for directory in CONSUMER_DIRS:
                if args.verbose:
                    print(f"Checking {directory.name}/...", file=sys.stderr)
                all_results.extend(analyze_directory(directory, verbose=args.verbose))
        else:
            # Only check the specific demo/benchmark files that changed
            for file_path in args.files:
                path = Path(file_path).resolve()
                if not path.exists():
                    print(f"File not found: {file_path}", file=sys.stderr)
                    continue
                result = parse_file(path)
                for imp in result.imports:
                    error = check_import_exists(imp)
                    if error:
                        result.import_errors.append(f"Line {imp.line}: {error}")
                for call in result.calls:
                    mismatch = check_call_signature(call)
                    if mismatch:
                        result.signature_mismatches.append(mismatch)
                all_results.append(result)
    else:
        # Check all consumer directories
        for directory in CONSUMER_DIRS:
            if args.verbose:
                print(f"Checking {directory.name}/...", file=sys.stderr)
            all_results.extend(analyze_directory(directory, verbose=args.verbose))

    # Report results
    total_imports = 0
    total_calls = 0
    total_import_errors = 0
    total_signature_mismatches = 0

    for result in all_results:
        total_imports += len(result.imports)
        total_calls += len(result.calls)

        rel_path = result.file.relative_to(REPO_ROOT)

        for error in result.import_errors:
            total_import_errors += 1
            print(f"IMPORT ERROR: {rel_path}: {error}")

        for mismatch in result.signature_mismatches:
            total_signature_mismatches += 1
            severity = mismatch.severity.upper()
            print(f"SIGNATURE {severity}: {rel_path}:{mismatch.call.line}")
            print(f"  Call: {mismatch.call.name}(...)")
            if mismatch.call.kwargs:
                print(f"  Kwargs used: {mismatch.call.kwargs}")
            print(f"  Issue: {mismatch.issue}")
            if mismatch.actual_params:
                print(f"  Actual params: {mismatch.actual_params}")
            print()

    # Summary
    print(f"\n{'=' * 60}")
    print("API Drift Check Summary")
    print(f"{'=' * 60}")
    print(f"Files analyzed:       {len(all_results)}")
    print(f"synth_ai imports:     {total_imports}")
    print(f"synth_ai calls:       {total_calls}")
    print(f"Import errors:        {total_import_errors}")
    print(f"Signature mismatches: {total_signature_mismatches}")
    print(f"{'=' * 60}")

    if total_import_errors > 0 or total_signature_mismatches > 0:
        print("\n❌ API drift detected!")
        return 1
    else:
        print("\n✓ No API drift detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
