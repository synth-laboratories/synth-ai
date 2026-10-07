"""AST-extract HTTP call sites (method, path template, operation id) from the synth-ai SDK.
Usage: python extract_sdk_calls.py <sdk_root> <out_json>
Handles f-string paths, local path variables, research_operation and _request helpers.
patterns, and session._request_json("METHOD", path). Path placeholders are normalized to {name}."""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
SCAN = [
    "synth_ai/sdk/research",
    "synth_ai/mcp/research",
    "synth_ai/core/http",
    "synth_ai/sdk/forge",
]
METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
PATH_RE = re.compile(r"^/(smr|api|v1|auth|artifacts|mcp|health|forge)\b")


def expr_name(e):
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return e.attr
    if isinstance(e, ast.Call):
        if e.args:
            return expr_name(e.args[0])
        if isinstance(e.func, (ast.Name, ast.Attribute)):
            return expr_name(e.func)
    if isinstance(e, ast.Subscript):
        s = e.slice
        if isinstance(s, ast.Constant):
            return str(s.value)
        return expr_name(e.value)
    return "x"


ALT = {}


def render(node, env):
    if isinstance(node, ast.IfExp):
        a, b = render(node.body, env), render(node.orelse, env)
        if a is not None and b is not None:
            ALT.setdefault(a, set()).add(b)
        return a if a is not None else b
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = v.value
                # inline a local path prefix variable
                if (
                    isinstance(inner, ast.Name)
                    and inner.id in env
                    and env[inner.id].startswith("/")
                ):
                    parts.append(env[inner.id])
                else:
                    parts.append("{" + expr_name(inner) + "}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = render(node.left, env), render(node.right, env)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    return None


def op_id_of(node):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
        and expr_name(node.func)
        in (
            "research_operation",
            "dataset_revision_publication_operation",
            "_operation_metadata",
            "RESEARCH_OPERATIONS",
        )
    ):
        if node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
    if (
        isinstance(node, ast.Subscript)
        and expr_name(node.value) == "RESEARCH_OPERATIONS"
        and isinstance(node.slice, ast.Constant)
    ):
        return node.slice.value
    return None


records = []
for sub in SCAN:
    base = ROOT / sub
    if not base.exists():
        continue
    for f in sorted(base.rglob("*.py")):
        tree = ast.parse(f.read_text(), str(f))
        funcs = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ] + [tree]
        seen = set()
        in_func = set()
        for fn in funcs[:-1]:
            for n in ast.walk(fn):
                if n is not fn:
                    in_func.add(id(n))
        for fn in funcs:
            if fn is tree:
                seen |= in_func
            env = {}
            for st in ast.walk(fn):
                if (
                    isinstance(st, ast.Assign)
                    and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                ):
                    r = render(st.value, env)
                    if r is not None and r.startswith("/"):
                        env[st.targets[0].id] = r
                if (
                    isinstance(st, ast.AnnAssign)
                    and isinstance(st.target, ast.Name)
                    and st.value is not None
                ):
                    r = render(st.value, env)
                    if r is not None and r.startswith("/"):
                        env[st.target.id] = r
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call) or id(call) in seen:
                    continue
                args = list(call.args) + [
                    k.value for k in call.keywords if k.arg in ("path", "url", "endpoint")
                ]
                path = None
                method = None
                opid = None
                for a in args:
                    r = render(a, env)
                    if r is not None and PATH_RE.match(r) and path is None:
                        path = r
                    if (
                        isinstance(a, ast.Constant)
                        and isinstance(a.value, str)
                        and a.value.upper() in METHODS
                        and a.value.isupper()
                    ):
                        method = a.value
                    o = op_id_of(a)
                    if o:
                        opid = o
                for k in call.keywords:
                    if k.arg == "method" and isinstance(k.value, ast.Constant):
                        method = str(k.value.value).upper()
                    o = op_id_of(k.value)
                    if o:
                        opid = o
                fname = expr_name(call.func)
                if path is None:
                    continue
                if method is None and fname.lower() in (
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                    "_get",
                    "_post",
                    "_put",
                    "_patch",
                    "_delete",
                    "get_json",
                    "post_json",
                ):
                    method = fname.strip("_").split("_")[0].upper()
                # The _request helper names its operation in the first positional argument.
                if (
                    opid is None
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                    and not call.args[0].value.startswith("/")
                    and call.args[0].value.upper() not in METHODS
                ):
                    opid = call.args[0].value
                seen.add(id(call))
                records.append(
                    {
                        "file": str(f.relative_to(ROOT)),
                        "line": call.lineno,
                        "callee": fname,
                        "method": method,
                        "path": path.split("?")[0],
                        "operation_id": opid,
                    }
                )
                for alt in sorted(ALT.get(path, ())):
                    records.append(
                        {
                            "file": str(f.relative_to(ROOT)),
                            "line": call.lineno,
                            "callee": fname,
                            "method": method,
                            "path": alt.split("?")[0],
                            "operation_id": opid,
                            "conditional_branch": True,
                        }
                    )
# dedupe identical (file,line)
uniq = {(r["file"], r["line"], r["path"]): r for r in records}
records = sorted(uniq.values(), key=lambda r: (r["file"], r["line"]))
OUT.write_text(json.dumps(records, indent=2) + "\n")
print(
    len(records),
    "call sites;",
    sum(1 for r in records if r["operation_id"]),
    "with op id;",
    sum(1 for r in records if r["method"]),
    "with method",
)
