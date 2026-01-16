#!/usr/bin/env python3
"""Check that URLs in synth_ai/core/urls.py match actual backend/frontend routes.

This script validates that SDK URL definitions correspond to real endpoints,
preventing drift where the SDK references non-existent routes.

Usage:
    python scripts/check_url_backend_drift.py --backend-path /path/to/monorepo/backend --frontend-path /path/to/monorepo/frontend

Requires both repos to be checked out (CI does this via actions/checkout).
"""

import argparse
import ast
import re
import sys
from pathlib import Path


def extract_sdk_url_paths(urls_file: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Extract URL path patterns from synth_ai/core/urls.py.

    Returns tuple of (backend_urls, frontend_urls) where each is
    a dict mapping function_name -> url_path_pattern.

    Backend functions start with 'synth_', frontend functions start with 'frontend_'.
    """
    content = urls_file.read_text()
    tree = ast.parse(content)

    backend_urls: dict[str, str] = {}
    frontend_urls: dict[str, str] = {}

    # Skip generic helper functions that take dynamic paths (not specific endpoints)
    skip_functions = {
        "frontend_url",  # Generic helper - takes dynamic path
        "frontend_api_url",  # Generic helper - takes dynamic path
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Skip private/internal functions
            if func_name.startswith("_"):
                continue

            # Skip generic helper functions
            if func_name in skip_functions:
                continue

            # Look for return statements with f-strings or string concatenation
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    path = _extract_path_from_expr(child.value, func_name)
                    if path:
                        if func_name.startswith("frontend_"):
                            frontend_urls[func_name] = path
                        else:
                            # synth_* and other functions are backend
                            backend_urls[func_name] = path
                        break

    return backend_urls, frontend_urls


def _extract_path_from_expr(expr: ast.expr, func_name: str = "") -> str | None:
    """Extract URL path from an AST expression."""
    if isinstance(expr, ast.JoinedStr):
        # f-string
        parts = []
        for value in expr.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                # Variable interpolation - use placeholder
                parts.append("{param}")
        return "".join(parts)

    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        # String concatenation
        left = _extract_path_from_expr(expr.left, func_name)
        right = _extract_path_from_expr(expr.right, func_name)
        if left and right:
            return left + right
        return left or right

    elif isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value

    elif isinstance(expr, ast.Call):
        # Function call - try to resolve known patterns
        called_func = ""
        if isinstance(expr.func, ast.Name):
            called_func = expr.func.id
        elif isinstance(expr.func, ast.Attribute):
            called_func = expr.func.attr

        # Extract path argument if present
        path_arg = ""
        if expr.args:
            first_arg = expr.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                path_arg = first_arg.value

        # Resolve known frontend helper functions
        if called_func == "frontend_url" and path_arg:
            return f"/{path_arg}"
        elif called_func == "frontend_api_url" and path_arg:
            return f"/api/{path_arg}"
        elif path_arg:
            # Generic fallback for other function calls
            return f"/.../{path_arg}"

    return None


def extract_backend_routes(backend_path: Path) -> set[str]:
    """Extract registered routes from backend FastAPI app.

    Parses route files to find @router.get/post/etc decorators.
    """
    routes: set[str] = set()
    routes_dir = backend_path / "app" / "routes"
    api_v1_dir = backend_path / "app" / "api" / "v1"

    # Pattern to match route decorators
    route_pattern = re.compile(
        r'@(?:router|app)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE
    )

    # Also match include_router calls with prefixes
    include_pattern = re.compile(
        r'include_router\s*\([^)]*prefix\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE
    )

    # Scan route files
    for py_file in list(routes_dir.rglob("*.py")) + list(api_v1_dir.rglob("*.py")):
        try:
            content = py_file.read_text()
        except Exception:
            continue

        # Find route decorators
        for match in route_pattern.finditer(content):
            path = match.group(2)
            routes.add(path)

        # Find router prefixes
        for match in include_pattern.finditer(content):
            prefix = match.group(1)
            routes.add(prefix)

    # Parse main.py for router registrations
    main_py = routes_dir / "main.py"
    if main_py.exists():
        content = main_py.read_text()
        for match in include_pattern.finditer(content):
            prefix = match.group(1)
            routes.add(prefix)

    return routes


def extract_frontend_routes(frontend_path: Path) -> set[str]:
    """Extract routes from Next.js App Router frontend.

    Scans app/ directory for page.tsx files and api/ route handlers.
    """
    routes: set[str] = set()

    # Check both possible locations for app directory
    app_dir = frontend_path / "app"
    if not app_dir.exists():
        app_dir = frontend_path / "src" / "app"
    if not app_dir.exists():
        return routes

    # Find all page.tsx and route.ts files (Next.js App Router)
    for file_path in app_dir.rglob("*.tsx"):
        if file_path.name in ("page.tsx", "layout.tsx"):
            # Convert file path to route
            rel_path = file_path.parent.relative_to(app_dir)
            route = "/" + str(rel_path).replace("\\", "/")
            # Normalize (group) segments like (auth) -> remove them
            route = re.sub(r"/\([^)]+\)", "", route)
            # Convert [param] to {param}
            route = re.sub(r"\[([^\]]+)\]", r"{\1}", route)
            route = route.rstrip("/")
            if route == "":
                route = "/"
            routes.add(route)

    for file_path in app_dir.rglob("route.ts"):
        # API routes
        rel_path = file_path.parent.relative_to(app_dir)
        route = "/" + str(rel_path).replace("\\", "/")
        route = re.sub(r"/\([^)]+\)", "", route)
        route = re.sub(r"\[([^\]]+)\]", r"{\1}", route)
        route = route.rstrip("/")
        routes.add(route)

    return routes


def normalize_path(path: str) -> str:
    """Normalize a URL path for comparison.

    - Remove base URL portions
    - Normalize path parameters to {param}
    - Remove trailing slashes
    """
    # Extract just the path portion
    if "://" in path:
        path = "/" + path.split("://", 1)[1].split("/", 1)[-1]

    # Normalize various parameter formats
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    path = re.sub(r":\w+", "{param}", path)  # Express-style :id

    # Remove trailing slash
    path = path.rstrip("/")

    # Remove leading .../ from partial paths
    path = re.sub(r"^/\.\.\.", "", path)

    return path


def path_matches(sdk_path: str, backend_routes: set[str]) -> bool:
    """Check if an SDK path matches any backend route."""
    normalized_sdk = normalize_path(sdk_path)

    # Direct match
    if normalized_sdk in backend_routes:
        return True

    # Check if it's a sub-path of a registered prefix
    for route in backend_routes:
        normalized_route = normalize_path(route)
        if normalized_sdk.startswith(normalized_route):
            return True
        if normalized_route.startswith(normalized_sdk):
            return True

    # Fuzzy match - check if path segments align
    sdk_segments = normalized_sdk.split("/")
    for route in backend_routes:
        route_segments = normalize_path(route).split("/")
        if _segments_match(sdk_segments, route_segments):
            return True

    return False


def _segments_match(sdk_segments: list[str], route_segments: list[str]) -> bool:
    """Check if path segments match, accounting for parameters."""
    # SDK path should be same length or longer (more specific)
    if len(sdk_segments) < len(route_segments):
        return False

    for i, route_seg in enumerate(route_segments):
        if i >= len(sdk_segments):
            break
        sdk_seg = sdk_segments[i]

        # Parameters match anything
        if route_seg == "{param}" or sdk_seg == "{param}":
            continue

        # Exact match required
        if route_seg != sdk_seg:
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK URLs against backend/frontend routes")
    parser.add_argument(
        "--backend-path", type=Path, required=True, help="Path to monorepo/backend directory"
    )
    parser.add_argument(
        "--frontend-path",
        type=Path,
        default=None,
        help="Path to monorepo/frontend directory (optional)",
    )
    parser.add_argument(
        "--urls-file", type=Path, default=None, help="Path to urls.py (default: auto-detect)"
    )
    args = parser.parse_args()

    # Find urls.py
    if args.urls_file:
        urls_file = args.urls_file
    else:
        # Auto-detect relative to script location
        script_dir = Path(__file__).parent.parent
        urls_file = script_dir / "synth_ai" / "core" / "urls.py"

    if not urls_file.exists():
        print(f"ERROR: urls.py not found at {urls_file}")
        return 1

    if not args.backend_path.exists():
        print(f"ERROR: Backend path not found at {args.backend_path}")
        return 1

    # Extract URL patterns from SDK
    print(f"Parsing SDK URLs from {urls_file}...")
    backend_urls, frontend_urls = extract_sdk_url_paths(urls_file)
    print(f"Found {len(backend_urls)} backend URL functions (synth_*)")
    print(f"Found {len(frontend_urls)} frontend URL functions (frontend_*)")

    # Extract routes from backend
    print(f"\nParsing backend routes from {args.backend_path}...")
    backend_routes = extract_backend_routes(args.backend_path)
    print(f"Found {len(backend_routes)} backend route patterns")

    # Check backend URLs
    missing_backend: list[tuple[str, str]] = []
    for func_name, url_path in backend_urls.items():
        if not path_matches(url_path, backend_routes):
            missing_backend.append((func_name, url_path))

    # Check frontend URLs if path provided
    missing_frontend: list[tuple[str, str]] = []
    if args.frontend_path:
        if not args.frontend_path.exists():
            print(f"ERROR: Frontend path not found at {args.frontend_path}")
            return 1

        print(f"\nParsing frontend routes from {args.frontend_path}...")
        frontend_routes = extract_frontend_routes(args.frontend_path)
        print(f"Found {len(frontend_routes)} frontend route patterns")

        for func_name, url_path in frontend_urls.items():
            if not path_matches(url_path, frontend_routes):
                missing_frontend.append((func_name, url_path))
    elif frontend_urls:
        print("\nSkipping frontend URL check (no --frontend-path provided)")
        print(f"  {len(frontend_urls)} frontend URLs not validated")

    # Report results
    has_failures = bool(missing_backend or missing_frontend)

    if missing_backend:
        print("\n" + "=" * 60)
        print("BACKEND URL DRIFT DETECTED")
        print("-" * 60)
        print("The following SDK URLs don't match any backend routes:")
        for func_name, url_path in sorted(missing_backend):
            print(f"  {func_name}() -> {url_path}")

    if missing_frontend:
        print("\n" + "=" * 60)
        print("FRONTEND URL DRIFT DETECTED")
        print("-" * 60)
        print("The following SDK URLs don't match any frontend routes:")
        for func_name, url_path in sorted(missing_frontend):
            print(f"  {func_name}() -> {url_path}")

    if has_failures:
        total = len(missing_backend) + len(missing_frontend)
        print("\n" + "=" * 60)
        print(f"URL Drift Check FAILED: {total} potentially drifted URLs")
        print("\nNote: Some false positives may occur for dynamically registered routes.")
        return 1

    print("\n" + "=" * 60)
    print("URL Drift Check PASSED")
    total_checked = len(backend_urls) + (len(frontend_urls) if args.frontend_path else 0)
    print(f"All {total_checked} SDK URL functions match their routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
