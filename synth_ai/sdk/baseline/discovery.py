"""AST-based discovery mechanism for baseline files."""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from synth_ai.sdk.baseline.config import BaselineConfig

# Search patterns for baseline files
BASELINE_FILE_PATTERNS = [
    "**/baseline/*.py",
    "**/baselines/*.py",
    "**/*_baseline.py",
]

# Directories to ignore during discovery
IGNORE_PATTERNS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass
class BaselineChoice:
    """Represents a discovered baseline configuration."""
    
    baseline_id: str
    path: Path
    lineno: int
    source: str  # "discovered" or "registered"
    config: Optional[BaselineConfig] = None


class BaselineConfigVisitor(ast.NodeVisitor):
    """AST visitor to find BaselineConfig instances."""
    
    def __init__(self):
        self.matches: List[Tuple[str, int]] = []  # (baseline_id, lineno)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements looking for BaselineConfig."""
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return
        
        # Check if right-hand side is BaselineConfig(...)
        func = node.value.func
        if isinstance(func, ast.Name) and func.id == "BaselineConfig":
            # Extract baseline_id from constructor args
            baseline_id = self._extract_baseline_id(node.value)
            if baseline_id:
                self.matches.append((baseline_id, node.lineno))
        
        self.generic_visit(node)
    
    def _extract_baseline_id(self, call_node: ast.Call) -> Optional[str]:
        """Extract baseline_id from BaselineConfig constructor."""
        for keyword in call_node.keywords:
            if keyword.arg == "baseline_id" and isinstance(keyword.value, ast.Constant):
                val = keyword.value.value
                if isinstance(val, str):
                    return val
        return None


def should_ignore_path(path: Path) -> bool:
    """Check if a path should be ignored during discovery."""
    return any(part in IGNORE_PATTERNS for part in path.parts)


def discover_baseline_files(search_roots: List[Path]) -> List[BaselineChoice]:
    """Discover baseline files via AST scanning.
    
    Args:
        search_roots: List of root directories to search in
    
    Returns:
        List of BaselineChoice objects representing discovered baselines
    """
    results: List[BaselineChoice] = []
    seen = set()
    
    for root in search_roots:
        if not root.exists():
            continue
        
        for pattern in BASELINE_FILE_PATTERNS:
            for path in root.glob(pattern):
                if should_ignore_path(path):
                    continue
                
                try:
                    source = path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(path))
                except (OSError, SyntaxError):
                    continue
                
                visitor = BaselineConfigVisitor()
                visitor.visit(tree)
                
                for baseline_id, lineno in visitor.matches:
                    key = (baseline_id, path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    results.append(
                        BaselineChoice(
                            baseline_id=baseline_id,
                            path=path.resolve(),
                            lineno=lineno,
                            source="discovered",
                        )
                    )
    
    return results


def load_baseline_config_from_file(
    baseline_id: str,
    path: Path,
) -> BaselineConfig:
    """Load a BaselineConfig from a Python file.
    
    Args:
        baseline_id: The baseline_id to look for
        path: Path to the Python file
    
    Returns:
        BaselineConfig instance
    
    Raises:
        ValueError: If baseline_id not found or file cannot be loaded
    """
    # Load the module
    spec = importlib.util.spec_from_file_location("baseline_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline file: {path}")
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
        raise ImportError(
            f"❌ Missing dependency for baseline '{baseline_id}'\n"
            f"   File: {path}\n"
            f"   Missing module: {missing_module}\n"
            f"   Fix: pip install {missing_module}  (or 'uv add {missing_module}')"
        ) from e
    except SyntaxError as e:
        raise ValueError(
            f"❌ Syntax error in baseline file '{baseline_id}'\n"
            f"   File: {path}\n"
            f"   Error at line {e.lineno}: {e.msg}\n"
            f"   Text: {e.text.strip() if e.text else 'N/A'}\n"
            f"   Fix: Check the Python syntax in the baseline file"
        ) from e
    except Exception as e:
        error_type = type(e).__name__
        raise ValueError(
            f"❌ Failed to load baseline '{baseline_id}'\n"
            f"   File: {path}\n"
            f"   Error type: {error_type}\n"
            f"   Message: {str(e)}\n"
            f"   This may be due to:\n"
            f"     - Missing dependencies (check imports)\n"
            f"     - Configuration errors in the baseline file\n"
            f"     - Environment variables not set\n"
            f"   Tip: Run with --verbose for more details"
        ) from e
    
    # Find the BaselineConfig instance
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        
        attr = getattr(module, attr_name)
        if isinstance(attr, BaselineConfig) and attr.baseline_id == baseline_id:
            # Set source path for reference
            attr._source_path = path
            return attr
    
    # Provide helpful error message
    found_configs = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, BaselineConfig):
            found_configs.append(attr.baseline_id)
    
    if found_configs:
        raise ValueError(
            f"❌ Baseline '{baseline_id}' not found in {path}\n"
            f"   Found baselines in this file: {', '.join(found_configs)}\n"
            f"   Fix: Use one of the above baseline IDs or check the baseline_id parameter"
        )
    else:
        raise ValueError(
            f"❌ No BaselineConfig instances found in {path}\n"
            f"   Expected to find a BaselineConfig with baseline_id='{baseline_id}'\n"
            f"   Fix: Ensure the file defines a BaselineConfig instance with baseline_id='{baseline_id}'"
        )

