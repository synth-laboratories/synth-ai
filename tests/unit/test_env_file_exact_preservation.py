"""Test that .env file writing preserves EXACT content including trailing newlines and whitespace."""

import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.utils import write_env_value
from synth_ai.cli.lib.task_app_env import save_to_env_file


def test_write_env_value_preserves_exact_formatting() -> None:
    """Test that write_env_value preserves exact formatting including trailing newlines."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # Write initial .env file with specific formatting
        initial_content = """# Comment line
VAR1=value1

VAR2=value2
# Trailing comment
"""
        env_path.write_text(initial_content)
        original_bytes = env_path.read_bytes()
    
    try:
        # Write a new variable
        write_env_value(env_path, "VAR3", "value3")
        
        # Read back as bytes to check exact preservation
        result_bytes = env_path.read_bytes()
        result_text = env_path.read_text()
        
        # The issue: splitlines() removes trailing newlines, so we lose the final \n
        # Check that we at least preserved the structure
        assert "VAR1=value1" in result_text
        assert "VAR2=value2" in result_text
        assert "VAR3=value3" in result_text
        assert "# Comment line" in result_text
        assert "# Trailing comment" in result_text
        
        # CRITICAL: Check that we didn't lose the empty line between VAR1 and VAR2
        # This is the bug - splitlines() + "\n".join() loses empty lines
        lines = result_text.splitlines(keepends=False)
        var1_idx = next((i for i, line in enumerate(lines) if line.strip() == "VAR1=value1"), None)
        var2_idx = next((i for i, line in enumerate(lines) if line.strip() == "VAR2=value2"), None)
        
        # There should be at least one empty line between VAR1 and VAR2
        if var1_idx is not None and var2_idx is not None:
            between_lines = lines[var1_idx + 1:var2_idx]
            empty_between = [l for l in between_lines if not l.strip()]
            # This will fail if empty lines are lost
            assert len(empty_between) >= 1, f"Lost empty line between VAR1 and VAR2. Lines between: {between_lines}"
        
    finally:
        env_path.unlink()


def test_write_env_value_preserves_trailing_newline() -> None:
    """Test that trailing newline is preserved."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # File with trailing newline
        initial_content = "VAR1=value1\nVAR2=value2\n"
        env_path.write_text(initial_content)
        original_ends_with_newline = initial_content.endswith("\n")
    
    try:
        write_env_value(env_path, "VAR3", "value3")
        result = env_path.read_text()
        
        # File should end with newline
        assert result.endswith("\n"), "Trailing newline was lost!"
        
    finally:
        env_path.unlink()


def test_save_to_env_file_preserves_exact_formatting() -> None:
    """Test that save_to_env_file preserves exact formatting."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        initial_content = """# Comment
VAR1=value1

VAR2=value2
"""
        env_path.write_text(initial_content)
    
    try:
        save_to_env_file(env_path, "VAR3", "value3")
        result = env_path.read_text()
        
        # Check structure is preserved
        assert "VAR1=value1" in result
        assert "VAR2=value2" in result
        assert "VAR3=value3" in result
        
        # Check empty line is preserved
        lines = result.splitlines(keepends=False)
        var1_idx = next((i for i, line in enumerate(lines) if line.strip() == "VAR1=value1"), None)
        var2_idx = next((i for i, line in enumerate(lines) if line.strip() == "VAR2=value2"), None)
        
        if var1_idx is not None and var2_idx is not None:
            between_lines = lines[var1_idx + 1:var2_idx]
            empty_between = [l for l in between_lines if not l.strip()]
            assert len(empty_between) >= 1, f"Lost empty line between VAR1 and VAR2"
        
        # Check trailing newline
        assert result.endswith("\n"), "Trailing newline was lost!"
        
    finally:
        env_path.unlink()

