"""Test that .env file writing preserves existing content, comments, and formatting."""

import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.utils import write_env_value
from synth_ai.cli.lib.task_app_env import save_to_env_file


def test_write_env_value_preserves_comments_and_empty_lines() -> None:
    """Test that write_env_value doesn't destroy comments and empty lines."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # Write initial .env file with comments, empty lines, and multiple vars
        initial_content = """# This is a comment
EXISTING_VAR=existing_value

# Another comment with empty line above
ANOTHER_VAR=another_value

# Trailing comment
"""
        env_path.write_text(initial_content)
    
    try:
        # Write a new variable - should preserve all existing content
        write_env_value(env_path, "NEW_VAR", "new_value")
        
        # Read back and verify all content is preserved
        result = env_path.read_text()
        
        # Check that comments are preserved
        assert "# This is a comment" in result
        assert "# Another comment with empty line above" in result
        assert "# Trailing comment" in result
        
        # Check that empty lines are preserved
        assert "\n\n" in result or result.count("\n") >= initial_content.count("\n")
        
        # Check that existing vars are preserved
        assert "EXISTING_VAR=existing_value" in result
        assert "ANOTHER_VAR=another_value" in result
        
        # Check that new var was added
        assert "NEW_VAR=new_value" in result
        
        # Verify we didn't lose any lines (allowing for one newline difference)
        initial_lines = initial_content.splitlines()
        result_lines = result.splitlines()
        # Should have at least the same number of non-empty lines plus the new one
        assert len([l for l in result_lines if l.strip()]) >= len([l for l in initial_lines if l.strip()]) + 1
        
    finally:
        env_path.unlink()


def test_save_to_env_file_preserves_comments_and_empty_lines() -> None:
    """Test that save_to_env_file doesn't destroy comments and empty lines."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # Write initial .env file with comments, empty lines, and multiple vars
        initial_content = """# Configuration file
EXISTING_VAR=existing_value

# Section separator

ANOTHER_VAR=another_value
# End of file comment
"""
        env_path.write_text(initial_content)
    
    try:
        # Write a new variable - should preserve all existing content
        save_to_env_file(env_path, "NEW_VAR", "new_value")
        
        # Read back and verify all content is preserved
        result = env_path.read_text()
        
        # Check that comments are preserved
        assert "# Configuration file" in result
        assert "# Section separator" in result
        assert "# End of file comment" in result
        
        # Check that empty lines are preserved (at least one double newline)
        assert "\n\n" in result
        
        # Check that existing vars are preserved
        assert "EXISTING_VAR=existing_value" in result
        assert "ANOTHER_VAR=another_value" in result
        
        # Check that new var was added
        assert "NEW_VAR=new_value" in result
        
    finally:
        env_path.unlink()


def test_write_env_value_updates_existing_var_preserves_formatting() -> None:
    """Test that updating an existing var preserves other content."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        initial_content = """# Header comment
VAR1=value1
VAR2=value2
# Footer comment
"""
        env_path.write_text(initial_content)
    
    try:
        # Update VAR1
        write_env_value(env_path, "VAR1", "updated_value1")
        
        result = env_path.read_text()
        
        # Check VAR1 was updated
        assert "VAR1=updated_value1" in result
        assert "VAR1=value1" not in result
        
        # Check VAR2 is still there
        assert "VAR2=value2" in result
        
        # Check comments are preserved
        assert "# Header comment" in result
        assert "# Footer comment" in result
        
    finally:
        env_path.unlink()


def test_save_to_env_file_updates_existing_var_preserves_formatting() -> None:
    """Test that updating an existing var preserves other content."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        initial_content = """# Header comment
VAR1=value1
VAR2=value2
# Footer comment
"""
        env_path.write_text(initial_content)
    
    try:
        # Update VAR1
        save_to_env_file(env_path, "VAR1", "updated_value1")
        
        result = env_path.read_text()
        
        # Check VAR1 was updated
        assert "VAR1=updated_value1" in result
        assert "VAR1=value1" not in result
        
        # Check VAR2 is still there
        assert "VAR2=value2" in result
        
        # Check comments are preserved
        assert "# Header comment" in result
        assert "# Footer comment" in result
        
    finally:
        env_path.unlink()

