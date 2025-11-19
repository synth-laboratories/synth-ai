"""Test that demonstrates the destructive .env file writing bug."""

import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.utils import write_env_value
from synth_ai.cli.lib.task_app_env import save_to_env_file


def test_write_env_value_destroys_file_when_updating() -> None:
    """
    CRITICAL BUG TEST: This test demonstrates that write_env_value
    destroys .env file content when updating an existing variable.
    
    The bug: Using splitlines() + "\n".join() loses:
    - Original line endings (could be \r\n on Windows)
    - Trailing whitespace on lines
    - Multiple consecutive empty lines might collapse
    - Comments that don't match the regex pattern (though they should be preserved)
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # Create a .env file with specific formatting
        original_content = """# API Keys
SYNTH_API_KEY=sk_test_12345
ENVIRONMENT_API_KEY=env_test_67890

# Other config
DEBUG=true
# End of file
"""
        env_path.write_text(original_content)
        original_bytes = env_path.read_bytes()
        original_line_count = original_content.count('\n')
    
    try:
        # Update an existing variable - this should NOT destroy the file
        write_env_value(env_path, "SYNTH_API_KEY", "sk_test_updated")
        
        result_content = env_path.read_text()
        result_bytes = env_path.read_bytes()
        result_line_count = result_content.count('\n')
        
        # CRITICAL ASSERTIONS - these should ALL pass but currently fail due to the bug
        
        # 1. All original variables should still be present
        assert "ENVIRONMENT_API_KEY=env_test_67890" in result_content, "Lost ENVIRONMENT_API_KEY!"
        assert "DEBUG=true" in result_content, "Lost DEBUG variable!"
        
        # 2. All comments should be preserved
        assert "# API Keys" in result_content, "Lost comment # API Keys!"
        assert "# Other config" in result_content, "Lost comment # Other config!"
        assert "# End of file" in result_content, "Lost comment # End of file!"
        
        # 3. Empty lines should be preserved (at least one)
        assert "\n\n" in result_content or result_content.count("\n\n") >= 1, \
            f"Lost empty lines! Original had empty line, result: {repr(result_content)}"
        
        # 4. Updated variable should have new value
        assert "SYNTH_API_KEY=sk_test_updated" in result_content, "Variable not updated!"
        assert "SYNTH_API_KEY=sk_test_12345" not in result_content, "Old value still present!"
        
        # 5. Line count should be preserved (allowing for one newline difference)
        assert abs(result_line_count - original_line_count) <= 1, \
            f"Lost lines! Original: {original_line_count}, Result: {result_line_count}"
        
        # 6. File should end with newline
        assert result_content.endswith("\n"), "Lost trailing newline!"
        
    finally:
        env_path.unlink()


def test_save_to_env_file_destroys_file_when_updating() -> None:
    """
    CRITICAL BUG TEST: This test demonstrates that save_to_env_file
    destroys .env file content when updating an existing variable.
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        original_content = """# Configuration
VAR1=value1
VAR2=value2

VAR3=value3
# Footer
"""
        env_path.write_text(original_content)
    
    try:
        # Update VAR1 - this triggers the destructive write path (line 183)
        save_to_env_file(env_path, "VAR1", "updated_value1")
        
        result = env_path.read_text()
        
        # All content should be preserved
        assert "VAR2=value2" in result, "Lost VAR2!"
        assert "VAR3=value3" in result, "Lost VAR3!"
        assert "# Configuration" in result, "Lost header comment!"
        assert "# Footer" in result, "Lost footer comment!"
        assert "\n\n" in result, "Lost empty line!"
        assert result.endswith("\n"), "Lost trailing newline!"
        assert "VAR1=updated_value1" in result, "VAR1 not updated!"
        assert "VAR1=value1" not in result, "Old VAR1 value still present!"
        
    finally:
        env_path.unlink()

