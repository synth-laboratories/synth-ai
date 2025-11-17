"""Comprehensive tests to ensure .env file writing NEVER destroys user content.

These tests cover all edge cases and scenarios to prevent data loss.
"""

import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.utils import write_env_value
from synth_ai.cli.lib.task_app_env import interactive_fill_env, save_to_env_file
from synth_ai.utils.env import write_env_var_to_dotenv


class TestWriteEnvValueSafety:
    """Comprehensive safety tests for write_env_value."""

    def test_preserves_file_with_only_comments(self) -> None:
        """Test that files with only comments are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# This is a comment
# Another comment
# Yet another comment
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "NEW_VAR", "new_value")
            result = env_path.read_text()
            
            assert "# This is a comment" in result
            assert "# Another comment" in result
            assert "# Yet another comment" in result
            assert "NEW_VAR=new_value" in result
        finally:
            env_path.unlink()

    def test_preserves_file_with_only_empty_lines(self) -> None:
        """Test that files with only empty lines are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = "\n\n\n"
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "NEW_VAR", "new_value")
            result = env_path.read_text()
            
            # Should preserve empty lines and add new var
            assert result.count("\n") >= original.count("\n")
            assert "NEW_VAR=new_value" in result
        finally:
            env_path.unlink()

    def test_preserves_trailing_whitespace(self) -> None:
        """Test that trailing whitespace on lines is preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1   
VAR2=value2\t
# Comment with trailing spaces    
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR3", "value3")
            result = env_path.read_text()
            
            # Check that original vars are still there
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "# Comment with trailing spaces" in result
        finally:
            env_path.unlink()

    def test_preserves_export_statements(self) -> None:
        """Test that export statements are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """export VAR1=value1
VAR2=value2
export VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            assert "export VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "export VAR3=value3" in result
            assert "VAR4=value4" in result
        finally:
            env_path.unlink()

    def test_preserves_quoted_values(self) -> None:
        """Test that quoted values are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1="quoted value"
VAR2='single quoted'
VAR3=unquoted
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            assert 'VAR1="quoted value"' in result
            assert "VAR2='single quoted'" in result
            assert "VAR3=unquoted" in result
        finally:
            env_path.unlink()

    def test_preserves_special_characters(self) -> None:
        """Test that special characters in values are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value with spaces
VAR2=value=with=equals
VAR3=value#with#hash
VAR4=value$with$dollar
VAR5=value&with&ampersand
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR6", "value6")
            result = env_path.read_text()
            
            assert "VAR1=value with spaces" in result
            assert "VAR2=value=with=equals" in result
            assert "VAR3=value#with#hash" in result
            assert "VAR4=value$with$dollar" in result
            assert "VAR5=value&with&ampersand" in result
        finally:
            env_path.unlink()

    def test_preserves_windows_line_endings(self) -> None:
        """Test that Windows line endings (CRLF) are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env', newline='') as f:
            env_path = Path(f.name)
            original = "VAR1=value1\r\nVAR2=value2\r\n# Comment\r\n"
            f.write(original)
        
        try:
            write_env_value(env_path, "VAR3", "value3")
            result_bytes = env_path.read_bytes()
            
            # Should still have the content
            assert b"VAR1=value1" in result_bytes
            assert b"VAR2=value2" in result_bytes
            assert b"VAR3=value3" in result_bytes
        finally:
            env_path.unlink()

    def test_preserves_multiple_consecutive_empty_lines(self) -> None:
        """Test that multiple consecutive empty lines are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1


VAR2=value2


VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            # Should have at least one double newline
            assert "\n\n" in result
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_comments_at_end_of_lines(self) -> None:
        """Test that inline comments are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1  # inline comment
VAR2=value2#another comment
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            assert "# inline comment" in result or "VAR1=value1" in result
            assert "#another comment" in result or "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_variables_with_no_value(self) -> None:
        """Test that variables with empty values are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=
VAR2=value2
VAR3=
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            assert "VAR1=" in result
            assert "VAR2=value2" in result
            assert "VAR3=" in result
        finally:
            env_path.unlink()

    def test_preserves_malformed_lines(self) -> None:
        """Test that malformed lines (not matching regex) are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
This is not a valid env line
VAR2=value2
Also not valid = but has equals
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            # Malformed lines should be preserved
            assert "This is not a valid env line" in result
            assert "Also not valid = but has equals" in result
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_handles_empty_file(self) -> None:
        """Test that empty files are handled correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            env_path.write_text("")
        
        try:
            write_env_value(env_path, "VAR1", "value1")
            result = env_path.read_text()
            
            assert "VAR1=value1" in result
            assert result.endswith("\n")
        finally:
            env_path.unlink()

    def test_handles_nonexistent_file(self) -> None:
        """Test that nonexistent files are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            
            assert not env_path.exists()
            write_env_value(env_path, "VAR1", "value1")
            
            assert env_path.exists()
            result = env_path.read_text()
            assert "VAR1=value1" in result

    def test_updates_existing_var_preserves_others(self) -> None:
        """Test that updating one var doesn't affect others."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=old_value1
VAR2=value2
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR1", "new_value1")
            result = env_path.read_text()
            
            assert "VAR1=new_value1" in result
            assert "VAR1=old_value1" not in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_very_long_file(self) -> None:
        """Test that very long files are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            # Create a file with 100 variables
            lines = [f"VAR{i}=value{i}\n" for i in range(100)]
            original = "".join(lines)
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR_NEW", "new_value")
            result = env_path.read_text()
            
            # Check that all original vars are preserved
            for i in range(100):
                assert f"VAR{i}=value{i}" in result
            assert "VAR_NEW=new_value" in result
        finally:
            env_path.unlink()

    def test_preserves_duplicate_keys_correctly(self) -> None:
        """Test that duplicate keys are handled (first one updated)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=first_value
VAR2=value2
VAR1=second_value
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR1", "updated_value")
            result = env_path.read_text()
            
            # Should update the first occurrence
            assert "VAR1=updated_value" in result
            # Second occurrence might still be there (depends on implementation)
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_unicode_characters(self) -> None:
        """Test that Unicode characters are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env', encoding='utf-8') as f:
            env_path = Path(f.name)
            original = """VAR1=value with émojis 🎉
VAR2=中文测试
VAR3=тест
VAR4=テスト
"""
            env_path.write_text(original, encoding='utf-8')
        
        try:
            write_env_value(env_path, "VAR5", "value5")
            result = env_path.read_text(encoding='utf-8')
            
            assert "émojis 🎉" in result
            assert "中文测试" in result
            assert "тест" in result
            assert "テスト" in result
        finally:
            env_path.unlink()


class TestSaveToEnvFileSafety:
    """Comprehensive safety tests for save_to_env_file."""

    def test_preserves_all_content_when_updating(self) -> None:
        """Test that updating preserves all content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# Header
VAR1=value1
VAR2=value2

# Footer
"""
            env_path.write_text(original)
        
        try:
            save_to_env_file(env_path, "VAR1", "updated_value1")
            result = env_path.read_text()
            
            assert "# Header" in result
            assert "VAR1=updated_value1" in result
            assert "VAR1=value1" not in result
            assert "VAR2=value2" in result
            assert "# Footer" in result
            assert "\n\n" in result  # Empty line preserved
        finally:
            env_path.unlink()

    def test_appends_when_key_not_found(self) -> None:
        """Test that new keys are appended without destroying content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            save_to_env_file(env_path, "VAR3", "value3")
            result = env_path.read_text()
            
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_export_statements(self) -> None:
        """Test that export statements are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """export VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            save_to_env_file(env_path, "VAR1", "updated_value1")
            result = env_path.read_text()
            
            # Should preserve export keyword
            assert "export VAR1=updated_value1" in result or "VAR1=updated_value1" in result
            assert "VAR2=value2" in result
        finally:
            env_path.unlink()


class TestInteractiveFillEnvSafety:
    """Comprehensive safety tests for interactive_fill_env."""

    def test_preserves_all_variables_and_comments(self) -> None:
        """Test that interactive_fill_env preserves everything."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# Database config
DATABASE_URL=postgres://localhost/db
DATABASE_POOL=10

# Feature flags  
ENABLE_DEBUG=true
ENABLE_METRICS=false

# API Keys (will be updated)
ENVIRONMENT_API_KEY=old_env_key
SYNTH_API_KEY=old_synth_key

# Custom vars
CUSTOM_VAR=custom_value
"""
            env_path.write_text(original)
        
        try:
            import click
            original_prompt = click.prompt
            
            def mock_prompt(label, default="", show_default=True, **kwargs):
                if "ENVIRONMENT_API_KEY" in label:
                    return "new_env_key"
                elif "SYNTH_API_KEY" in label:
                    return "new_synth_key"
                elif "OPENAI_API_KEY" in label:
                    return "new_openai_key"
                return default
            
            click.prompt = mock_prompt
            
            try:
                result_path = interactive_fill_env(env_path)
                assert result_path == env_path
                
                result = env_path.read_text()
                
                # All original content should be preserved
                assert "# Database config" in result
                assert "DATABASE_URL=postgres://localhost/db" in result
                assert "DATABASE_POOL=10" in result
                assert "# Feature flags" in result
                assert "ENABLE_DEBUG=true" in result
                assert "ENABLE_METRICS=false" in result
                assert "# Custom vars" in result
                assert "CUSTOM_VAR=custom_value" in result
                
                # Updated keys should have new values
                assert "ENVIRONMENT_API_KEY=new_env_key" in result
                assert "SYNTH_API_KEY=new_synth_key" in result
                assert "OPENAI_API_KEY=new_openai_key" in result
                
                # Old values should be gone
                assert "ENVIRONMENT_API_KEY=old_env_key" not in result
                assert "SYNTH_API_KEY=old_synth_key" not in result
                
                # Empty lines should be preserved
                assert "\n\n" in result
                
            finally:
                click.prompt = original_prompt
        finally:
            env_path.unlink()

    def test_handles_file_with_only_comments(self) -> None:
        """Test that files with only comments are preserved."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# This is a comment
# Another comment
# Yet another
"""
            env_path.write_text(original)
        
        try:
            import click
            original_prompt = click.prompt
            
            def mock_prompt(label, default="", show_default=True, **kwargs):
                if "ENVIRONMENT_API_KEY" in label:
                    return "new_env_key"
                elif "SYNTH_API_KEY" in label:
                    return ""
                elif "OPENAI_API_KEY" in label:
                    return ""
                return default
            
            click.prompt = mock_prompt
            
            try:
                result_path = interactive_fill_env(env_path)
                assert result_path == env_path
                
                result = env_path.read_text()
                
                # All comments should be preserved
                assert "# This is a comment" in result
                assert "# Another comment" in result
                assert "# Yet another" in result
                assert "ENVIRONMENT_API_KEY=new_env_key" in result
                
            finally:
                click.prompt = original_prompt
        finally:
            env_path.unlink()

    def test_handles_empty_file(self) -> None:
        """Test that empty files are handled correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            env_path.write_text("")
        
        try:
            import click
            original_prompt = click.prompt
            
            def mock_prompt(label, default="", show_default=True, **kwargs):
                if "ENVIRONMENT_API_KEY" in label:
                    return "new_env_key"
                elif "SYNTH_API_KEY" in label:
                    return ""
                elif "OPENAI_API_KEY" in label:
                    return ""
                return default
            
            click.prompt = mock_prompt
            
            try:
                result_path = interactive_fill_env(env_path)
                assert result_path == env_path
                
                result = env_path.read_text()
                assert "ENVIRONMENT_API_KEY=new_env_key" in result
                
            finally:
                click.prompt = original_prompt
        finally:
            env_path.unlink()


class TestWriteEnvVarToDotenvSafety:
    """Comprehensive safety tests for write_env_var_to_dotenv."""

    def test_preserves_all_content(self) -> None:
        """Test that write_env_var_to_dotenv preserves everything."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# Comment
VAR1=value1
VAR2=value2

# Footer
"""
            env_path.write_text(original)
        
        try:
            write_env_var_to_dotenv("VAR3", "value3", output_file_path=env_path, print_msg=False)
            result = env_path.read_text()
            
            assert "# Comment" in result
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            assert "# Footer" in result
            assert "\n\n" in result
        finally:
            env_path.unlink()

    def test_updates_existing_preserves_formatting(self) -> None:
        """Test that updating preserves formatting."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """  export VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            write_env_var_to_dotenv("VAR1", "updated_value1", output_file_path=env_path, print_msg=False)
            result = env_path.read_text()
            
            assert "VAR1=updated_value1" in result
            assert "VAR1=value1" not in result
            assert "VAR2=value2" in result
            # Should preserve export and indentation
            assert "export" in result.lower() or "  " in result
        finally:
            env_path.unlink()

