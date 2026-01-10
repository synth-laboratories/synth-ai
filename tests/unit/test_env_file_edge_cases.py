"""Edge case tests to ensure .env file writing is bulletproof."""

import tempfile
from pathlib import Path

from synth_ai.cli.lib.task_app_env import interactive_fill_env, save_to_env_file
from synth_ai.core.env_utils import write_env_var_to_dotenv
from synth_ai.sdk.api.train.utils import write_env_value


class TestEdgeCases:
    """Edge cases and corner cases for .env file safety."""

    def test_preserves_mixed_content_complex_formatting(self) -> None:
        """Test complex real-world .env file formatting."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """# Production Environment
# Generated: 2024-01-01

# Database
DATABASE_URL=postgres://user:pass@localhost:5432/db
DATABASE_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys
SYNTH_API_KEY=sk_prod_12345
ENVIRONMENT_API_KEY=env_prod_67890

# Feature Flags
ENABLE_FEATURE_X=true
ENABLE_FEATURE_Y=false

# Custom Config
CUSTOM_SETTING=some_value
ANOTHER_SETTING=another_value

# End of config
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "NEW_VAR", "new_value")
            result = env_path.read_text()

            # Verify EVERYTHING is preserved
            assert "# Production Environment" in result
            assert "# Generated: 2024-01-01" in result
            assert "# Database" in result
            assert "DATABASE_URL=postgres://user:pass@localhost:5432/db" in result
            assert "DATABASE_POOL_SIZE=10" in result
            assert "# Redis" in result
            assert "REDIS_URL=redis://localhost:6379/0" in result
            assert "# API Keys" in result
            assert "SYNTH_API_KEY=sk_prod_12345" in result
            assert "ENVIRONMENT_API_KEY=env_prod_67890" in result
            assert "# Feature Flags" in result
            assert "ENABLE_FEATURE_X=true" in result
            assert "ENABLE_FEATURE_Y=false" in result
            assert "# Custom Config" in result
            assert "CUSTOM_SETTING=some_value" in result
            assert "ANOTHER_SETTING=another_value" in result
            assert "# End of config" in result
            assert "NEW_VAR=new_value" in result

            # Verify structure (multiple empty lines)
            assert result.count("\n\n") >= 3

        finally:
            env_path.unlink()

    def test_preserves_leading_trailing_whitespace_on_file(self) -> None:
        """Test that leading/trailing whitespace on file is preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """
VAR1=value1
VAR2=value2

"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR3", "value3")
            result = env_path.read_text()

            # Should preserve structure
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_handles_file_with_only_newline(self) -> None:
        """Test file with only a newline character."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            env_path.write_text("\n")

        try:
            write_env_value(env_path, "VAR1", "value1")
            result = env_path.read_text()

            assert "VAR1=value1" in result
            assert result.endswith("\n")
        finally:
            env_path.unlink()

    def test_preserves_indentation_and_spacing(self) -> None:
        """Test that indentation and spacing are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """    VAR1=value1
  VAR2=value2
VAR3=value3
    # Indented comment
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()

            # Should preserve indentation
            assert "    VAR1=value1" in result
            assert "  VAR2=value2" in result
            assert "VAR3=value3" in result
            assert "    # Indented comment" in result
        finally:
            env_path.unlink()

    def test_handles_variables_with_spaces_around_equals(self) -> None:
        """Test variables with spaces around = sign."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """VAR1 = value1
VAR2=value2
VAR3 =value3
VAR4= value4
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR5", "value5")
            result = env_path.read_text()

            # Should preserve original formatting
            assert "VAR1 = value1" in result or "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3 =value3" in result or "VAR3=value3" in result
            assert "VAR4= value4" in result or "VAR4=value4" in result
        finally:
            env_path.unlink()

    def test_preserves_backslash_escapes(self) -> None:
        """Test that backslash escapes are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """VAR1=value\\nwith\\tnewline
VAR2=value\\"with\\"quotes
VAR3=normal_value
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()

            assert "VAR1=value" in result
            assert "VAR2=value" in result
            assert "VAR3=normal_value" in result
        finally:
            env_path.unlink()

    def test_handles_very_long_lines(self) -> None:
        """Test files with very long lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            long_value = "x" * 10000
            original = f"""VAR1={long_value}
VAR2=short
VAR3={long_value}
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()

            assert f"VAR1={long_value}" in result
            assert "VAR2=short" in result
            assert f"VAR3={long_value}" in result
            assert "VAR4=value4" in result
        finally:
            env_path.unlink()

    def test_preserves_multiline_comments(self) -> None:
        """Test that multi-line comment blocks are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """# This is a
# multi-line comment
# block

VAR1=value1

# Another
# multi-line
# comment

VAR2=value2
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR3", "value3")
            result = env_path.read_text()

            # All comment lines should be preserved
            assert "# This is a" in result
            assert "# multi-line comment" in result
            assert "# block" in result
            assert "# Another" in result
            assert "# multi-line" in result
            assert "# comment" in result
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
        finally:
            env_path.unlink()

    def test_handles_special_regex_characters_in_values(self) -> None:
        """Test that regex special characters in values don't break parsing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """VAR1=value.with.dots
VAR2=value*with*stars
VAR3=value+with+plus
VAR4=value?with?question
VAR5=value[with]brackets
VAR6=value{with}braces
VAR7=value(with)parens
VAR8=value^with^caret
VAR9=value$with$dollar
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR10", "value10")
            result = env_path.read_text()

            # All vars should be preserved
            for i in range(1, 10):
                assert f"VAR{i}=" in result
        finally:
            env_path.unlink()

    def test_preserves_case_sensitivity(self) -> None:
        """Test that case sensitivity is preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """VAR1=value1
var1=lowercase_value
VAR_1=value_with_underscore
Var1=MixedCase
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR2", "value2")
            result = env_path.read_text()

            # All should be preserved (they're different keys)
            assert "VAR1=value1" in result
            assert "var1=lowercase_value" in result
            assert "VAR_1=value_with_underscore" in result
            assert "Var1=MixedCase" in result
        finally:
            env_path.unlink()

    def test_handles_file_with_bom(self) -> None:
        """Test file with UTF-8 BOM."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            content = "VAR1=value1\nVAR2=value2\n".encode("utf-8-sig")  # Adds BOM
            f.write(content)

        try:
            write_env_value(env_path, "VAR3", "value3")
            result = env_path.read_text(encoding="utf-8-sig")

            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_preserves_tabs_vs_spaces(self) -> None:
        """Test that tabs vs spaces are preserved."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """\tVAR1=value1
    VAR2=value2
VAR3=value3
"""
            env_path.write_text(original)

        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()

            # Should preserve tabs and spaces
            assert "\tVAR1=value1" in result or "VAR1=value1" in result
            assert "    VAR2=value2" in result or "VAR2=value2" in result
            assert "VAR3=value3" in result
        finally:
            env_path.unlink()

    def test_interactive_fill_preserves_complex_real_world_file(self) -> None:
        """Test interactive_fill_env with a complex real-world .env file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """# Synth AI Configuration
# This file contains API keys and configuration

# Required: Environment API Key
ENVIRONMENT_API_KEY=old_env_key_12345

# Optional: Synth API Key
SYNTH_API_KEY=old_synth_key_67890

# Database Configuration
DATABASE_URL=postgres://localhost:5432/mydb
DATABASE_POOL_SIZE=20
DATABASE_TIMEOUT=30

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=secret_password

# Feature Flags
ENABLE_DEBUG_MODE=true
ENABLE_METRICS=false
ENABLE_TRACING=true

# Custom Application Settings
APP_NAME=MyApp
APP_VERSION=1.0.0
LOG_LEVEL=INFO

# External Service URLs
EXTERNAL_API_URL=https://api.example.com
WEBHOOK_URL=https://webhook.example.com

# Security Settings
SESSION_TIMEOUT=3600
MAX_RETRIES=3

# End of configuration
"""
            env_path.write_text(original)

        try:
            import click

            original_prompt = click.prompt

            def mock_prompt(label, default="", show_default=True, **kwargs):
                if "ENVIRONMENT_API_KEY" in label:
                    return "new_env_key_99999"
                elif "SYNTH_API_KEY" in label:
                    return "new_synth_key_11111"
                elif "OPENAI_API_KEY" in label:
                    return "new_openai_key_22222"
                return default

            click.prompt = mock_prompt

            try:
                result_path = interactive_fill_env(env_path)
                assert result_path == env_path

                result = env_path.read_text()

                # Verify ALL original content is preserved
                assert "# Synth AI Configuration" in result
                assert "# This file contains API keys and configuration" in result
                assert "# Required: Environment API Key" in result
                assert "# Optional: Synth API Key" in result
                assert "# Database Configuration" in result
                assert "DATABASE_URL=postgres://localhost:5432/mydb" in result
                assert "DATABASE_POOL_SIZE=20" in result
                assert "DATABASE_TIMEOUT=30" in result
                assert "# Redis Configuration" in result
                assert "REDIS_URL=redis://localhost:6379/0" in result
                assert "REDIS_PASSWORD=secret_password" in result
                assert "# Feature Flags" in result
                assert "ENABLE_DEBUG_MODE=true" in result
                assert "ENABLE_METRICS=false" in result
                assert "ENABLE_TRACING=true" in result
                assert "# Custom Application Settings" in result
                assert "APP_NAME=MyApp" in result
                assert "APP_VERSION=1.0.0" in result
                assert "LOG_LEVEL=INFO" in result
                assert "# External Service URLs" in result
                assert "EXTERNAL_API_URL=https://api.example.com" in result
                assert "WEBHOOK_URL=https://webhook.example.com" in result
                assert "# Security Settings" in result
                assert "SESSION_TIMEOUT=3600" in result
                assert "MAX_RETRIES=3" in result
                assert "# End of configuration" in result

                # Verify keys were updated
                assert "ENVIRONMENT_API_KEY=new_env_key_99999" in result
                assert "SYNTH_API_KEY=new_synth_key_11111" in result
                assert "OPENAI_API_KEY=new_openai_key_22222" in result

                # Verify old values are gone
                assert "ENVIRONMENT_API_KEY=old_env_key_12345" not in result
                assert "SYNTH_API_KEY=old_synth_key_67890" not in result

                # Count total variables - should have all original + 1 new
                var_lines = [
                    line
                    for line in result.splitlines()
                    if "=" in line and not line.strip().startswith("#")
                ]
                assert len(var_lines) >= 15, (
                    f"Lost variables! Expected at least 15, got {len(var_lines)}"
                )

            finally:
                click.prompt = original_prompt
        finally:
            env_path.unlink()

    def test_save_to_env_file_preserves_complex_file(self) -> None:
        """Test save_to_env_file with complex file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """# Config file
VAR1=value1

VAR2=value2
VAR3=value3

# More vars
VAR4=value4
"""
            env_path.write_text(original)

        try:
            save_to_env_file(env_path, "VAR1", "updated_value1")
            result = env_path.read_text()

            assert "# Config file" in result
            assert "VAR1=updated_value1" in result
            assert "VAR1=value1" not in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            assert "# More vars" in result
            assert "VAR4=value4" in result
            assert "\n\n" in result  # Empty lines preserved
        finally:
            env_path.unlink()

    def test_write_env_var_to_dotenv_preserves_everything(self) -> None:
        """Test write_env_var_to_dotenv preserves everything."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
            env_path = Path(f.name)
            original = """# Header
VAR1=value1

VAR2=value2
# Footer
"""
            env_path.write_text(original)

        try:
            write_env_var_to_dotenv("VAR3", "value3", output_file_path=env_path, print_msg=False)
            result = env_path.read_text()

            assert "# Header" in result
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            assert "# Footer" in result
            assert "\n\n" in result
        finally:
            env_path.unlink()

    def test_all_functions_preserve_empty_file(self) -> None:
        """Test that all functions handle empty files correctly."""
        functions = [
            lambda p, k, v: write_env_value(p, k, v),
            lambda p, k, v: save_to_env_file(p, k, v),
            lambda p, k, v: write_env_var_to_dotenv(k, v, output_file_path=p, print_msg=False),
        ]

        for func in functions:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
                env_path = Path(f.name)
                env_path.write_text("")

            try:
                func(env_path, "TEST_VAR", "test_value")
                result = env_path.read_text()

                assert "TEST_VAR=test_value" in result
                assert result.endswith("\n")
            finally:
                env_path.unlink()

    def test_all_functions_preserve_file_with_only_comments(self) -> None:
        """Test that all functions preserve files with only comments."""
        functions = [
            lambda p, k, v: write_env_value(p, k, v),
            lambda p, k, v: save_to_env_file(p, k, v),
            lambda p, k, v: write_env_var_to_dotenv(k, v, output_file_path=p, print_msg=False),
        ]

        for func in functions:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
                env_path = Path(f.name)
                env_path.write_text("# Comment 1\n# Comment 2\n")

            try:
                func(env_path, "TEST_VAR", "test_value")
                result = env_path.read_text()

                assert "# Comment 1" in result
                assert "# Comment 2" in result
                assert "TEST_VAR=test_value" in result
            finally:
                env_path.unlink()
