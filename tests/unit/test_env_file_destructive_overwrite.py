"""Test that demonstrates the CRITICAL bug where .env files are completely overwritten."""

import tempfile
from pathlib import Path

import pytest

from synth_ai.cli.lib.task_app_env import interactive_fill_env


def test_interactive_fill_env_destroys_existing_content() -> None:
    """
    CRITICAL BUG TEST: interactive_fill_env completely overwrites .env file,
    destroying all comments, other variables, and formatting.
    
    This is EXTREMELY DANGEROUS and will make users very pissed.
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
        # Create a .env file with multiple variables, comments, and formatting
        original_content = """# API Configuration
SYNTH_API_KEY=sk_existing_12345
ENVIRONMENT_API_KEY=env_existing_67890

# Database config
DATABASE_URL=postgres://localhost/db
DATABASE_POOL_SIZE=10

# Feature flags
ENABLE_DEBUG=true
ENABLE_METRICS=false

# Custom user variables
CUSTOM_VAR=custom_value
# End of file comment
"""
        env_path.write_text(original_content)
        original_line_count = original_content.count('\n')
        original_vars = [
            "DATABASE_URL",
            "DATABASE_POOL_SIZE", 
            "ENABLE_DEBUG",
            "ENABLE_METRICS",
            "CUSTOM_VAR"
        ]
    
    try:
        # Simulate interactive_fill_env being called
        # It reads existing values, but then OVERWRITES the entire file
        # with only ENVIRONMENT_API_KEY, SYNTH_API_KEY, OPENAI_API_KEY
        
        # Mock the interactive prompts to return values
        import synth_ai.cli.lib.task_app_env as task_app_env_module
        original_prompt = None
        try:
            import click
            original_prompt = click.prompt
            
            def mock_prompt(label, default="", show_default=True, **kwargs):
                # Return the existing value or a new one
                if "ENVIRONMENT_API_KEY" in label:
                    return "env_new_99999"
                elif "SYNTH_API_KEY" in label:
                    return "sk_new_11111"
                elif "OPENAI_API_KEY" in label:
                    return "openai_new_22222"
                return default
            
            click.prompt = mock_prompt
            
            # Call the function - this will DESTROY the file
            result_path = interactive_fill_env(env_path)
            
        finally:
            if original_prompt:
                click.prompt = original_prompt
        
        assert result_path == env_path
        
        # Read the destroyed file
        result_content = env_path.read_text()
        result_line_count = result_content.count('\n')
        
        # CRITICAL ASSERTIONS - these will FAIL due to the bug
        
        # 1. All other variables should be preserved
        for var in original_vars:
            assert var in result_content, f"CRITICAL: Lost variable {var}! File was completely overwritten!"
        
        # 2. Comments should be preserved
        assert "# API Configuration" in result_content or "# Database config" in result_content, \
            "CRITICAL: Lost all comments! File was completely overwritten!"
        
        # 3. Empty lines should be preserved
        assert "\n\n" in result_content, \
            "CRITICAL: Lost empty lines! File was completely overwritten!"
        
        # 4. File should have more than just 3 variables
        var_count = len([line for line in result_content.splitlines() if "=" in line and not line.strip().startswith("#")])
        assert var_count > 3, \
            f"CRITICAL: File only has {var_count} variables! Should have {len(original_vars) + 3}. File was completely overwritten!"
        
        # 5. Updated variables should have new values
        assert "ENVIRONMENT_API_KEY=env_new_99999" in result_content
        assert "SYNTH_API_KEY=sk_new_11111" in result_content
        
    finally:
        env_path.unlink()

