"""Stress tests and sequential operation tests for .env file safety."""

import tempfile
from pathlib import Path

import pytest

from synth_ai.sdk.api.train.utils import write_env_value
from synth_ai.cli.lib.task_app_env import save_to_env_file
from synth_ai.utils.env import write_env_var_to_dotenv


class TestSequentialOperations:
    """Test that multiple sequential operations preserve content."""

    def test_multiple_writes_preserve_all_content(self) -> None:
        """Test that multiple writes don't destroy content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            # Write multiple variables sequentially
            write_env_value(env_path, "VAR4", "value4")
            write_env_value(env_path, "VAR5", "value5")
            write_env_value(env_path, "VAR6", "value6")
            
            result = env_path.read_text()
            
            # All original vars should still be there
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            
            # All new vars should be there
            assert "VAR4=value4" in result
            assert "VAR5=value5" in result
            assert "VAR6=value6" in result
            
            # Count total vars
            var_count = len([l for l in result.splitlines() if "=" in l and not l.strip().startswith("#")])
            assert var_count == 6, f"Expected 6 variables, got {var_count}"
            
        finally:
            env_path.unlink()

    def test_multiple_updates_preserve_other_vars(self) -> None:
        """Test that updating multiple vars preserves others."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
VAR3=value3
VAR4=value4
"""
            env_path.write_text(original)
        
        try:
            # Update multiple vars
            write_env_value(env_path, "VAR1", "updated1")
            write_env_value(env_path, "VAR3", "updated3")
            
            result = env_path.read_text()
            
            assert "VAR1=updated1" in result
            assert "VAR1=value1" not in result
            assert "VAR2=value2" in result  # Should be unchanged
            assert "VAR3=updated3" in result
            assert "VAR3=value3" not in result
            assert "VAR4=value4" in result  # Should be unchanged
            
        finally:
            env_path.unlink()

    def test_mixed_operations_preserve_content(self) -> None:
        """Test mixing updates and new writes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            # Mix of updates and new writes
            write_env_value(env_path, "VAR1", "updated1")  # Update
            write_env_value(env_path, "VAR3", "value3")    # New
            write_env_value(env_path, "VAR2", "updated2")  # Update
            write_env_value(env_path, "VAR4", "value4")    # New
            
            result = env_path.read_text()
            
            assert "VAR1=updated1" in result
            assert "VAR2=updated2" in result
            assert "VAR3=value3" in result
            assert "VAR4=value4" in result
            
            # Old values should be gone
            assert "VAR1=value1" not in result
            assert "VAR2=value2" not in result
            
        finally:
            env_path.unlink()

    def test_save_to_env_file_multiple_operations(self) -> None:
        """Test multiple save_to_env_file operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            save_to_env_file(env_path, "VAR1", "updated1")
            save_to_env_file(env_path, "VAR3", "value3")
            save_to_env_file(env_path, "VAR2", "updated2")
            
            result = env_path.read_text()
            
            assert "VAR1=updated1" in result
            assert "VAR2=updated2" in result
            assert "VAR3=value3" in result
            
        finally:
            env_path.unlink()

    def test_write_env_var_to_dotenv_multiple_operations(self) -> None:
        """Test multiple write_env_var_to_dotenv operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
"""
            env_path.write_text(original)
        
        try:
            write_env_var_to_dotenv("VAR1", "updated1", output_file_path=env_path, print_msg=False)
            write_env_var_to_dotenv("VAR3", "value3", output_file_path=env_path, print_msg=False)
            write_env_var_to_dotenv("VAR2", "updated2", output_file_path=env_path, print_msg=False)
            
            result = env_path.read_text()
            
            assert "VAR1=updated1" in result
            assert "VAR2=updated2" in result
            assert "VAR3=value3" in result
            
        finally:
            env_path.unlink()


class TestLargeFileSafety:
    """Test that large files are handled safely."""

    def test_preserves_large_file_with_many_vars(self) -> None:
        """Test file with many variables."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            # Create file with 500 variables
            lines = []
            for i in range(500):
                lines.append(f"VAR{i}=value{i}\n")
            original = "".join(lines)
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR_NEW", "new_value")
            result = env_path.read_text()
            
            # Check that all original vars are preserved
            for i in range(500):
                assert f"VAR{i}=value{i}" in result, f"Lost VAR{i}!"
            
            assert "VAR_NEW=new_value" in result
            
            # Count vars
            var_lines = [l for l in result.splitlines() if "=" in l]
            assert len(var_lines) == 501, f"Expected 501 variables, got {len(var_lines)}"
            
        finally:
            env_path.unlink()

    def test_preserves_large_file_with_many_comments(self) -> None:
        """Test file with many comments."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            lines = []
            for i in range(100):
                lines.append(f"# Comment {i}\n")
                lines.append(f"VAR{i}=value{i}\n")
            original = "".join(lines)
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR_NEW", "new_value")
            result = env_path.read_text()
            
            # Check comments are preserved
            for i in range(100):
                assert f"# Comment {i}" in result, f"Lost comment {i}!"
                assert f"VAR{i}=value{i}" in result, f"Lost VAR{i}!"
            
        finally:
            env_path.unlink()


class TestContentIntegrity:
    """Test that content integrity is maintained."""

    def test_exact_byte_preservation_for_non_matching_lines(self) -> None:
        """Test that lines not matching the pattern are preserved exactly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
This line doesn't match any pattern
VAR2=value2
  This line has leading spaces and doesn't match
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            write_env_value(env_path, "VAR4", "value4")
            result = env_path.read_text()
            
            # Non-matching lines should be preserved exactly
            assert "This line doesn't match any pattern" in result
            assert "  This line has leading spaces and doesn't match" in result
            
            # Matching vars should be preserved
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            
        finally:
            env_path.unlink()

    def test_no_data_loss_on_repeated_writes(self) -> None:
        """Test that repeated writes don't cause data loss."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
VAR3=value3
VAR4=value4
VAR5=value5
"""
            env_path.write_text(original)
        
        try:
            # Write the same var multiple times
            for _ in range(10):
                write_env_value(env_path, "VAR1", "value1")  # Same value
            
            result = env_path.read_text()
            
            # All vars should still be there
            assert "VAR1=value1" in result
            assert "VAR2=value2" in result
            assert "VAR3=value3" in result
            assert "VAR4=value4" in result
            assert "VAR5=value5" in result
            
            # Should only have one VAR1
            var1_count = result.count("VAR1=value1")
            assert var1_count == 1, f"VAR1 appears {var1_count} times, should be 1"
            
        finally:
            env_path.unlink()

    def test_preserves_file_structure_exactly(self) -> None:
        """Test that file structure is preserved exactly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """# Section 1
VAR1=value1

VAR2=value2

# Section 2
VAR3=value3

VAR4=value4

# Section 3
VAR5=value5
"""
            env_path.write_text(original)
            original_structure = original.splitlines()
        
        try:
            write_env_value(env_path, "VAR6", "value6")
            result = env_path.read_text()
            result_structure = result.splitlines()
            
            # Should preserve the general structure
            # Count sections
            assert result.count("# Section") == 3
            
            # Count empty lines (allowing for one difference)
            original_empty = original.count("\n\n")
            result_empty = result.count("\n\n")
            assert abs(result_empty - original_empty) <= 1
            
            # All vars should be there
            for i in range(1, 6):
                assert f"VAR{i}=value{i}" in result
            assert "VAR6=value6" in result
            
        finally:
            env_path.unlink()


class TestCrossFunctionSafety:
    """Test that different functions work together safely."""

    def test_mixing_functions_preserves_content(self) -> None:
        """Test that mixing different write functions preserves content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
            env_path = Path(f.name)
            original = """VAR1=value1
VAR2=value2
VAR3=value3
"""
            env_path.write_text(original)
        
        try:
            # Mix different functions
            write_env_value(env_path, "VAR1", "updated1")
            save_to_env_file(env_path, "VAR2", "updated2")
            write_env_var_to_dotenv("VAR4", "value4", output_file_path=env_path, print_msg=False)
            write_env_value(env_path, "VAR3", "updated3")
            
            result = env_path.read_text()
            
            assert "VAR1=updated1" in result
            assert "VAR2=updated2" in result
            assert "VAR3=updated3" in result
            assert "VAR4=value4" in result
            
            # Old values should be gone
            assert "VAR1=value1" not in result
            assert "VAR2=value2" not in result
            assert "VAR3=value3" not in result
            
        finally:
            env_path.unlink()

