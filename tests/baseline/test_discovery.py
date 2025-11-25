"""Unit tests for baseline file discovery."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from synth_ai.sdk.baseline.discovery import (
    BaselineConfigVisitor,
    discover_baseline_files,
    load_baseline_config_from_file,
    should_ignore_path,
)


class TestBaselineConfigVisitor:
    """Test AST visitor for BaselineConfig detection."""
    
    def test_finds_simple_baseline_config(self):
        """Test visitor finds simple BaselineConfig assignment."""
        code = """
from synth_ai.sdk.baseline import BaselineConfig, DataSplit, TaskResult

async def run_task(seed, policy_config, env_config):
    return TaskResult(seed=seed, success=True, outcome_reward=1.0)

my_baseline = BaselineConfig(
    baseline_id="test-baseline",
    name="Test Baseline",
    task_runner=run_task,
    splits={"train": DataSplit(name="train", seeds=list(range(10)))},
)
"""
        tree = ast.parse(code)
        visitor = BaselineConfigVisitor()
        visitor.visit(tree)
        
        assert len(visitor.matches) == 1
        assert visitor.matches[0][0] == "test-baseline"
    
    def test_finds_multiple_baselines(self):
        """Test visitor finds multiple BaselineConfig instances."""
        code = """
from synth_ai.sdk.baseline import BaselineConfig

baseline1 = BaselineConfig(baseline_id="first", name="First", task_runner=lambda: None, splits={})
baseline2 = BaselineConfig(baseline_id="second", name="Second", task_runner=lambda: None, splits={})
"""
        tree = ast.parse(code)
        visitor = BaselineConfigVisitor()
        visitor.visit(tree)
        
        assert len(visitor.matches) == 2
        assert visitor.matches[0][0] == "first"
        assert visitor.matches[1][0] == "second"
    
    def test_ignores_non_baseline_code(self):
        """Test visitor ignores unrelated code."""
        code = """
my_config = SomeOtherConfig(id="test")
regular_variable = "hello"
"""
        tree = ast.parse(code)
        visitor = BaselineConfigVisitor()
        visitor.visit(tree)
        
        assert len(visitor.matches) == 0


class TestDiscoverBaselineFiles:
    """Test baseline file discovery."""
    
    def test_discovers_files_in_baseline_directory(self, tmp_path):
        """Test discovery finds files in baseline/ directory."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        
        # Create a baseline file
        baseline_file = baseline_dir / "test_baseline.py"
        baseline_file.write_text("""
from synth_ai.sdk.baseline import BaselineConfig, DataSplit

async def task_runner(seed, policy_config, env_config):
    from synth_ai.sdk.baseline import TaskResult
    return TaskResult(seed=seed, success=True, outcome_reward=1.0)

test_baseline = BaselineConfig(
    baseline_id="test",
    name="Test",
    task_runner=task_runner,
    splits={},
)
""")
        
        results = discover_baseline_files([tmp_path])
        
        assert len(results) == 1
        assert results[0].baseline_id == "test"
        assert results[0].path == baseline_file.resolve()
    
    def test_discovers_files_with_baseline_suffix(self, tmp_path):
        """Test discovery finds *_baseline.py files."""
        baseline_file = tmp_path / "crafter_baseline.py"
        baseline_file.write_text("""
from synth_ai.sdk.baseline import BaselineConfig

async def task_runner(seed, policy_config, env_config):
    from synth_ai.sdk.baseline import TaskResult
    return TaskResult(seed=seed, success=True, outcome_reward=1.0)

crafter = BaselineConfig(
    baseline_id="crafter",
    name="Crafter",
    task_runner=task_runner,
    splits={},
)
""")
        
        results = discover_baseline_files([tmp_path])
        
        assert len(results) == 1
        assert results[0].baseline_id == "crafter"
    
    def test_ignores_pycache_directories(self, tmp_path):
        """Test discovery ignores __pycache__."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        
        (pycache / "baseline.py").write_text("""
from synth_ai.sdk.baseline import BaselineConfig
baseline = BaselineConfig(baseline_id="test", name="Test", task_runner=lambda: None, splits={})
""")
        
        results = discover_baseline_files([tmp_path])
        
        assert len(results) == 0
    
    def test_ignores_git_directories(self, tmp_path):
        """Test discovery ignores .git directories."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        
        (git_dir / "baseline.py").write_text("""
from synth_ai.sdk.baseline import BaselineConfig
baseline = BaselineConfig(baseline_id="test", name="Test", task_runner=lambda: None, splits={})
""")
        
        results = discover_baseline_files([tmp_path])
        
        assert len(results) == 0
    
    def test_handles_syntax_errors(self, tmp_path):
        """Test discovery skips files with syntax errors."""
        baseline_file = tmp_path / "broken_baseline.py"
        baseline_file.write_text("this is not valid python {{{")
        
        results = discover_baseline_files([tmp_path])
        
        # Should not crash, just skip the file
        assert len(results) == 0
    
    def test_deduplicates_baselines(self, tmp_path):
        """Test discovery deduplicates same baseline in same file."""
        baseline_file = tmp_path / "duplicate_baseline.py"
        baseline_file.write_text("""
from synth_ai.sdk.baseline import BaselineConfig

async def task_runner(seed, policy_config, env_config):
    from synth_ai.sdk.baseline import TaskResult
    return TaskResult(seed=seed, success=True, outcome_reward=1.0)

baseline1 = BaselineConfig(baseline_id="dup", name="Dup", task_runner=task_runner, splits={})
baseline2 = BaselineConfig(baseline_id="dup", name="Dup", task_runner=task_runner, splits={})
""")
        
        results = discover_baseline_files([tmp_path])
        
        # Should only return one (deduplicated by baseline_id + path)
        baseline_ids = [r.baseline_id for r in results]
        assert len(set(baseline_ids)) == 1


class TestShouldIgnorePath:
    """Test path filtering logic."""
    
    def test_ignores_pycache(self):
        assert should_ignore_path(Path("some/path/__pycache__/file.py"))
    
    def test_ignores_venv(self):
        assert should_ignore_path(Path("some/.venv/lib/python3.10/file.py"))
    
    def test_ignores_git(self):
        assert should_ignore_path(Path(".git/hooks/pre-commit"))
    
    def test_allows_normal_paths(self):
        assert not should_ignore_path(Path("baseline/crafter_baseline.py"))


class TestLoadBaselineConfigFromFile:
    """Test loading baseline configs from files."""
    
    def test_loads_valid_baseline(self, tmp_path):
        """Test loading a valid baseline config."""
        baseline_file = tmp_path / "test_baseline.py"
        baseline_file.write_text("""
from synth_ai.sdk.baseline import BaselineConfig, DataSplit, TaskResult

async def task_runner(seed, policy_config, env_config):
    return TaskResult(seed=seed, success=True, outcome_reward=1.0)

test_baseline = BaselineConfig(
    baseline_id="test",
    name="Test Baseline",
    task_runner=task_runner,
    splits={
        "train": DataSplit(name="train", seeds=list(range(10))),
    },
)
""")
        
        config = load_baseline_config_from_file("test", baseline_file)
        
        assert config.baseline_id == "test"
        assert config.name == "Test Baseline"
        assert "train" in config.splits
        assert len(config.splits["train"].seeds) == 10
    
    def test_raises_on_missing_baseline_id(self, tmp_path):
        """Test error when baseline_id not found."""
        baseline_file = tmp_path / "test_baseline.py"
        baseline_file.write_text("# Empty file")
        
        with pytest.raises(ValueError, match="Baseline 'nonexistent' not found|No BaselineConfig instances found"):
            load_baseline_config_from_file("nonexistent", baseline_file)

