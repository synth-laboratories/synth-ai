"""Unit tests to prevent regressions in GEPA TOML pattern configuration.

This test suite ensures that:
1. TOML patterns use single braces {wildcard} not double braces {{wildcard}}
2. TOML patterns use actual newlines \n not literal \\n
3. Patterns match actual task app message formats
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class TestGEPATOMLPatternRegressions:
    """Test suite to prevent TOML pattern configuration regressions."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Get the repository root directory."""
        # tests/unit/api/train/test_gepa_toml_pattern_regressions.py
        # Go up 5 levels: train -> api -> unit -> tests -> repo_root
        return Path(__file__).parent.parent.parent.parent.parent

    @pytest.fixture
    def heartdisease_config_path(self, repo_root: Path) -> Path:
        """Path to heartdisease GEPA config."""
        return repo_root / "examples" / "blog_posts" / "langprobe" / "task_specific" / "heartdisease" / "heartdisease_gepa.toml"

    @pytest.fixture
    def banking77_config_path(self, repo_root: Path) -> Path:
        """Path to banking77 GEPA config."""
        return repo_root / "examples" / "blog_posts" / "langprobe" / "task_specific" / "banking77" / "banking77_gepa.toml"

    def test_heartdisease_toml_uses_single_braces(self, heartdisease_config_path: Path) -> None:
        """Verify heartdisease TOML uses single braces {wildcard} not double {{wildcard}}."""
        with open(heartdisease_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        for msg in messages:
            pattern = msg.get("pattern", "")
            # Check for double braces (regression)
            double_braces = re.findall(r'\{\{(\w+)\}\}', pattern)
            assert not double_braces, (
                f"Found double braces {{}} in pattern (should use single braces {{}}):\n"
                f"Pattern: {pattern!r}\n"
                f"Double braces found: {double_braces}"
            )

            # Verify single braces are used for wildcards
            single_braces = re.findall(r'\{(\w+)\}', pattern)
            if single_braces:
                # All wildcards should use single braces
                assert all(f"{{{name}}}" in pattern for name in single_braces), (
                    f"Pattern should use single braces for wildcards:\n"
                    f"Pattern: {pattern!r}\n"
                    f"Wildcards found: {single_braces}"
                )

    def test_banking77_toml_uses_single_braces(self, banking77_config_path: Path) -> None:
        """Verify banking77 TOML uses single braces {wildcard} not double {{wildcard}}."""
        with open(banking77_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        for msg in messages:
            pattern = msg.get("pattern", "")
            # Check for double braces (regression)
            double_braces = re.findall(r'\{\{(\w+)\}\}', pattern)
            assert not double_braces, (
                f"Found double braces {{}} in pattern (should use single braces {{}}):\n"
                f"Pattern: {pattern!r}\n"
                f"Double braces found: {double_braces}"
            )

            # Verify single braces are used for wildcards
            single_braces = re.findall(r'\{(\w+)\}', pattern)
            if single_braces:
                # All wildcards should use single braces
                assert all(f"{{{name}}}" in pattern for name in single_braces), (
                    f"Pattern should use single braces for wildcards:\n"
                    f"Pattern: {pattern!r}\n"
                    f"Wildcards found: {single_braces}"
                )

    def test_heartdisease_toml_uses_actual_newlines(self, heartdisease_config_path: Path) -> None:
        """Verify heartdisease TOML uses actual newlines \n not literal \\n."""
        with open(heartdisease_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        for msg in messages:
            pattern = msg.get("pattern", "")
            # Check for literal backslash-n (regression)
            # After TOML parsing, literal \n would appear as two characters: backslash + n
            # Actual newlines appear as single newline character
            has_literal_backslash_n = "\\n" in pattern or r"\n" in pattern
            has_actual_newline = "\n" in pattern

            if has_literal_backslash_n and not has_actual_newline:
                pytest.fail(
                    f"Pattern contains literal \\n instead of actual newline:\n"
                    f"Pattern repr: {pattern!r}\n"
                    f"Pattern should use \\n (actual newline) not \\\\n (literal backslash-n)"
                )

            # If pattern should have newlines (contains newline-like content), verify it does
            if "Patient Features" in pattern or "Classify" in pattern:
                assert has_actual_newline, (
                    f"Pattern should contain actual newlines but doesn't:\n"
                    f"Pattern repr: {pattern!r}"
                )

    def test_banking77_toml_uses_actual_newlines(self, banking77_config_path: Path) -> None:
        """Verify banking77 TOML uses actual newlines \n not literal \\n."""
        with open(banking77_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        for msg in messages:
            pattern = msg.get("pattern", "")
            # Check for literal backslash-n (regression)
            has_literal_backslash_n = "\\n" in pattern or r"\n" in pattern
            has_actual_newline = "\n" in pattern

            if has_literal_backslash_n and not has_actual_newline:
                pytest.fail(
                    f"Pattern contains literal \\n instead of actual newline:\n"
                    f"Pattern repr: {pattern!r}\n"
                    f"Pattern should use \\n (actual newline) not \\\\n (literal backslash-n)"
                )

            # If pattern should have newlines (contains newline-like content), verify it does
            if "Customer Query" in pattern or "Available Intents" in pattern:
                assert has_actual_newline, (
                    f"Pattern should contain actual newlines but doesn't:\n"
                    f"Pattern repr: {pattern!r}"
                )

    def test_heartdisease_pattern_matches_task_app_format(self, heartdisease_config_path: Path) -> None:
        """Verify heartdisease pattern matches actual task app message format."""
        with open(heartdisease_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        # Find user message pattern
        user_pattern = None
        for msg in messages:
            if msg.get("role") == "user":
                user_pattern = msg.get("pattern", "")
                break

        assert user_pattern is not None, "User message pattern not found"

        # Simulate actual task app message format
        actual_message = (
            "Patient Features:\n"
            "age: 63\n"
            "sex: 1\n"
            "cp: 1\n"
            "trestbps: 145\n"
            "chol: 233\n"
            "fbs: 1\n"
            "restecg: 2\n"
            "thalach: 150\n"
            "exang: 0\n"
            "oldpeak: 2.3\n"
            "slope: 3\n"
            "ca: 0\n"
            "thal: fixed\n"
            "\n"
            "Classify: Does this patient have heart disease? Respond with '1' for yes or '0' for no."
        )

        # Build regex pattern from TOML pattern (simulating backend pattern matching)
        # Use the same logic as backend's MessagePattern.build_regex()
        regex_pattern_parts = []
        i = 0
        while i < len(user_pattern):
            char = user_pattern[i]
            if char == '\n':
                regex_pattern_parts.append('\n')
            elif char == '\t':
                regex_pattern_parts.append('\t')
            elif char == '\r':
                regex_pattern_parts.append('\r')
            elif char == '{':
                wildcard_match = re.match(r'\{(\w+)\}', user_pattern[i:])
                if wildcard_match:
                    # Replace with capture group (last wildcard uses .+, others use .+?)
                    regex_pattern_parts.append(r'(?P<features>.+)')
                    i += len(wildcard_match.group(0)) - 1
                else:
                    regex_pattern_parts.append(re.escape(char))
            elif char == '}':
                regex_pattern_parts.append(re.escape(char))
            else:
                regex_pattern_parts.append(re.escape(char))
            i += 1

        regex_pattern = ''.join(regex_pattern_parts)

        # Compile and test match
        compiled = re.compile(regex_pattern, re.DOTALL)
        match = compiled.search(actual_message)

        assert match is not None, (
            f"Pattern does not match actual task app message format!\n"
            f"Pattern: {user_pattern!r}\n"
            f"Actual message: {actual_message[:200]}...\n"
            f"Regex pattern: {regex_pattern[:200]}..."
        )

        # Verify wildcard extraction
        if match:
            features = match.groupdict().get("features")
            assert features is not None, "Failed to extract 'features' wildcard from pattern"

    def test_banking77_pattern_matches_task_app_format(self, banking77_config_path: Path) -> None:
        """Verify banking77 pattern matches actual task app message format."""
        with open(banking77_config_path, "rb") as f:
            config = tomllib.load(f)

        pl_config = config.get("prompt_learning", {})
        initial_prompt = pl_config.get("initial_prompt", {})
        messages = initial_prompt.get("messages", [])

        # Find user message pattern
        user_pattern = None
        for msg in messages:
            if msg.get("role") == "user":
                user_pattern = msg.get("pattern", "")
                break

        assert user_pattern is not None, "User message pattern not found"

        # Simulate actual task app message format (from error message)
        actual_message = (
            "Customer Query: I am still waiting on my card?\n"
            "\n"
            "Available Intents:\n"
            "1. activate_my_card\n"
            "2. age_limit\n"
            "3. apple_pay_or_google_pay\n"
            # ... truncated for brevity, but pattern should match
            "77. wrong_exchange_rate_for_cash_withdrawal\n"
            "\n"
            "Classify this query into one of the above banking intents using the tool call."
        )

        # Build regex pattern from TOML pattern (simulating backend pattern matching)
        # Use the same logic as backend's MessagePattern.build_regex()
        regex_pattern_parts = []
        i = 0
        wildcard_positions = []
        while i < len(user_pattern):
            char = user_pattern[i]
            if char == '\n':
                regex_pattern_parts.append('\n')
            elif char == '\t':
                regex_pattern_parts.append('\t')
            elif char == '\r':
                regex_pattern_parts.append('\r')
            elif char == '{':
                wildcard_match = re.match(r'\{(\w+)\}', user_pattern[i:])
                if wildcard_match:
                    wildcard_name = wildcard_match.group(1)
                    wildcard_positions.append((len(regex_pattern_parts), wildcard_name))
                    # Placeholder, will be replaced later
                    regex_pattern_parts.append(wildcard_match.group(0))
                    i += len(wildcard_match.group(0)) - 1
                else:
                    regex_pattern_parts.append(re.escape(char))
            elif char == '}':
                regex_pattern_parts.append(re.escape(char))
            else:
                regex_pattern_parts.append(re.escape(char))
            i += 1

        # Replace wildcards with capture groups
        escaped = ''.join(regex_pattern_parts)
        regex_pattern = escaped
        for idx, (pos, wildcard_name) in enumerate(wildcard_positions):
            is_last = idx == len(wildcard_positions) - 1
            if is_last:
                replacement = f'(?P<{wildcard_name}>.+)'
            else:
                replacement = f'(?P<{wildcard_name}>.+?)'
            regex_pattern = regex_pattern.replace(f'{{{wildcard_name}}}', replacement, 1)

        # Compile and test match
        compiled = re.compile(regex_pattern, re.DOTALL)
        match = compiled.search(actual_message)

        assert match is not None, (
            f"Pattern does not match actual task app message format!\n"
            f"Pattern: {user_pattern!r}\n"
            f"Actual message: {actual_message[:200]}...\n"
            f"Regex pattern: {regex_pattern[:200]}..."
        )

        # Verify wildcard extraction
        if match:
            query = match.groupdict().get("query")
            available_intents = match.groupdict().get("available_intents")
            assert query is not None, "Failed to extract 'query' wildcard from pattern"
            assert available_intents is not None, "Failed to extract 'available_intents' wildcard from pattern"

    def test_heartdisease_config_parses_correctly(self, heartdisease_config_path: Path) -> None:
        """Verify heartdisease config parses without errors."""
        with open(heartdisease_config_path, "rb") as f:
            config = tomllib.load(f)

        # Verify required sections exist
        assert "prompt_learning" in config
        pl_config = config["prompt_learning"]
        assert "initial_prompt" in pl_config
        assert "gepa" in pl_config
        assert "evaluation" in pl_config["gepa"]

        # Verify initial_prompt has messages
        initial_prompt = pl_config["initial_prompt"]
        assert "messages" in initial_prompt
        assert len(initial_prompt["messages"]) >= 2  # Should have system and user

        # Verify messages have required fields
        for msg in initial_prompt["messages"]:
            assert "role" in msg
            assert "pattern" in msg
            assert msg["role"] in ["system", "user"]

    def test_banking77_config_parses_correctly(self, banking77_config_path: Path) -> None:
        """Verify banking77 config parses without errors."""
        with open(banking77_config_path, "rb") as f:
            config = tomllib.load(f)

        # Verify required sections exist
        assert "prompt_learning" in config
        pl_config = config["prompt_learning"]
        assert "initial_prompt" in pl_config
        assert "gepa" in pl_config
        assert "evaluation" in pl_config["gepa"]

        # Verify initial_prompt has messages
        initial_prompt = pl_config["initial_prompt"]
        assert "messages" in initial_prompt
        assert len(initial_prompt["messages"]) >= 2  # Should have system and user

        # Verify messages have required fields
        for msg in initial_prompt["messages"]:
            assert "role" in msg
            assert "pattern" in msg
            assert msg["role"] in ["system", "user"]

    def test_patterns_have_required_wildcards(self, heartdisease_config_path: Path, banking77_config_path: Path) -> None:
        """Verify patterns declare required wildcards."""
        for config_path, expected_wildcards in [
            (heartdisease_config_path, ["features"]),
            (banking77_config_path, ["query", "available_intents"]),
        ]:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            pl_config = config.get("prompt_learning", {})
            initial_prompt = pl_config.get("initial_prompt", {})
            wildcards = initial_prompt.get("wildcards", {})

            # Check that all expected wildcards are declared
            for wildcard in expected_wildcards:
                assert wildcard in wildcards, (
                    f"Required wildcard '{wildcard}' not declared in {config_path.name}\n"
                    f"Found wildcards: {list(wildcards.keys())}"
                )

            # Check that all declared wildcards appear in patterns
            messages = initial_prompt.get("messages", [])
            all_patterns = " ".join(msg.get("pattern", "") for msg in messages)
            for wildcard_name in wildcards.keys():
                assert f"{{{wildcard_name}}}" in all_patterns, (
                    f"Declared wildcard '{wildcard_name}' not found in any pattern in {config_path.name}"
                )

