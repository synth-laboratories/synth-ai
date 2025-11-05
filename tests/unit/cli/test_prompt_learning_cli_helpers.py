"""Unit tests for prompt learning CLI helper functions."""

from __future__ import annotations

import pytest
from synth_ai.api.train.cli import _format_text_replacements

pytestmark = pytest.mark.unit


class TestFormatTextReplacements:
    """Tests for _format_text_replacements helper function."""

    def test_format_single_replacement(self) -> None:
        """Test formatting a single text replacement."""
        obj = {
            "text_replacements": [
                {
                    "new_text": "You are a helpful assistant.",
                    "apply_to_role": "system",
                }
            ]
        }
        lines = _format_text_replacements(obj)
        assert len(lines) == 2  # Role line + empty line
        assert "[SYSTEM]: You are a helpful assistant." in lines[0]
        assert lines[1] == ""

    def test_format_multiple_replacements(self) -> None:
        """Test formatting multiple text replacements."""
        obj = {
            "text_replacements": [
                {
                    "new_text": "System message",
                    "apply_to_role": "system",
                },
                {
                    "new_text": "User message",
                    "apply_to_role": "user",
                },
                {
                    "new_text": "Assistant message",
                    "apply_to_role": "assistant",
                },
            ]
        }
        lines = _format_text_replacements(obj)
        assert len(lines) == 6  # 3 replacements * 2 lines each
        assert "[SYSTEM]: System message" in lines[0]
        assert "[USER]: User message" in lines[2]
        assert "[ASSISTANT]: Assistant message" in lines[4]

    def test_format_with_max_display_limit(self) -> None:
        """Test that max_display limits the number of replacements shown."""
        obj = {
            "text_replacements": [
                {"new_text": f"Message {i}", "apply_to_role": "system"}
                for i in range(10)
            ]
        }
        lines = _format_text_replacements(obj, max_display=3)
        # Should only show 3 replacements (6 lines total)
        assert len(lines) == 6
        assert "[SYSTEM]: Message 0" in lines[0]
        assert "[SYSTEM]: Message 2" in lines[4]

    def test_format_empty_text_replacements(self) -> None:
        """Test formatting with empty text_replacements list."""
        obj = {"text_replacements": []}
        lines = _format_text_replacements(obj)
        assert lines == []

    def test_format_missing_text_replacements(self) -> None:
        """Test formatting when text_replacements key is missing."""
        obj = {}
        lines = _format_text_replacements(obj)
        assert lines == []

    def test_format_none_object(self) -> None:
        """Test formatting with None object."""
        lines = _format_text_replacements(None)
        assert lines == []

    def test_format_non_dict_object(self) -> None:
        """Test formatting with non-dict object."""
        lines = _format_text_replacements("not a dict")
        assert lines == []

    def test_format_text_replacements_not_list(self) -> None:
        """Test formatting when text_replacements is not a list."""
        obj = {"text_replacements": "not a list"}
        lines = _format_text_replacements(obj)
        assert lines == []

    def test_format_replacement_with_empty_new_text(self) -> None:
        """Test that replacements with empty new_text are skipped."""
        obj = {
            "text_replacements": [
                {"new_text": "", "apply_to_role": "system"},
                {"new_text": "Valid message", "apply_to_role": "user"},
            ]
        }
        lines = _format_text_replacements(obj)
        # Should only show the valid message
        assert len(lines) == 2
        assert "[USER]: Valid message" in lines[0]

    def test_format_replacement_with_default_role(self) -> None:
        """Test that default role is 'system' when not specified."""
        obj = {
            "text_replacements": [
                {"new_text": "Message without role"}
            ]
        }
        lines = _format_text_replacements(obj)
        assert "[SYSTEM]: Message without role" in lines[0]

    def test_format_replacement_with_non_dict_item(self) -> None:
        """Test that non-dict items in text_replacements are skipped."""
        obj = {
            "text_replacements": [
                {"new_text": "Valid", "apply_to_role": "system"},
                "invalid string",
                {"new_text": "Also valid", "apply_to_role": "user"},
            ]
        }
        lines = _format_text_replacements(obj)
        # Should only show the two valid replacements
        assert len(lines) == 4
        assert "[SYSTEM]: Valid" in lines[0]
        assert "[USER]: Also valid" in lines[2]

    def test_format_with_special_characters(self) -> None:
        """Test formatting with special characters in text."""
        obj = {
            "text_replacements": [
                {
                    "new_text": "Message with\nnewlines and\t\ttabs",
                    "apply_to_role": "system",
                }
            ]
        }
        lines = _format_text_replacements(obj)
        assert "Message with\nnewlines and\t\ttabs" in lines[0]

    def test_format_role_case_insensitive(self) -> None:
        """Test that role is converted to uppercase in output."""
        obj = {
            "text_replacements": [
                {"new_text": "Message", "apply_to_role": "system"},
                {"new_text": "Message", "apply_to_role": "USER"},
                {"new_text": "Message", "apply_to_role": "Assistant"},
            ]
        }
        lines = _format_text_replacements(obj)
        assert "[SYSTEM]: Message" in lines[0]
        assert "[USER]: Message" in lines[2]
        assert "[ASSISTANT]: Message" in lines[4]

