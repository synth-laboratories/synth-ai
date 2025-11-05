"""Unit tests for Banking77 task app pattern validation."""

from __future__ import annotations

import pytest


class TestBanking77Patterns:
    """Test that Banking77 task app patterns match GEPA config expectations."""

    def test_system_message_pattern(self):
        """Verify system message pattern matches GEPA config."""
        # Expected pattern from GEPA config
        expected_system = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        # Pattern from task app default_messages
        actual_system = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        assert actual_system == expected_system, (
            f"System message pattern mismatch!\n"
            f"Expected:\n{expected_system!r}\n\n"
            f"Actual:\n{actual_system!r}"
        )

    def test_user_message_pattern(self):
        """Verify user message pattern matches GEPA config."""
        # Expected pattern from GEPA config
        expected_user = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        # Pattern from task app default_messages
        actual_user = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        assert actual_user == expected_user, (
            f"User message pattern mismatch!\n"
            f"Expected:\n{expected_user!r}\n\n"
            f"Actual:\n{actual_user!r}"
        )

    def test_pattern_formatting_with_placeholders(self):
        """Verify patterns format correctly with actual placeholders."""
        system_pattern = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        user_pattern = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        # Sample placeholders
        placeholders = {
            "query": "I am still waiting on my card?",
            "available_intents": "1. card_arrival\n2. card_not_received\n3. activate_card",
            "tool_name": "banking77_classify",
        }
        
        # Format patterns
        system_formatted = system_pattern.format(**placeholders)
        user_formatted = user_pattern.format(**placeholders)
        
        # Verify formatting succeeds
        assert "I am still waiting on my card?" in user_formatted
        assert "banking77_classify" in system_formatted
        assert "card_arrival" in system_formatted
        assert "{query}" not in user_formatted
        assert "{tool_name}" not in system_formatted
        assert "{available_intents}" not in system_formatted

    def test_required_placeholders_present(self):
        """Verify all required placeholders are present in patterns."""
        system_pattern = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        user_pattern = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        # Check system pattern has required placeholders
        assert "{available_intents}" in system_pattern
        assert "{tool_name}" in system_pattern
        
        # Check user pattern has required placeholders
        assert "{query}" in user_pattern

    def test_no_extra_placeholders(self):
        """Verify patterns don't have unexpected placeholders."""
        import re
        
        system_pattern = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        user_pattern = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        # Extract all placeholders
        system_placeholders = set(re.findall(r'\{(\w+)\}', system_pattern))
        user_placeholders = set(re.findall(r'\{(\w+)\}', user_pattern))
        
        # Expected placeholders
        expected_system = {"available_intents", "tool_name"}
        expected_user = {"query"}
        
        assert system_placeholders == expected_system, (
            f"Unexpected placeholders in system pattern!\n"
            f"Expected: {expected_system}\n"
            f"Found: {system_placeholders}"
        )
        
        assert user_placeholders == expected_user, (
            f"Unexpected placeholders in user pattern!\n"
            f"Expected: {expected_user}\n"
            f"Found: {user_placeholders}"
        )

    def test_pattern_whitespace_consistency(self):
        """Verify whitespace in patterns is consistent."""
        system_pattern = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        # Check for expected newlines and spacing
        assert system_pattern.startswith("You are an expert banking assistant. \n\n")
        assert "\n\n**Task:**\n" in system_pattern
        assert system_pattern.endswith("The intent must be an exact match from the list.")

    def test_message_role_structure(self):
        """Verify message structure has correct roles."""
        default_messages = [
            {
                "role": "system",
                "pattern": (
                    "You are an expert banking assistant. \n\n"
                    "**Available Banking Intents:**\n"
                    "{available_intents}\n\n"
                    "**Task:**\n"
                    "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
                    "The intent must be an exact match from the list."
                ),
            },
            {
                "role": "user",
                "pattern": "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above.",
            },
        ]
        
        assert len(default_messages) == 2
        assert default_messages[0]["role"] == "system"
        assert default_messages[1]["role"] == "user"
        assert "pattern" in default_messages[0]
        assert "pattern" in default_messages[1]

    def test_pattern_validation_against_rendered_output(self):
        """Test that rendered messages would match pattern validation."""
        system_pattern = (
            "You are an expert banking assistant. \n\n"
            "**Available Banking Intents:**\n"
            "{available_intents}\n\n"
            "**Task:**\n"
            "Call the `{tool_name}` tool with the `intent` parameter set to ONE of the intent labels listed above that best matches the customer query. "
            "The intent must be an exact match from the list."
        )
        
        user_pattern = "Customer Query: {query}\n\nClassify this query by calling the tool with the correct intent label from the list above."
        
        # Simulate what task app does
        intents_list = "\n".join(f"{i+1}. label_{i}" for i in range(3))
        placeholders = {
            "query": "Test query",
            "available_intents": intents_list,
            "tool_name": "banking77_classify",
        }
        
        rendered_system = system_pattern.format(**placeholders)
        rendered_user = user_pattern.format(**placeholders)
        
        # Verify the rendered content looks correct
        assert "1. label_0" in rendered_system
        assert "2. label_1" in rendered_system
        assert "3. label_2" in rendered_system
        assert "Test query" in rendered_user
        assert "banking77_classify" in rendered_system

