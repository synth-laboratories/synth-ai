"""Unit tests for the help command system."""

import pytest
from click.testing import CliRunner
from synth_ai.cli.commands.help import COMMAND_HELP, DEPLOY_HELP, SETUP_HELP, get_command_help
from synth_ai.cli.commands.help.core import help_command


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestHelpContent:
    """Test help content availability and structure."""

    def test_deploy_help_exists(self) -> None:
        """Verify DEPLOY_HELP content is defined."""
        assert DEPLOY_HELP is not None
        assert len(DEPLOY_HELP) > 0
        assert isinstance(DEPLOY_HELP, str)

    def test_setup_help_exists(self) -> None:
        """Verify SETUP_HELP content is defined."""
        assert SETUP_HELP is not None
        assert len(SETUP_HELP) > 0
        assert isinstance(SETUP_HELP, str)

    def test_deploy_help_has_key_sections(self) -> None:
        """Verify DEPLOY_HELP contains expected sections."""
        assert "OVERVIEW" in DEPLOY_HELP
        assert "USAGE" in DEPLOY_HELP
        assert "MODAL DEPLOYMENT" in DEPLOY_HELP
        assert "LOCAL DEVELOPMENT" in DEPLOY_HELP
        assert "TROUBLESHOOTING" in DEPLOY_HELP
        assert "ENVIRONMENT VARIABLES" in DEPLOY_HELP

    def test_setup_help_has_key_sections(self) -> None:
        """Verify SETUP_HELP contains expected sections."""
        assert "OVERVIEW" in SETUP_HELP
        assert "USAGE" in SETUP_HELP
        assert "WHAT YOU'LL NEED" in SETUP_HELP
        assert "TROUBLESHOOTING" in SETUP_HELP
        assert "WHERE ARE KEYS STORED" in SETUP_HELP
        assert "NEXT STEPS" in SETUP_HELP

    def test_deploy_help_has_examples(self) -> None:
        """Verify DEPLOY_HELP contains practical examples."""
        assert "uvx synth-ai deploy" in DEPLOY_HELP
        assert "Examples:" in DEPLOY_HELP or "EXAMPLES" in DEPLOY_HELP

    def test_setup_help_has_examples(self) -> None:
        """Verify SETUP_HELP contains practical examples."""
        assert "uvx synth-ai setup" in SETUP_HELP

    def test_command_help_dict_populated(self) -> None:
        """Verify COMMAND_HELP dictionary contains expected commands."""
        assert "deploy" in COMMAND_HELP
        assert "setup" in COMMAND_HELP
        assert COMMAND_HELP["deploy"] == DEPLOY_HELP
        assert COMMAND_HELP["setup"] == SETUP_HELP

    def test_get_command_help_returns_correct_help(self) -> None:
        """Verify get_command_help returns the right content."""
        assert get_command_help("deploy") == DEPLOY_HELP
        assert get_command_help("setup") == SETUP_HELP

    def test_get_command_help_returns_none_for_unknown(self) -> None:
        """Verify get_command_help returns None for unknown commands."""
        assert get_command_help("nonexistent") is None
        assert get_command_help("") is None


class TestHelpCommand:
    """Test the help command CLI behavior."""

    def test_help_command_without_args_shows_list(self, runner: CliRunner) -> None:
        """Test help command without arguments shows available topics."""
        result = runner.invoke(help_command, [])
        assert result.exit_code == 0
        assert "Available help topics:" in result.output
        assert "deploy" in result.output
        assert "setup" in result.output
        assert "Usage:" in result.output
        assert "uvx synth-ai help [COMMAND]" in result.output

    def test_help_command_with_deploy_shows_deploy_help(self, runner: CliRunner) -> None:
        """Test help command with 'deploy' shows DEPLOY_HELP."""
        result = runner.invoke(help_command, ["deploy"])
        assert result.exit_code == 0
        assert "OVERVIEW" in result.output
        assert "MODAL DEPLOYMENT" in result.output
        assert "LOCAL DEVELOPMENT" in result.output
        assert "TROUBLESHOOTING" in result.output

    def test_help_command_with_setup_shows_setup_help(self, runner: CliRunner) -> None:
        """Test help command with 'setup' shows SETUP_HELP."""
        result = runner.invoke(help_command, ["setup"])
        assert result.exit_code == 0
        assert "Configure Synth AI credentials" in result.output or "OVERVIEW" in result.output
        assert "WHAT YOU'LL NEED" in result.output
        assert "WHERE ARE KEYS STORED" in result.output

    def test_help_command_with_unknown_command_fails_gracefully(
        self, runner: CliRunner
    ) -> None:
        """Test help command with unknown command provides helpful error."""
        result = runner.invoke(help_command, ["nonexistent"])
        assert result.exit_code != 0
        assert "No detailed help available for 'nonexistent'" in result.output
        assert "uvx synth-ai nonexistent --help" in result.output
        assert "uvx synth-ai help" in result.output

    def test_help_command_shows_all_topics(self, runner: CliRunner) -> None:
        """Test help command shows all available topics from COMMAND_HELP."""
        result = runner.invoke(help_command, [])
        assert result.exit_code == 0
        for command_name in COMMAND_HELP:
            assert command_name in result.output

    def test_help_command_suggests_standard_help_flag(self, runner: CliRunner) -> None:
        """Test help command suggests using standard --help flags."""
        result = runner.invoke(help_command, [])
        assert result.exit_code == 0
        assert "--help" in result.output
        assert "uvx synth-ai deploy --help" in result.output or "standard --help flags" in result.output


class TestHelpIntegrationWithCommands:
    """Test that help is properly integrated with actual commands."""

    def test_deploy_command_is_registered(self) -> None:
        """Test deploy command is exposed under the new CLI module."""
        from synth_ai.cli.deploy import deploy_cmd

        assert deploy_cmd is not None, "deploy_cmd should not be None"
        assert deploy_cmd.name == "deploy-cmd"

    def test_deploy_command_help_flag_works(self, runner: CliRunner) -> None:
        """Test deploy command --help flag displays help."""
        from synth_ai.cli.deploy import deploy_cmd

        assert deploy_cmd is not None, "deploy_cmd should not be None"
        result = runner.invoke(deploy_cmd, ["--help"])
        assert result.exit_code == 0
        # Should show either full DEPLOY_HELP or at least key sections
        assert (
            "Deploy" in result.output
            or "OVERVIEW" in result.output
            or "--runtime" in result.output
        )


class TestHelpContentQuality:
    """Test the quality and completeness of help content."""

    def test_deploy_help_mentions_common_errors(self) -> None:
        """Test deploy help includes common error messages."""
        assert "ENVIRONMENT_API_KEY" in DEPLOY_HELP
        assert "Modal CLI not found" in DEPLOY_HELP or "Modal" in DEPLOY_HELP
        assert "Port already in use" in DEPLOY_HELP or "port" in DEPLOY_HELP.lower()

    def test_deploy_help_includes_both_runtimes(self) -> None:
        """Test deploy help documents both modal and uvicorn runtimes."""
        assert "modal" in DEPLOY_HELP.lower()
        assert "uvicorn" in DEPLOY_HELP.lower()

    def test_setup_help_mentions_modal_authentication(self) -> None:
        """Test setup help mentions Modal authentication."""
        assert "Modal" in SETUP_HELP
        assert "modal.com" in SETUP_HELP or "modal token" in SETUP_HELP.lower()

    def test_setup_help_mentions_key_storage(self) -> None:
        """Test setup help explains where keys are stored."""
        assert ".synth/config" in SETUP_HELP or "config" in SETUP_HELP.lower()

    def test_help_content_provides_next_steps(self) -> None:
        """Test help content guides users to next steps."""
        # Setup should guide to deploy
        assert "deploy" in SETUP_HELP.lower()
        # Deploy should mention documentation or next steps
        assert (
            "docs.usesynth.ai" in DEPLOY_HELP
            or "documentation" in DEPLOY_HELP.lower()
            or "more information" in DEPLOY_HELP.lower()
        )


class TestHelpCommandEdgeCases:
    """Test edge cases and error handling."""

    def test_help_command_handles_empty_string(self, runner: CliRunner) -> None:
        """Test help command handles empty string gracefully."""
        # Empty string should be treated like no argument
        result = runner.invoke(help_command, [""])
        # Should either show list or handle gracefully
        assert result.exit_code in (0, 1)

    def test_help_command_is_case_sensitive(self, runner: CliRunner) -> None:
        """Test help command requires exact case match."""
        result = runner.invoke(help_command, ["Deploy"])  # Capital D
        # Should not find it (case sensitive)
        assert result.exit_code != 0 or "No detailed help available" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
