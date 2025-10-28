# Help System Documentation

This directory contains the help content and command implementation for the Synth AI CLI.

## Overview

The help system provides comprehensive documentation for CLI commands that is displayed when users run `--help` flags or the dedicated `help` command.

## Usage

### For End Users

```bash
# Get detailed help for a specific command
uvx synth-ai help deploy
uvx synth-ai help setup

# List all available help topics
uvx synth-ai help

# Use standard --help flags (same content)
uvx synth-ai deploy --help
uvx synth-ai setup --help
```

### For Developers

To add help content for a new command:

1. **Add help text to `__init__.py`:**

```python
NEW_COMMAND_HELP = """
Your detailed help content here...

OVERVIEW
--------
Description of the command

USAGE
-----
Examples and usage patterns

TROUBLESHOOTING
---------------
Common issues and solutions
"""

COMMAND_HELP = {
    "deploy": DEPLOY_HELP,
    "setup": SETUP_HELP,
    "new-command": NEW_COMMAND_HELP,  # Add your new command
}
```

2. **Import and use in your command:**

```python
# In your command file (e.g., synth_ai/cli/commands/mycommand/core.py)
try:
    from synth_ai.cli.commands.help import NEW_COMMAND_HELP
except ImportError:
    NEW_COMMAND_HELP = "Brief description for fallback."

@click.command(
    "new-command",
    help=NEW_COMMAND_HELP,
    epilog="Run 'uvx synth-ai new-command --help' for detailed usage information.",
)
def my_command():
    """Brief docstring for the command.
    
    This appears in the command list when running 'uvx synth-ai --help'.
    The detailed help from NEW_COMMAND_HELP appears when running the --help flag.
    """
    pass
```

## Structure

```
synth_ai/cli/commands/help/
├── __init__.py       # Help content storage and retrieval
├── core.py           # Help command implementation
└── README.md         # This file
```

### `__init__.py`

Contains:
- `DEPLOY_HELP`: Detailed help for the deploy command
- `SETUP_HELP`: Detailed help for the setup command
- `COMMAND_HELP`: Dictionary mapping command names to help text
- `get_command_help()`: Function to retrieve help for a specific command

### `core.py`

Contains:
- `help_command()`: Click command that displays detailed help
- `register()`: Function to register the help command with the CLI
- `get_command()`: Function to get the help command for registration

## Help Content Guidelines

When writing help content, follow these patterns:

1. **Structure:**
   - OVERVIEW: High-level description
   - USAGE: Basic usage examples
   - [SECTION]: Specific use cases (e.g., MODAL DEPLOYMENT, LOCAL DEVELOPMENT)
   - TROUBLESHOOTING: Common issues and solutions
   - ENVIRONMENT VARIABLES: Relevant env vars (if applicable)
   - Links: Documentation URLs for more info

2. **Formatting:**
   - Use clear section headers with underlines
   - Include practical examples
   - Keep lines under 80 characters when possible
   - Use bullet points (•) for lists
   - Use arrows (→) for actions/solutions

3. **Content:**
   - Start with the most common use cases
   - Include specific command examples
   - Anticipate common errors and provide solutions
   - Link to external docs for more details

## Example

```python
MY_COMMAND_HELP = """
Brief one-line description of what the command does.

OVERVIEW
--------
Detailed explanation of the command's purpose and what it accomplishes.

USAGE
-----
  # Basic usage
  uvx synth-ai my-command

  # With options
  uvx synth-ai my-command --option value

OPTIONS
-------
  --option VALUE     Description of what this option does
  --flag             Description of what this flag does

EXAMPLES
--------
  # Example 1: Common use case
  uvx synth-ai my-command --option foo

  # Example 2: Another use case
  uvx synth-ai my-command --flag

TROUBLESHOOTING
---------------
1. "Common Error Message"
   → Solution: What to do to fix it

2. "Another Error"
   → Run: specific command to fix
   → Or: alternative solution

For more information: https://docs.usesynth.ai/my-command
"""
```

## Testing

Test the help system:

```bash
# Test the help command
uvx synth-ai help

# Test specific command help
uvx synth-ai help deploy
uvx synth-ai help setup

# Test --help flags
uvx synth-ai deploy --help
uvx synth-ai setup --help
```

## Integration

The help command is automatically registered in `synth_ai/cli/__init__.py`:

```python
# Register help command
_maybe_call("synth_ai.cli.commands.help.core", "register", cli)
```

This ensures the help command is available whenever the CLI is loaded.

