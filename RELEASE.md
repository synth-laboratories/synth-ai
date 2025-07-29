# Release Process

This document describes the automated release process for the `synth-ai` package.

## Overview

The release process is fully automated through the `dev/cut_release.sh` script, which:

1. **Generates Documentation**: Runs `dev/build_docs.sh` using `pdoc` to generate API reference from docstrings
2. **Syncs Documentation**: Copies generated docs to Mintlify directory
3. **Version Management**: Bumps version (dev, patch, minor, or major)
4. **Package Building**: Builds the Python package using `uv build`
5. **PyPI Upload**: Uploads to PyPI using `twine`
6. **Git Operations**: Creates feature branch, commits changes, and creates PR

## Release Workflow

### Prerequisites

- `pdoc` installed: `pip install pdoc`
- `uv` installed for building
- `twine` installed for PyPI upload
- GitHub CLI (`gh`) installed and authenticated
- Proper PyPI credentials configured

### Steps

1. **Documentation Generation**: Runs `dev/build_docs.sh` using `pdoc`
2. **Documentation Cleaning**: Applies Mintlify compatibility fixes
3. **Documentation Sync**: Copies cleaned docs to Mintlify directory
4. **Version Bump**: Prompts for version type and updates `pyproject.toml`
5. **Package Build**: Builds distribution files with `uv build`
6. **PyPI Upload**: Uploads to PyPI with `twine upload`
7. **Git Operations**: Creates branch, commits, and opens PR

## Usage

```bash
# Run the full release process
./dev/cut_release.sh

# The script will prompt for:
# - Version bump type (dev, patch, minor, major)
# - Confirmation before PyPI upload
# - Confirmation before Git operations
```

## Documentation Scripts

- **`dev/build_docs.sh`** - Standalone documentation generation script using `pdoc` (recommended)
- **`dev/sync_docs.sh`** - Syncs docs to Mintlify directory

### Documentation Generation

The documentation is generated using `pdoc`, a modern Python documentation generator that:

- Automatically extracts docstrings from Python modules
- Generates clean Markdown output
- Handles complex type annotations and signatures
- Avoids Unicode escape sequence issues
- Provides consistent, professional documentation

The build script focuses on core modules:
- `synth_ai.environments.environment.core`
- `synth_ai.environments.environment.registry`
- `synth_ai.environments.environment.tools`
- `synth_ai.environments.stateful.core`

### Documentation Cleaning

The `dev/clean_pdoc_output.py` script fixes Mintlify compatibility issues:

- Removes Python REPL examples (`>>>` and `...`) that cause parsing errors
- Converts colons in parameter descriptions to dashes
- Escapes HTML entities and problematic characters
- Ensures clean, parseable Markdown output

## Version Management

The script supports four version bump types:

- **dev**: Development version (e.g., 0.2.1.dev0 → 0.2.1.dev1)
- **patch**: Bug fixes (e.g., 0.2.1 → 0.2.2)
- **minor**: New features (e.g., 0.2.1 → 0.3.0)
- **major**: Breaking changes (e.g., 0.2.1 → 1.0.0)

## Git Workflow

1. Creates a feature branch: `release/v{version}`
2. Commits documentation and version changes
3. Pushes to remote
4. Creates a pull request to main branch
5. Provides summary of changes

## Error Handling

The script includes comprehensive error handling:

- Validates prerequisites before starting
- Checks for existing Git changes
- Validates version bump logic
- Confirms PyPI upload before proceeding
- Provides clear error messages and rollback instructions

## Manual Steps

After the script completes:

1. **Review the PR**: Check the generated pull request
2. **Test Documentation**: Verify Mintlify documentation is updated
3. **Merge PR**: Merge the release PR to main
4. **Tag Release**: Create a GitHub release tag (optional)

## Troubleshooting

### Common Issues

- **PyPI Upload Fails**: Check credentials and network connection
- **Git Operations Fail**: Ensure GitHub CLI is authenticated
- **Documentation Errors**: Check that `pdoc` is installed and modules are importable
- **Mintlify Parsing Errors**: Run `dev/build_docs.sh` to regenerate cleaned documentation

### Rollback

If something goes wrong:

1. **PyPI**: Contact PyPI admin to remove the uploaded version
2. **Git**: Reset the branch or delete the PR
3. **Version**: Manually revert version in `pyproject.toml`

## Configuration

The script uses these configuration files:

- `pyproject.toml`: Package metadata and version
- `dev/build_docs.sh`: Documentation generation settings
- `dev/sync_docs.sh`: Mintlify sync configuration
- `mintlify-docs/mint.json`: Mintlify site configuration 