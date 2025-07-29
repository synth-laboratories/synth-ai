# Release Process

This document describes the release process for `synth-ai` using the automated release scripts.

## Overview

The release process is automated through several scripts:

1. **`dev/cut_release.sh`** - Main release script that handles the complete workflow
2. **`dev/build_docs.sh`** - Standalone documentation generation script
3. **`dev/generate_api_docs.py`** - Python script that generates API documentation from source code
4. **`dev/sync_docs.sh`** - Syncs documentation to Mintlify

## Prerequisites

Before running a release, ensure you have:

- `git` - Version control
- `python` - Python interpreter
- `uv` - Python package manager
- `twine` - PyPI upload tool
- `gh` (optional) - GitHub CLI for automatic PR creation

## Release Types

The release script supports four version bump types:

- **`dev`** - Development version bump (0.2.1.dev0 → 0.2.1.dev1)
- **`patch`** - Patch release (0.2.1.dev0 → 0.2.1)
- **`minor`** - Minor release (0.2.1.dev0 → 0.2.2)
- **`major`** - Major release (0.2.1.dev0 → 0.3.0)

## Quick Start

### Development Release (Most Common)

```bash
# Bump dev version
./dev/cut_release.sh dev

# Preview what would happen
./dev/cut_release.sh dev --dry-run
```

### Production Release

```bash
# Release patch version
./dev/cut_release.sh patch

# Release minor version
./dev/cut_release.sh minor

# Release major version
./dev/cut_release.sh major
```

## Release Workflow

The `dev/cut_release.sh` script performs the following steps:

1. **Prerequisites Check**
   - Verifies required tools are installed
   - Checks git repository status
   - Warns about uncommitted changes

2. **Version Bumping**
   - Updates `pyproject.toml` version
   - Supports dev/patch/minor/major bumps

3. **Documentation Generation**
   - Runs `python dev/generate_api_docs.py`
   - Generates API reference from source code
   - Syncs docs to Mintlify (if available)

4. **Changelog Update**
   - Creates new changelog entry
   - Backs up existing changelog

5. **PyPI Upload**
   - Builds package with `uv build`
   - Uploads to PyPI with `twine`

6. **Git Workflow**
   - Creates release branch
   - Commits all changes
   - Pushes branch
   - Creates PR (if `gh` is available)

## Script Options

### `dev/cut_release.sh` Options

```bash
--dry-run           # Preview changes without making them
--skip-docs         # Skip documentation generation and sync
--skip-pypi         # Skip PyPI upload
--skip-git          # Skip git workflow
--help, -h          # Show help message
```

### Examples

```bash
# Preview a minor release
./dev/cut_release.sh minor --dry-run

# Release without documentation
./dev/cut_release.sh patch --skip-docs

# Release without PyPI upload (for testing)
./dev/cut_release.sh dev --skip-pypi

# Release without git workflow (for CI/CD)
./dev/cut_release.sh patch --skip-git
```

## Documentation Generation

### Standalone Documentation Generation

The `dev/build_docs.sh` script provides a standalone way to generate documentation:

```bash
# Basic usage
./dev/build_docs.sh

# With options
./dev/build_docs.sh --verbose --fallback
```

**Features:**
- ✅ Prerequisites checking (Python, pyproject.toml, etc.)
- ✅ Automatic fallback if Python script fails
- ✅ Detailed reporting of generated files
- ✅ Multiple output formats and options
- ✅ Can be run independently or by cut_release.sh

**Options:**
- `--verbose, -v` - Show detailed output
- `--fallback` - Force fallback documentation generation
- `--check-only` - Only check prerequisites, don't generate docs
- `--help, -h` - Show help message

### Manual Documentation Generation

```bash
# Generate API docs (recommended)
./dev/build_docs.sh

# Generate with verbose output
./dev/build_docs.sh --verbose

# Force fallback documentation
./dev/build_docs.sh --fallback

# Check prerequisites only
./dev/build_docs.sh --check-only

# Sync to Mintlify
./dev/sync_docs.sh

# Preview sync
./dev/sync_docs.sh --dry-run
```

### Documentation Scripts

- **`dev/build_docs.sh`** - Standalone documentation generation script (recommended)
- **`dev/generate_api_docs.py`** - Python script that generates API reference from source code
- **`dev/sync_docs.sh`** - Syncs docs to Mintlify directory

## Release Checklist

Before running a release:

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Changelog is prepared
- [ ] No uncommitted changes (or commit them)
- [ ] PyPI credentials are configured

## Post-Release Steps

After a successful release:

1. **Review Pull Request**
   - Check the automated PR
   - Review changes and documentation

2. **Merge to Main**
   - Merge the release branch to main
   - Delete the release branch

3. **Tag Release**
   ```bash
   git tag v0.2.1
   git push origin v0.2.1
   ```

4. **Update Mintlify**
   - Review synced documentation
   - Update `mint.json` if needed
   - Deploy to Mintlify

## Troubleshooting

### Common Issues

**"Missing required commands"**
- Install missing tools: `pip install twine`, `brew install gh`

**"Not in a git repository"**
- Run from the `synth-ai` directory

**"Uncommitted changes"**
- Commit or stash changes before release

**"PyPI upload failed"**
- Check credentials: `twine check dist/*`
- Verify package builds correctly

**"Documentation sync failed"**
- Check Mintlify directory exists
- Verify docs were generated: `ls docs/api/`

### Debug Mode

For troubleshooting, run with verbose output:

```bash
# Test documentation generation
python dev/generate_api_docs.py

# Test sync
./dev/sync_docs.sh --dry-run

# Test release workflow
./dev/cut_release.sh dev --dry-run --skip-pypi --skip-git
```

## Configuration

### Version Management

Versions are managed in `pyproject.toml`:

```toml
[project]
version = "0.2.1.dev0"
```

### Documentation Paths

- **Source**: `docs/` (generated docs)
- **Target**: `../mintlify-docs/synth-ai/` (Mintlify docs)

### PyPI Configuration

Configure PyPI credentials:

```bash
# Set up credentials
twine upload --help

# Test upload
twine check dist/*
```

## CI/CD Integration

For automated releases, you can:

1. **Skip Git Workflow**: Use `--skip-git` for CI environments
2. **Dry Run**: Use `--dry-run` to preview changes
3. **Selective Steps**: Use `--skip-*` flags to customize workflow

Example CI script:

```bash
#!/bin/bash
# CI release script
./dev/cut_release.sh patch --skip-git --skip-docs
```

## Contributing

To improve the release process:

1. **Add New Scripts**: Place in `dev/` directory
2. **Update Documentation**: Modify this README
3. **Test Changes**: Use `--dry-run` flag
4. **Version Bumps**: Follow semantic versioning

## Support

For issues with the release process:

1. Check this documentation
2. Run with `--dry-run` to debug
3. Check prerequisites and configuration
4. Review error messages and logs 