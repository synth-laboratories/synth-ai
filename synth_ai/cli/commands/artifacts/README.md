# Artifacts CLI

The `artifacts` command provides a unified interface for managing and inspecting artifacts created by Synth AI, including fine-tuned models, RL-trained models, and optimized prompts.

## Overview

The artifacts CLI allows you to:
- **List** all artifacts (models and prompts) with filtering options
- **Show** detailed information about specific artifacts
- **Export** models to HuggingFace (private or public)
- **Download** optimized prompts in various formats (JSON, YAML, text)

## Installation

The artifacts CLI is included with the Synth AI CLI. No additional installation required.

## Command Structure

```bash
uvx synth-ai artifacts <subcommand> [options]
```

### Subcommands

- `list` - List all artifacts
- `show` - Show detailed information about an artifact
- `export` - Export a model to HuggingFace
- `download` - Download optimized prompts

## Common Options

All commands support these options:

- `--base-url <URL>` - Backend base URL (default: from environment)
- `--api-key <KEY>` - API key for authentication (default: from `SYNTH_API_KEY` env var)
- `--timeout <SECONDS>` - Request timeout in seconds (default: 30.0)

## Artifact ID Formats

### Model IDs

Models use a prefix-based format:

- **Fine-tuned models**: `ft:BASE_MODEL:JOB_ID` or `peft:BASE_MODEL:JOB_ID`
- **RL models**: `rl:BASE_MODEL:JOB_ID`

Examples:
- `ft:Qwen/Qwen3-0.6B:job_658ba4f3a93845aa`
- `peft:Qwen/Qwen3-0.6B:job_658ba4f3a93845aa`
- `rl:Qwen/Qwen3-0.6B:job_abc123def456`

### Prompt IDs

Prompt optimization jobs use a simple prefix format:

- **Canonical format**: `pl_JOB_ID`
- **Alternative format**: `job_pl_JOB_ID`
- **Bare job ID**: `JOB_ID` (assumed to be prompt learning)

Examples:
- `pl_71c12c4c7c474c34`
- `job_pl_71c12c4c7c474c34`

## Commands

### `artifacts list`

List all artifacts (models and prompts) with optional filtering.

**Usage:**
```bash
uvx synth-ai artifacts list [options]
```

**Options:**
- `--type <TYPE>` - Filter by type: `models`, `prompts`, or `all` (default: `all`)
- `--status <STATUS>` - Filter by status: `succeeded`, `failed`, or `running` (default: `succeeded`)
- `--limit <N>` - Maximum items per type (default: 50)
- `--format <FORMAT>` - Output format: `table` or `json` (default: `table`)

**Examples:**
```bash
# List all artifacts
uvx synth-ai artifacts list

# List only models
uvx synth-ai artifacts list --type models

# List only prompts
uvx synth-ai artifacts list --type prompts

# List in JSON format
uvx synth-ai artifacts list --format json

# List with custom limit
uvx synth-ai artifacts list --limit 100
```

**Output:**
- **Table format**: Pretty-printed tables with summary counts
- **JSON format**: Complete JSON response with all artifact data

### `artifacts show`

Show detailed information about a specific artifact (model or prompt).

**Usage:**
```bash
uvx synth-ai artifacts show <ARTIFACT_ID> [options]
```

**Options:**
- `--format <FORMAT>` - Output format: `table` or `json` (default: `table`)
- `--verbose` / `-v` - Show verbose details (full metadata, snapshot, etc.) - only applies to prompts

**Examples:**
```bash
# Show model details
uvx synth-ai artifacts show ft:Qwen/Qwen3-0.6B:job_12345

# Show prompt details (default: summary + best prompt)
uvx synth-ai artifacts show pl_71c12c4c7c474c34

# Show prompt with verbose details
uvx synth-ai artifacts show pl_71c12c4c7c474c34 --verbose

# Export prompt data as JSON
uvx synth-ai artifacts show pl_71c12c4c7c474c34 --format json > prompt.json
```

**Output for Models:**
- Model ID, type, base model
- Job ID, status, creation timestamp
- Additional fields for RL models (dtype, weights path)

**Output for Prompts (default):**
- Job summary (algorithm, scores, status, timestamps)
- **Best optimized prompt** extracted from snapshot
- Syntax-highlighted prompt text with role colors

**Output for Prompts (verbose):**
- All default information
- Full metadata with important fields highlighted
- Complete snapshot JSON
- All metadata keys listed

### `artifacts export`

Export a fine-tuned or RL model to HuggingFace Hub.

**Usage:**
```bash
uvx synth-ai artifacts export <MODEL_ID> [options]
```

**Options:**
- `--repo-id <REPO_ID>` - HuggingFace repository ID (required)
- `--private` - Make repository private (default: public)
- `--token <TOKEN>` - HuggingFace token (optional, uses backend credentials if not provided)

**Examples:**
```bash
# Export model to HuggingFace (public)
uvx synth-ai artifacts export ft:Qwen/Qwen3-0.6B:job_12345 \
  --repo-id username/my-model

# Export as private repository
uvx synth-ai artifacts export ft:Qwen/Qwen3-0.6B:job_12345 \
  --repo-id username/my-model --private

# Export with explicit HF token
uvx synth-ai artifacts export ft:Qwen/Qwen3-0.6B:job_12345 \
  --repo-id username/my-model --token hf_xxxxx
```

**Note:** The export uses backend-stored HuggingFace credentials by default. You can provide your own token via `--token` if needed.

### `artifacts download`

Download optimized prompts in various formats.

**Usage:**
```bash
uvx synth-ai artifacts download <PROMPT_ID> [options]
```

**Options:**
- `--output <FILE>` - Output file path (default: prints to stdout)
- `--format <FORMAT>` - Output format: `json`, `yaml`, or `text` (default: `json`)
- `--snapshot-id <ID>` - Download specific snapshot (default: best snapshot)
- `--all-snapshots` - Download all snapshots (not yet implemented)

**Examples:**
```bash
# Download best prompt as JSON
uvx synth-ai artifacts download pl_71c12c4c7c474c34

# Download as YAML
uvx synth-ai artifacts download pl_71c12c4c7c474c34 --format yaml

# Download as plain text
uvx synth-ai artifacts download pl_71c12c4c7c474c34 --format text

# Save to file
uvx synth-ai artifacts download pl_71c12c4c7c474c34 --output prompt.json
```

**Output Formats:**
- **JSON**: Complete prompt snapshot as JSON
- **YAML**: Human-readable YAML format
- **Text**: Extracted prompt messages as plain text

## Authentication

The artifacts CLI uses the same authentication as other Synth AI CLI commands:

1. **API Key**: Set `SYNTH_API_KEY` environment variable, or use `--api-key` flag
2. **Backend URL**: Set `BACKEND_BASE_URL` environment variable, or use `--base-url` flag

**Example:**
```bash
export SYNTH_API_KEY="sk_live_..."
export BACKEND_BASE_URL="https://api.useautumn.com"
uvx synth-ai artifacts list
```

## Error Handling

The CLI provides clear error messages for common issues:

- **Invalid artifact ID format**: Shows expected format
- **Artifact not found**: 404 error with helpful message
- **Authentication failure**: 401/403 error with guidance
- **Network errors**: Timeout and connection error handling

## Examples

### List all successful prompts
```bash
uvx synth-ai artifacts list --type prompts --status succeeded
```

### Show best prompt from a job
```bash
uvx synth-ai artifacts show pl_71c12c4c7c474c34
```

### Export model and download prompt
```bash
# Export model
uvx synth-ai artifacts export ft:Qwen/Qwen3-0.6B:job_12345 \
  --repo-id myorg/my-model --private

# Download prompt
uvx synth-ai artifacts download pl_71c12c4c7c474c34 \
  --format json --output best_prompt.json
```

### Get all artifacts as JSON for scripting
```bash
uvx synth-ai artifacts list --format json | jq '.prompts[] | select(.algorithm == "gepa")'
```

## Implementation Details

### ID Parsing

The CLI uses centralized parsing logic in `synth_ai.cli.commands.artifacts.parsing`:

- **Model IDs**: Parsed into `ParsedModelId` with `prefix`, `base_model`, `job_id`, and `full_id`
- **Prompt IDs**: Parsed into `ParsedPromptId` with `job_id` and `full_id`
- **Type Detection**: Automatically detects artifact type from ID format
- **Validation**: Validates ID format before making API calls

### Backend Endpoints

The CLI communicates with these backend endpoints:

- `GET /api/artifacts` - List all artifacts
- `GET /api/artifacts/models/{model_id}` - Get model details
- `GET /api/artifacts/prompts/{job_id}` - Get prompt details
- `POST /api/learning/exports/hf` - Export to HuggingFace

### Prompt Extraction

The `show` command intelligently extracts prompt messages from various snapshot structures:

1. Direct `messages` array
2. `object` → `messages` array
3. `object` → `text_replacements` array (GEPA structure)
4. `initial_prompt` → `data` → `messages`

## Troubleshooting

### "Artifact not found" errors

- Verify the artifact ID format is correct
- Check that the artifact belongs to your organization
- Ensure you're using the correct backend URL

### "Authentication failed" errors

- Verify `SYNTH_API_KEY` is set correctly
- Check API key permissions
- Ensure backend URL is correct

### Empty metadata in prompt details

- This may indicate the job is still running or failed
- Use `--verbose` flag to see full details
- Check backend logs for job status

## Related Commands

- `synth-ai status` - Check system status and job states
- `synth-ai train` - Train new models and optimize prompts

## See Also

- [Synth AI Documentation](https://docs.useautumn.com)
- [API Reference](https://docs.useautumn.com/api)

