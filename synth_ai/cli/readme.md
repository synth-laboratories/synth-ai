# synth-ai CLI

Collection of CLI commands to run demos, setup credentials, start training jobs, validate task apps, and more.

## Commands

### Core Commands

- `synth-ai setup` – Interactive credential pairing with browser
- `synth-ai demo` – Create a demo directory for quick starts
- `synth-ai deploy` – Deploy local services to Modal
- `synth-ai run` – Start a training run
- `synth-ai train` – Interactive training launcher

### Task App Commands

- `synth-ai task-app list` – List registered task apps
- `synth-ai task-app serve <app_id>` – Start a local task app server
- `synth-ai task-app validate <app_id>` – Validate task app deployment readiness
- `synth-ai task-app deploy <app_id>` – Deploy a task app to Modal
- `synth-ai task-app info <app_id>` – Fetch task metadata from a running app
- `synth-ai task-app eval <app_id>` – Run evaluations against a task app
- `synth-ai task-app filter` – Filter and export SFT data from traces

### Task App Validation

The `validate` command is particularly useful for ensuring your task app is deployment-ready:

```bash
# Validate grpo-crafter (starts local server automatically)
synth-ai task-app validate grpo-crafter

# Validate with verbose output to see all endpoint details
synth-ai task-app validate sokoban --verbose

# Validate a remote deployment
synth-ai task-app validate grpo-crafter --url https://my-crafter.modal.run

# Use in CI/CD with JSON output
synth-ai task-app validate sokoban --json
```

The validation checks:
- ✅ All required HTTP endpoints (/, /health, /info, /task_info, /rollout)
- ✅ Authentication configuration
- ✅ Task instance availability (default: min 10 instances)

**Options:**
- `--url TEXT` – Validate a remote deployment (otherwise starts local server)
- `--port INTEGER` – Port for temporary server (default: 8765)
- `--api-key TEXT` – API key for authentication (default: $ENVIRONMENT_API_KEY)
- `--min-instances INTEGER` – Minimum required task instances (default: 10)
- `-v, --verbose` – Show detailed endpoint information
- `--json` – Output results as JSON for automation

**Common use cases:**
- Pre-deployment verification: Check task app works before deploying to Modal
- CI/CD integration: Use `--json` flag for automated validation in pipelines
- Debug failing deployments: Use `--verbose` to see detailed endpoint responses
- Test API key configuration: Verify authentication is set up correctly

For full details:

```bash
synth-ai --help
synth-ai task-app --help
synth-ai task-app validate --help
```




