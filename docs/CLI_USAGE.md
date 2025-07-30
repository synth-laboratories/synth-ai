# Synth AI CLI Usage

After installing the `synth-ai` package, you can use the CLI to start the required services.

## Installation

```bash
pip install synth-ai
# or with uv
uv pip install synth-ai
```

## Starting Services

The `synth serve` command starts the required services for Synth AI:
- **sqld daemon**: SQLite-compatible database server for v3 tracing
- **Environment service**: API for managing AI environments

### Basic Usage

```bash
# Start all services
synth serve

# Or using uvx (no installation needed)
uvx synth serve

# Or as a Python module
python -m synth_ai serve
```

### Command Options

```bash
synth serve --help

Options:
  --db-file TEXT       Database file path (default: synth_ai.db)
  --sqld-port INTEGER  Port for sqld HTTP interface (default: 8080)
  --env-port INTEGER   Port for environment service (default: 8901)
  --no-sqld            Skip starting sqld daemon
  --no-env             Skip starting environment service
```

### Examples

```bash
# Use custom ports
synth serve --sqld-port 9000 --env-port 9001

# Use a different database file
synth serve --db-file /path/to/my_database.db

# Start only sqld (no environment service)
synth serve --no-env

# Start only environment service (assumes sqld is already running)
synth serve --no-sqld
```

## Automatic sqld Installation

If `sqld` is not found on your system, the CLI will automatically download and install it to `~/.local/bin/sqld`.

To manually install sqld:
```bash
# Using the included script
python -c "from synth_ai.cli import install_sqld; install_sqld()"

# Or download directly
curl -L https://github.com/tursodatabase/libsql/releases/download/libsql-server-v0.26.2/sqld-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m).tar.xz | tar -xJ
```

## Service Details

### sqld Daemon
- HTTP API: `http://127.0.0.1:8080` (default)
- Database file: `synth_ai.db` (default)
- Log file: `sqld.log`

### Environment Service
- API endpoint: `http://127.0.0.1:8901` (default)
- Auto-reloads on code changes
- Provides environment management for AI agents

## Stopping Services

Press `Ctrl+C` to gracefully stop all services.

## Troubleshooting

### Port Already in Use
If you see a "port already in use" error, either:
1. Stop the existing service using that port
2. Use a different port with `--sqld-port` or `--env-port`

### Database Issues
Check `sqld.log` for database-related errors.

### Missing Dependencies
Make sure all dependencies are installed:
```bash
pip install synth-ai[all]
```