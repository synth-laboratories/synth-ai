# Scan Command

The `scan` command discovers and performs health checks on running task applications deployed locally or via Cloudflare tunnels. It provides structured output in both human-readable table format and machine-readable JSON format, making it suitable for terminal use and programmatic consumption by CLI agents and automation tools.

## Overview

The scan command uses multiple discovery methods to find active task apps:

- **Port Scanning**: Scans specified port ranges for local HTTP servers
- **Service Records**: Reads deployed local services from persistent records (`~/.synth-ai/services.json`)
- **Tunnel Records**: Reads deployed Cloudflare tunnels from persistent records
- **Process Scanning**: Inspects running `cloudflared` processes for tunnel URLs
- **Backend API**: Queries backend for managed tunnel information
- **Registry**: Checks task app registry for registered apps

For each discovered app, the command performs health checks by:
- Making HTTP GET requests to `/health` endpoints
- Extracting metadata from `/info` endpoints (app_id, version, task_name, etc.)
- Supporting API key authentication via `X-API-Key` header

## Usage

### Basic Usage

```bash
# Scan default port range (8000-8100) and show table
synth-ai scan

# Scan specific port range
synth-ai scan --port-range 8000:9000

# Get JSON output for programmatic use
synth-ai scan --json

# Show verbose scanning progress
synth-ai scan --verbose
```

### Options

- `--port-range START:END`: Port range to scan (default: `8000:8100`)
- `--timeout SECONDS`: Health check timeout in seconds (default: `2.0`)
- `--api-key KEY`: API key for health checks (default: from `ENVIRONMENT_API_KEY` env var)
- `--json`: Output results as JSON instead of table
- `--verbose`: Show detailed scanning progress

### Examples

```bash
# Scan default range with table output
$ synth-ai scan
Found 3 active task apps:

Name                                             Port  Status     Type       App ID              Version Discovered Via
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
banking77                                        8000  ✅ healthy  local      banking77           1.0.0   service_records
banking77                                        8001  ✅ healthy  cloudflare banking77           1.0.0   tunnel_records
localhost:8002                                   8002  ⚠️  unhealthy local      -                   -       port_scan
```

```bash
# Get JSON output
$ synth-ai scan --json
{
  "apps": [
    {
      "name": "banking77",
      "url": "http://127.0.0.1:8000",
      "type": "local",
      "health_status": "healthy",
      "port": 8000,
      "tunnel_mode": null,
      "tunnel_hostname": null,
      "app_id": "banking77",
      "task_name": "Banking77 Intent Classification",
      "dataset_id": null,
      "version": "1.0.0",
      "metadata": {
        "service": {
          "task": {
            "id": "banking77",
            "name": "Banking77 Intent Classification",
            "version": "1.0.0"
          }
        }
      },
      "discovered_via": "service_records"
    }
  ],
  "scan_summary": {
    "total_found": 1,
    "healthy": 1,
    "unhealthy": 0,
    "local_count": 1,
    "cloudflare_count": 0
  }
}
```

```bash
# Scan wider port range with custom timeout
$ synth-ai scan --port-range 7000:9000 --timeout 5.0 --verbose
Scanning ports 7000-9000...
Found 5 local app(s)
Found 2 Cloudflare app(s)
Found 3 service record(s)
...
```

## Output Formats

### Table Format (Default)

The table format provides a human-readable view with columns for:
- **Name**: App identifier (from `/info` endpoint or fallback)
- **Port**: Local port number
- **Status**: Health status with icon (✅ healthy, ⚠️ unhealthy, ❓ unknown)
- **Type**: Deployment type (`local` or `cloudflare`)
- **App ID**: Task app identifier
- **Version**: App version
- **Discovered Via**: Discovery method used

### JSON Format (`--json`)

The JSON format provides machine-readable output with:
- **apps**: Array of app objects with full metadata
- **scan_summary**: Aggregated statistics (total_found, healthy, unhealthy, local_count, cloudflare_count)

Each app object includes:
- Basic info: `name`, `url`, `type`, `health_status`, `port`
- Tunnel info: `tunnel_mode`, `tunnel_hostname` (for Cloudflare apps)
- App metadata: `app_id`, `task_name`, `dataset_id`, `version`
- Discovery: `discovered_via`
- Full metadata: `metadata` dictionary with complete `/info` response and deployment details

## Health Status

Health status is determined by checking the `/health` endpoint:

- **healthy**: HTTP 200 with valid JSON response containing `"status": "healthy"` or `"healthy": true`, or any HTTP 200 with valid JSON
- **unhealthy**: HTTP error status (4xx, 5xx), Cloudflare tunnel errors (530, 502), or HTML responses (Cloudflare error pages)
- **unknown**: Request timeout, connection errors, or other exceptions

## Discovery Methods

### Port Scanning

Scans specified port ranges by:
1. Checking if ports are open using socket connections
2. Making HTTP requests to `/health` endpoints on open ports
3. Extracting metadata from `/info` endpoints

**Discovered via**: `port_scan`

### Service Records

Reads from persistent service records file (`~/.synth-ai/services.json`) created when deploying local services via `synth-ai deploy --runtime local`. These records include:
- Service URL and port
- Process ID (PID)
- App ID and task app path
- Creation timestamp

**Discovered via**: `service_records`

### Tunnel Records

Reads from persistent tunnel records file created when deploying tunnels via `synth-ai deploy --runtime tunnel`. These records include:
- Tunnel URL and hostname
- Tunnel mode (quick/managed)
- Process ID (PID)
- Target port and local host

**Discovered via**: `tunnel_records`

### Process Scanning

Inspects running `cloudflared` processes to extract tunnel URLs from:
- Process command-line arguments (`--url` flag)
- Process stdout/stderr output
- Associated log files

**Discovered via**: `cloudflared_process`

### Backend API

Queries the backend API for managed tunnel information (requires `SYNTH_API_KEY`).

**Discovered via**: `backend_api`

### Registry

Checks the task app registry for registered apps (may not have active deployments).

**Discovered via**: `registry`

## Integration with Deploy Command

The scan command works seamlessly with the `deploy` command:

1. **Deploy a local service**:
   ```bash
   synth-ai deploy --runtime local --task-app examples/task_apps/banking77/banking77_task_app.py --port 8000 --env .env
   ```

2. **Scan discovers it**:
   ```bash
   synth-ai scan
   # Shows the deployed service with health status
   ```

3. **Deploy a tunnel**:
   ```bash
   synth-ai deploy --runtime tunnel --task-app examples/task_apps/banking77/banking77_task_app.py --port 8001 --env .env --tunnel-mode quick
   ```

4. **Scan discovers it**:
   ```bash
   synth-ai scan --json
   # Shows both local service and tunnel
   ```

## Error Handling

The scan command handles errors gracefully:

- **Invalid port ranges**: Returns error with helpful message
- **Network errors**: Marks apps as "unknown" status
- **Missing API keys**: Attempts health checks without authentication (may fail for protected apps)
- **Stale records**: Automatically cleans up records for processes that are no longer running

## Performance

- **Concurrent scanning**: Uses asyncio for parallel port and health checks
- **Configurable concurrency**: Defaults to 20 concurrent port checks, 10 concurrent health checks
- **Timeout handling**: Configurable timeout prevents hanging on unresponsive services
- **Efficient discovery**: Prioritizes service/tunnel records (fast) over port scanning (slower)

## Use Cases

### For Developers

- Quickly see what task apps are running locally
- Verify deployments are healthy
- Debug connectivity issues
- Monitor multiple services

### For CI/CD and Automation

- Programmatically discover deployed services
- Health check endpoints for monitoring
- Integration with deployment pipelines
- Service discovery for testing

### For AI Agents

- Structured JSON output for programmatic consumption
- Reliable discovery of available services
- Health status for service selection
- Metadata for service configuration

## Troubleshooting

### No apps found

- Check that services are actually running
- Verify port range includes the ports your apps use
- Check that apps have `/health` endpoints
- Ensure API key is set if apps require authentication

### Apps showing as "unknown"

- Check network connectivity
- Verify apps are responding on expected ports
- Increase timeout with `--timeout` option
- Check firewall settings

### Tunnel apps not discovered

- Verify `cloudflared` processes are running
- Check tunnel records file exists (`~/.synth-ai/services.json`)
- Ensure tunnels were deployed via `synth-ai deploy`
- Try `--verbose` to see discovery progress

## See Also

- `synth-ai deploy`: Deploy task apps locally or via tunnels
- `synth-ai task-app serve`: Serve a task app directly
- Task App Standards: Documentation on task app structure and endpoints




