"""Help content for CLI commands."""

DEPLOY_HELP = """
Deploy a Synth AI task app locally, to Modal, or via Cloudflare Tunnel.

OVERVIEW
--------
The deploy command supports three runtimes:
  • modal: Deploy to Modal's cloud platform (default)
  • local: Run locally with FastAPI/Uvicorn
  • tunnel: Expose local app via Cloudflare Tunnel (for RL/prompt optimization)

BASIC USAGE
-----------
  # Deploy to Modal (production)
  uvx synth-ai deploy

  # Deploy specific task app
  uvx synth-ai deploy my-math-app

  # Run locally for development
  uvx synth-ai deploy --runtime=local --port 8001

  # Deploy via Cloudflare Tunnel (for training)
  uvx synth-ai deploy --runtime=tunnel --tunnel-mode quick

MODAL DEPLOYMENT
----------------
Modal deployment requires:
  1. Modal authentication (run: modal token new)
  2. ENVIRONMENT_API_KEY (run: uvx synth-ai setup)

Options:
  --modal-mode [deploy|serve]  Use 'deploy' for production (default), 
                                'serve' for ephemeral development
  --name TEXT                   Override Modal app name
  --dry-run                     Preview the deploy command without executing
  --env-file PATH               Env file(s) to load (can be repeated)

Examples:
  # Standard production deployment
  uvx synth-ai deploy --runtime=modal

  # Deploy with custom name
  uvx synth-ai deploy --runtime=modal --name my-task-app-v2

  # Preview deployment command
  uvx synth-ai deploy --dry-run

  # Deploy with custom env file
  uvx synth-ai deploy --env-file .env.production

LOCAL DEVELOPMENT
-----------------
Run locally with auto-reload and tracing:

  uvx synth-ai deploy --runtime=uvicorn --port 8001 --reload

Options:
  --host TEXT                   Bind address (default: 0.0.0.0)
  --port INTEGER                Port number (prompted if not provided)
  --reload/--no-reload          Enable auto-reload on code changes
  --force/--no-force            Kill existing process on port
  --trace PATH                  Enable tracing to directory (default: traces/v3)
  --trace-db PATH               SQLite DB for traces

Examples:
  # Basic local server
  uvx synth-ai deploy --runtime=uvicorn

  # Development with auto-reload
  uvx synth-ai deploy --runtime=uvicorn --reload --port 8001

  # With custom trace directory
  uvx synth-ai deploy --runtime=uvicorn --trace ./my-traces

TROUBLESHOOTING
---------------
Common issues:

1. "ENVIRONMENT_API_KEY is required"
   → Run: uvx synth-ai setup

2. "Modal CLI not found"
   → Install: pip install modal
   → Authenticate: modal token new

3. "Task app not found"
   → Check app_id matches your task_app.py configuration
   → Run: uvx synth-ai task-app list (if available)

4. "Port already in use" (local/tunnel)
   → Use --force to kill existing process
   → Or specify different --port

5. "cloudflared not found" (tunnel)
   → Install: brew install cloudflare/cloudflare/cloudflared
   → Or: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/

6. "No env file discovered"
   → Create .env file with required keys
   → Or pass --env-file explicitly

ENVIRONMENT VARIABLES
---------------------
  SYNTH_API_KEY              Your Synth platform API key
  ENVIRONMENT_API_KEY        Task environment authentication
  TASK_APP_BASE_URL          Base URL for deployed task app
  DEMO_DIR                   Demo directory path
  SYNTH_DEMO_DIR             Alternative demo directory

For more information: https://docs.usesynth.ai/deploy
"""

SETUP_HELP = """
Configure Synth AI credentials and environment.

OVERVIEW
--------
The setup command initializes your Synth AI environment by:
  1. Authenticating with the Synth platform via browser
  2. Saving your API keys to ~/.synth/config
  3. Verifying Modal authentication (for deployments)
  4. Testing connectivity to backend services

USAGE
-----
  uvx synth-ai setup

The command will:
  • Open your browser for authentication (or prompt for manual entry)
  • Save SYNTH_API_KEY and ENVIRONMENT_API_KEY
  • Verify Modal is authenticated
  • Test backend connectivity

WHAT YOU'LL NEED
----------------
  • Web browser for authentication
  • Modal account (for deployments): https://modal.com
  • Active internet connection

TROUBLESHOOTING
---------------
1. "Failed to fetch keys from frontend"
   → You'll be prompted to enter keys manually
   → Get keys from: https://www.usesynth.ai/dashboard/settings

2. "Modal authentication status: not authenticated"
   → Run: modal token new
   → Then re-run: uvx synth-ai setup

3. Browser doesn't open
   → Check your default browser settings
   → Or enter keys manually when prompted

WHERE ARE KEYS STORED?
----------------------
Keys are saved to: ~/.synth/config

This file is read automatically by all Synth AI commands.
You can also use .env files in your project directory.

NEXT STEPS
----------
After setup completes:
  1. Deploy your task app: uvx synth-ai deploy
  2. Start local development: uvx synth-ai deploy --runtime=uvicorn
  3. Run training: uvx synth-ai train

For more information: https://docs.usesynth.ai/setup
"""

COMMAND_HELP = {
    "deploy": DEPLOY_HELP,
    "setup": SETUP_HELP,
}


def get_command_help(command: str) -> str | None:
    """Get detailed help text for a command."""
    return COMMAND_HELP.get(command)


__all__ = ["DEPLOY_HELP", "SETUP_HELP", "COMMAND_HELP", "get_command_help"]

