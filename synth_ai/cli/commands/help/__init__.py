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

PROMPT_LEARNING_HELP = """
Optimize prompts using evolutionary algorithms (GEPA) or meta-learning (MIPRO).

OVERVIEW
--------
Prompt Learning automatically discovers better prompts for your LLM task by:
  • Running your task app with different prompt variations
  • Measuring performance on a dataset of examples (seeds)
  • Evolving prompts to maximize reward/accuracy

Two algorithms are available:
  • GEPA: Genetic Evolution of Prompt Architectures - evolutionary optimization
         with crossover, mutation, and selection across generations
  • MIPRO: Meta-learning with bootstrap phase and TPE optimization

QUICK START
-----------
  # Interactive mode (prompts for config)
  uvx synth-ai train --type prompt

  # With TOML config file
  uvx synth-ai train my_config.toml

  # With explicit options
  uvx synth-ai train --type prompt --backend http://localhost:8000 --poll

EXAMPLE TOML CONFIG (GEPA)
--------------------------
  [prompt_learning]
  algorithm = "gepa"
  task_app_id = "banking77"
  task_app_url = "https://your-tunnel.trycloudflare.com"

  [prompt_learning.initial_prompt]
  id = "banking77_pattern"
  name = "Banking77 Classification"

  [[prompt_learning.initial_prompt.messages]]
  role = "system"
  pattern = "You are an expert assistant that classifies queries."
  order = 0

  [[prompt_learning.initial_prompt.messages]]
  role = "user"
  pattern = "Query: {query}\\n\\nClassify this query."
  order = 1

  [prompt_learning.initial_prompt.wildcards]
  query = "REQUIRED"

  [prompt_learning.policy]
  model = "gpt-4o-mini"
  provider = "openai"
  inference_mode = "synth_hosted"
  temperature = 0.0
  max_completion_tokens = 256

  [prompt_learning.gepa]
  env_name = "banking77"

  [prompt_learning.gepa.rollout]
  budget = 500
  max_concurrent = 20
  minibatch_size = 10

  [prompt_learning.gepa.evaluation]
  seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Training examples
  validation_seeds = [10, 11, 12, 13, 14]  # Held-out validation

  [prompt_learning.gepa.population]
  initial_size = 5
  num_generations = 3
  children_per_generation = 2

EXAMPLE TOML CONFIG (MIPRO)
---------------------------
  [prompt_learning]
  algorithm = "mipro"
  task_app_id = "banking77"
  task_app_url = "http://127.0.0.1:8102"

  [prompt_learning.initial_prompt]
  id = "banking77_pattern"
  name = "Banking77 Classification Pattern"

  [[prompt_learning.initial_prompt.messages]]
  role = "system"
  pattern = "You are an expert banking assistant."
  order = 0

  [prompt_learning.policy]
  model = "gpt-4o-mini"
  provider = "openai"
  inference_mode = "synth_hosted"

  [prompt_learning.mipro]
  env_name = "banking77"
  num_iterations = 10
  batch_size = 16
  max_concurrent = 20
  proposer_effort = "LOW"
  bootstrap_train_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  online_pool = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  test_pool = [20, 21, 22, 23, 24]

KEY CONCEPTS
------------
Task App:
  Your FastAPI server that receives rollout requests and returns rewards.
  Create one with: uvx synth-ai task-app init

Seeds:
  Integer IDs that identify examples in your dataset. The optimizer
  evaluates prompts on these seeds to measure performance.

Initial Prompt:
  Starting prompt with wildcards like {query} that get filled at runtime.
  GEPA/MIPRO evolve variations of this prompt.

Policy:
  LLM configuration (model, provider, temperature) used during rollouts.

Proposer Effort (GEPA/MIPRO):
  Controls which model generates prompt mutations:
  • LOW_CONTEXT: Fastest/cheapest (gpt-oss-120b)
  • LOW: Good balance (gpt-4o-mini)
  • MEDIUM: Higher quality (gpt-4o)
  • HIGH: Best quality but expensive (gpt-5)

CLI OPTIONS
-----------
  --type prompt           Explicitly set training type to prompt learning
  --backend URL           Backend URL (default: https://api.usesynth.ai)
  --local-backend         Use http://localhost:8000
  --poll                  Wait for job completion
  --poll-timeout SECS     Max wait time (default: 3600)
  --stream-format cli|chart  Output style (cli=lines, chart=live panel)
  --tui                   Enable live TUI dashboard
  --show-curve            Show optimization curve at end
  --verbose-summary       Show detailed final summary

WORKFLOW
--------
  1. Create a task app:
       uvx synth-ai task-app init my-app
       cd my-app && uvx synth-ai deploy --runtime=uvicorn

  2. Create TOML config with your prompt and seeds

  3. Run optimization:
       uvx synth-ai train my_config.toml --poll --show-curve

  4. Retrieve optimized prompt:
       uvx synth-ai artifacts list --type prompt
       uvx synth-ai artifacts show <job_id>

DEMOS
-----
  Banking77 (classification):
    cd synth-ai/demos/gepa_banking77
    python run_demo.py

  Crafter VLM (vision agent):
    cd synth-ai/demos/gepa_crafter_vlm
    python run_notebook.py

  Image Style Matching (GraphGen):
    cd synth-ai/demos/image_style_matching
    python run_notebook.py

TROUBLESHOOTING
---------------
1. "Task app not reachable"
   → Ensure your task app is running and accessible
   → For local dev: uvx synth-ai deploy --runtime=uvicorn --port 8102
   → For training: Use Cloudflare tunnel or deploy to Modal

2. "ENVIRONMENT_API_KEY required"
   → Run: uvx synth-ai setup
   → Or set ENVIRONMENT_API_KEY env var

3. "No improvement after N generations"
   → Increase rollout budget
   → Increase num_generations or children_per_generation
   → Try different initial prompt
   → Check if task app rewards are calibrated correctly

4. "Job failed with timeout"
   → Increase --poll-timeout
   → Check backend health: curl https://api.usesynth.ai/health

5. "Invalid TOML config"
   → Validate with: uvx synth-ai train my_config.toml --dry-run
   → Check for deprecated fields (meta_model, etc.)

ENVIRONMENT VARIABLES
---------------------
  SYNTH_API_KEY           Your Synth platform API key
  ENVIRONMENT_API_KEY     Task environment authentication
  BACKEND_BASE_URL        Override backend URL
  SDK_EXPERIMENTAL        Enable experimental models

For more information: https://docs.usesynth.ai/prompt-learning
"""

COMMAND_HELP = {
    "deploy": DEPLOY_HELP,
    "setup": SETUP_HELP,
    "prompt": PROMPT_LEARNING_HELP,
    "prompt-learning": PROMPT_LEARNING_HELP,
    "train": PROMPT_LEARNING_HELP,
}


def get_command_help(command: str) -> str | None:
    """Get detailed help text for a command."""
    return COMMAND_HELP.get(command)


__all__ = ["DEPLOY_HELP", "SETUP_HELP", "PROMPT_LEARNING_HELP", "COMMAND_HELP", "get_command_help"]

