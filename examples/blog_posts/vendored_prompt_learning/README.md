# Vendored Prompt Learning Examples

This directory contains **production-ready examples** for optimizing prompts on the fly using Synth AI's GEPA and MIPRO algorithms.

## 🎯 Purpose: Prompt Optimization on the Fly in Production

These examples demonstrate how to **automatically optimize prompts in production** without manual intervention:

- **A/B Testing**: Automatically find better prompts for your use case
- **Performance Tuning**: Continuously improve prompt performance as your data changes
- **Multi-Tenant Optimization**: Optimize prompts per customer or use case
- **Rapid Iteration**: Test and deploy better prompts faster than manual tuning

**Key Features:**
- ✅ **Automated**: No manual prompt engineering required
- ✅ **Production-Ready**: In-process task apps with automatic tunnel management
- ✅ **Fast**: Minimal budgets for quick testing (~1 minute)
- ✅ **Complete Pipeline**: Baseline → Optimization → Final Evaluation
- ✅ **Self-Contained**: Everything in one script, no external dependencies

**Note**: This directory consolidates **all** code from `blog_posts/gepa/` and `blog_posts/mipro/` into a unified, easy-to-use format. See [CONSOLIDATION.md](CONSOLIDATION.md) for details.

📖 **For production integration guide, see [PRODUCTION_PROMPT_OPTIMIZATION.md](PRODUCTION_PROMPT_OPTIMIZATION.md)**

## Directory Structure

```
vendored_prompt_learning/
├── README.md                    # This file
├── CONSOLIDATION.md            # Consolidation notes
├── task_app.py                 # Task app helper for HeartDisease
├── run_gepa_example.py        # Complete GEPA pipeline (recommended)
├── run_mipro_example.py       # Complete MIPRO pipeline (recommended)
├── configs/                    # All configuration files
│   ├── banking77_*.toml       # Banking77 configs
│   ├── heartdisease_*.toml    # HeartDisease configs
│   ├── hotpotqa_*.toml        # HotpotQA configs
│   └── ...                    # Other benchmarks
├── scripts/                    # All Python and shell scripts
│   ├── run_*.py               # Python scripts
│   ├── run_*.sh               # Shell scripts
│   ├── deploy_*.sh            # Deployment scripts
│   └── ...                    # Other utilities
├── docs/                       # Documentation files
│   ├── README.md              # Original GEPA README
│   ├── HEARTDISEASE_DEMO.md   # HeartDisease demo guide
│   └── ...                    # Other docs
└── results/                    # Historical results (optional)
```

## Quick Start

### Prerequisites

1. **Environment Variables** (in `.env` file):
   ```bash
   GROQ_API_KEY=your_groq_key          # For policy model
   OPENAI_API_KEY=your_openai_key      # For meta-model (MIPRO) and mutation LLM (GEPA)
   SYNTH_API_KEY=test                  # Backend API key
   ENVIRONMENT_API_KEY=test            # Task app authentication
   BACKEND_BASE_URL=http://localhost:8000  # Backend URL
   ```

2. **Backend Running**: Make sure the synth-ai backend is running on `localhost:8000` (or your `BACKEND_BASE_URL`)

3. **Cloudflare Tunnel** (for production): The `cloudflared` binary will be auto-installed if missing

### Recommended: Complete Pipeline Examples

These are the **recommended** scripts that show the full pipeline (baseline → optimization → final eval):

#### GEPA Example

```bash
# Local development (no tunnel)
SYNTH_TUNNEL_MODE=local uv run run_gepa_example.py

# Production (with Cloudflare tunnel)
uv run run_gepa_example.py
```

#### MIPRO Example

```bash
# Local development (no tunnel)
SYNTH_TUNNEL_MODE=local uv run run_mipro_example.py

# Production (with Cloudflare tunnel)
uv run run_mipro_example.py
```

### Other Available Scripts

All scripts from the original `gepa/` and `mipro/` directories are available in `scripts/`:

- **In-Process Scripts**: `scripts/run_fully_in_process.py`, `scripts/run_mipro_in_process.py`
- **Banking77 Scripts**: `scripts/run_gepa_banking77.sh`, `scripts/run_mipro_banking77.sh`
- **Pipeline Scripts**: `scripts/run_gepa_banking77_pipeline.sh`, `scripts/run_mipro_banking77_pipeline.sh`
- **Baseline Scripts**: `scripts/gepa_baseline.py`, `scripts/heartdisease_baseline.py`
- **Deployment Scripts**: `scripts/deploy_banking77_task_app.sh`, etc.

## Architecture

### In-Process Task App

The examples use `InProcessTaskApp` which automatically:

- Starts a FastAPI server in a background thread
- Opens a Cloudflare tunnel (or uses localhost in dev mode)
- Provides the tunnel URL for optimization jobs
- Cleans up everything on exit

```python
from synth_ai.task import InProcessTaskApp
from task_app import build_config

async with InProcessTaskApp(
    config_factory=build_config,
    port=8114,
    tunnel_mode="quick",  # or "local" for dev
) as task_app:
    # Use task_app.url for your optimization jobs
    print(f"Task app running at: {task_app.url}")
```

### Prompt Optimization Flow

```
┌─────────────────┐
│   Your Script   │ 1. Evaluate baseline
└────────┬────────┘ 2. Start task app
         │           3. Submit optimization job
         │           4. Poll for completion
         ▼
┌─────────────────┐
│  Synth Backend  │ 1. Receives optimization job
│  (GEPA/MIPRO)   │ 2. Registers prompts with interceptor
└────────┬────────┘ 3. Calls task app /rollout endpoint
         │
         │ POST /rollout
         ▼
┌─────────────────┐
│   Task App      │ 1. Receives rollout request
│  (In-Process)   │ 2. Builds baseline messages (NO prompts!)
└────────┬────────┘ 3. Calls inference_url (interceptor)
         │
         │ inference_url = http://interceptor/v1/{job_id}/chat/completions
         ▼
┌─────────────────┐
│  Interceptor    │ 1. Receives baseline messages
│                 │ 2. Applies prompt transformation
│                 │ 3. Forwards to real LLM (Groq/OpenAI/etc)
└────────┬────────┘ 4. Returns response
         │
         │ Response with optimized prompt
         ▼
┌─────────────────┐
│   Task App      │ 1. Evaluates response
│                 │ 2. Computes score/metrics
└────────┬────────┘ 3. Returns to backend
         │
         │ Metrics (accuracy, reward, etc.)
         ▼
┌─────────────────┐
│  Synth Backend  │ 1. Updates optimization state
│                 │ 2. Generates new candidates
└─────────────────┘ 3. Repeats until complete
```

### Key Concepts

- **Task App**: Your application that runs evaluations (HeartDisease classification in these examples)
- **Optimizer**: GEPA or MIPRO algorithm that optimizes prompts
- **Interceptor**: Proxy server that injects optimized prompts into LLM calls
- **Rollout**: Single evaluation run on a specific seed/task instance

## Benchmarks

### HeartDisease

- **Task**: Classify patients as having heart disease (1) or not (0)
- **Dataset**: `buio/heart-disease` from HuggingFace
- **Baseline Performance**: ~54% accuracy
- **Expected Improvement**: GEPA/MIPRO typically achieve 70-75% accuracy

### Banking77

- **Task**: Intent classification (77 banking intents)
- **Configs**: `configs/banking77_*.toml`
- **Scripts**: `scripts/run_gepa_banking77.sh`, `scripts/run_mipro_banking77.sh`

### Other Benchmarks

- **HotpotQA**: Multi-hop question answering
- **IFBench**: Instruction following benchmark
- **HoVer**: Claim verification against Wikipedia
- **PUPA**: Privacy-aware task delegation

See `configs/` for all available benchmark configurations.

## Customization

### Changing the Benchmark

To use a different benchmark:

1. **Use existing config**: Check `configs/` for your benchmark
2. **Update script**: Modify the config path in your script
3. **Update task app**: Change `task_app.py` to import your benchmark's task app

### Adjusting Optimization Parameters

See the config files in `configs/` for examples of how to adjust:
- Rollout budgets
- Population sizes
- Mutation rates
- Seed pools
- Model configurations

## Troubleshooting

### Task App Not Starting

- Check that port 8114 is available (or change `port` parameter)
- Verify `ENVIRONMENT_API_KEY` is set
- Check logs for import errors

### Tunnel Issues

- For local development, use `SYNTH_TUNNEL_MODE=local`
- For production, ensure `cloudflared` is installed (auto-installed if missing)
- Check firewall/network settings

### Backend Connection Errors

- Verify backend is running: `curl http://localhost:8000/health`
- Check `BACKEND_BASE_URL` environment variable
- Verify `SYNTH_API_KEY` is correct

### Path Issues

All scripts have been updated to use paths relative to `vendored_prompt_learning/`. If you encounter path issues:
- Make sure you're running scripts from the `vendored_prompt_learning/` directory
- Check that configs are in `configs/` relative to the script location
- Update any hardcoded paths in scripts if needed

## Migration from Old Directories

If you were using scripts from `blog_posts/gepa/` or `blog_posts/mipro/`:

1. **Update paths**: All config paths now point to `vendored_prompt_learning/configs/`
2. **Update imports**: Scripts use relative paths from `vendored_prompt_learning/`
3. **Use new scripts**: Consider using `run_gepa_example.py` and `run_mipro_example.py` for the complete pipeline

See [CONSOLIDATION.md](CONSOLIDATION.md) for detailed migration notes.

## References

- [GEPA Documentation](../../../docs/gepa.md)
- [MIPRO Documentation](../../../docs/mipro.md)
- [Task App Integration Guide](../../../docs/task-app-integration.md)
- [In-Process Task App Guide](../../../docs/in-process-task-app.md)
