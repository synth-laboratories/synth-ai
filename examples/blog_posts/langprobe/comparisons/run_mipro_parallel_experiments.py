#!/usr/bin/env python3
"""Run MIPRO experiments via experiment queue for Banking77 and other tasks.

This script:
- Checks that Redis and queue worker are running
- Submits experiments to the experiment queue for MIPRO optimization
- Polls for status until experiments complete
- Returns results at the end
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None  # Will fail gracefully if requests not available

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

# Load .env file
try:
    from dotenv import load_dotenv, find_dotenv
    
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # Try repo root
        REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
        repo_root_env = REPO_ROOT / ".env"
        if repo_root_env.exists():
            env_path = str(repo_root_env)
    
    if env_path:
        load_dotenv(env_path, override=False)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print("⚠️  No .env file found - environment variables may not be set")
except ImportError:
    print("⚠️  python-dotenv not available - environment variables may not be loaded")
except Exception as e:
    print(f"⚠️  Failed to load .env file: {e}")

# Load configuration from YAML file (source of truth for benchmarks + models + configs)
COMPARISONS_DIR = Path(__file__).parent
CONFIG_FILE = COMPARISONS_DIR / "synth_mipro_config.yaml"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def check_redis() -> bool:
    """Check if Redis is running."""
    try:
        import redis
        broker_url = os.getenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
        r = redis.from_url(broker_url)
        r.ping()
        print("✅ Redis is running")
        return True
    except ImportError:
        print("⚠️  redis-py not available, skipping Redis check")
        return True  # Assume it's running
    except Exception as e:
        print(f"❌ Redis check failed: {e}")
        print("   Make sure Redis is running: brew services start redis")
        return False


def check_queue_worker() -> bool:
    """Check if experiment queue worker is running."""
    try:
        from synth_ai.cli.queue import _get_running_workers
        
        workers = _get_running_workers()
        if not workers:
            print("❌ No experiment queue workers running")
            print("   Start a worker with: synth-ai queue start")
            return False
        
        print(f"✅ Found {len(workers)} queue worker(s) running")
        for i, worker in enumerate(workers, 1):
            print(f"   Worker {i}: PID {worker['pid']}, DB: {worker['db_path']}")
        return True
    except Exception as e:
        print(f"❌ Failed to check queue worker: {e}")
        return False


def load_yaml_config() -> Dict:
    """Load configuration from YAML file.
    
    Returns:
        Dict with:
            - benchmarks: Dict mapping benchmark name to config dict
            - defaults: Dict with global default values
    """
    assert CONFIG_FILE.exists(), f"CRITICAL: Config file not found: {CONFIG_FILE}"
    
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    assert config is not None, f"CRITICAL: Failed to parse YAML config: {CONFIG_FILE}"
    
    benchmarks = config.get("benchmarks", {})
    defaults = config.get("defaults", {})
    
    # Validate benchmarks exist
    if not benchmarks:
        raise ValueError(f"No benchmarks defined in {CONFIG_FILE}")
    
    # Resolve config paths relative to repo root
    # Filter by 'run' field: only include benchmarks with run=true (or missing, default True)
    resolved_benchmarks = {}
    for benchmark_name, benchmark_config in benchmarks.items():
        # Check if run field is explicitly False (skip if False, include if True or missing)
        run_flag = benchmark_config.get("run", True)  # Default to True if not specified
        if not run_flag:
            continue
        
        config_path_str = benchmark_config.get("config_path")
        if not config_path_str:
            raise ValueError(f"benchmark '{benchmark_name}' missing 'config_path'")
        
        # Resolve path relative to repo root
        config_path = (REPO_ROOT / config_path_str).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found for benchmark '{benchmark_name}': {config_path}")
        
        resolved_benchmarks[benchmark_name] = {
            **benchmark_config,
            "config_path": str(config_path),
        }
    
    return {
        "benchmarks": resolved_benchmarks,
        "defaults": defaults,
    }


def _prepare_experiment_request(
    benchmark_name: str,
    benchmark_config: Dict,
    defaults: Dict,
) -> Tuple[str, Any, Path, int, Optional[Dict]]:
    """Prepare experiment request for a single benchmark.
    
    Returns:
        Tuple of (display_name, request, config_path, rollout_limit, model_config)
    """
    from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest, ExperimentJobSpec
    from synth_ai.experiment_queue.models import ExperimentJobType
    
    config_path = Path(benchmark_config["config_path"])
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Get per-benchmark rollout limit (required)
    rollout_limit = benchmark_config.get("rollout_limit")
    if rollout_limit is None:
        raise ValueError(f"benchmark '{benchmark_name}' missing 'rollout_limit'")
    
    # Merge defaults with per-benchmark overrides
    time_limit = benchmark_config.get("time_limit_seconds", defaults.get("time_limit_seconds", 600))
    # Increase max_trials to 1000 so it doesn't stop early (MIPRO evaluates trials, not just rollouts)
    # Each trial can involve multiple rollouts, so we need a high trial limit to reach rollout_limit
    max_trials = benchmark_config.get("max_trials", defaults.get("max_trials", 1000))
    max_cost_usd = benchmark_config.get("max_cost_usd", defaults.get("max_cost_usd", 10.0))
    
    # Build config overrides
    config_overrides = {
        # Algorithm override (ensure it's set to mipro)
        "prompt_learning.algorithm": "mipro",
        # Termination config (for termination conditions)
        "prompt_learning.termination_config.max_rollouts": rollout_limit,
        "prompt_learning.termination_config.max_seconds": time_limit,
        "prompt_learning.termination_config.max_trials": max_trials,
        "prompt_learning.termination_config.max_cost_usd": max_cost_usd,
        # MIPRO rollout budget (for MIPRO internal tracking - must match max_rollouts)
        "prompt_learning.mipro.max_rollouts": rollout_limit,
    }
    
    # Debug: Print override to verify it's set correctly
    print(f"  [DEBUG] Setting rollout budget override: {config_overrides['prompt_learning.mipro.max_rollouts']} (from YAML limit: {rollout_limit})")
    
    # Apply model overrides if specified
    model_config = benchmark_config.get("model")
    if model_config:
        if "provider" in model_config:
            config_overrides["prompt_learning.policy.provider"] = model_config["provider"]
            print(f"  [DEBUG] Setting provider override: {config_overrides['prompt_learning.policy.provider']}")
        if "model" in model_config:
            config_overrides["prompt_learning.policy.model"] = model_config["model"]
            print(f"  [DEBUG] Setting model override: {config_overrides['prompt_learning.policy.model']}")
    
    # Apply proposer_mode override if specified (for MIPRO: "dspy" or "synth")
    proposer_mode = benchmark_config.get("proposer_mode")
    if proposer_mode:
        config_overrides["prompt_learning.mipro.proposer_mode"] = proposer_mode
        print(f"  [DEBUG] Setting proposer_mode override: {proposer_mode}")
    
    # Apply any other per-benchmark overrides (flatten nested dicts with dot notation)
    for key, value in benchmark_config.items():
        if key not in ("config_path", "rollout_limit", "time_limit_seconds", "max_trials", 
                      "max_cost_usd", "model", "proposer_mode"):
            # Assume it's a config override path (e.g., "prompt_learning.mipro.num_iterations")
            if isinstance(value, dict):
                # Flatten nested dicts
                for nested_key, nested_value in value.items():
                    config_overrides[f"{key}.{nested_key}"] = nested_value
            else:
                config_overrides[key] = value
    
    # Format benchmark name for display (capitalize first letter)
    display_name = benchmark_name[0].upper() + benchmark_name[1:] if benchmark_name else benchmark_name
    
    # Get parallelism from config or default to 1 (one job per experiment)
    parallelism = benchmark_config.get("parallelism", 1)
    if parallelism < 1:
        parallelism = 1
    
    request = ExperimentSubmitRequest(
        name=f"MIPRO {display_name}",
        description=f"MIPRO optimization for {display_name}",
        parallelism=parallelism,
        jobs=[
            ExperimentJobSpec(
                job_type=ExperimentJobType.MIPRO,
                config_path=str(config_path),
                config_overrides=config_overrides,
            )
        ],
    )
    
    # ASSERT: Verify critical overrides are set correctly
    job_spec = request.jobs[0]
    assert job_spec.config_overrides is not None, f"config_overrides must be set for {benchmark_name}"
    
    # Assert rollout limit is set correctly
    rollout_budget_key = "prompt_learning.mipro.max_rollouts"
    max_rollouts_key = "prompt_learning.termination_config.max_rollouts"
    assert rollout_budget_key in job_spec.config_overrides, (
        f"Missing {rollout_budget_key} override for {benchmark_name}"
    )
    assert job_spec.config_overrides[rollout_budget_key] == rollout_limit, (
        f"Rollout budget mismatch for {benchmark_name}: "
        f"expected {rollout_limit}, got {job_spec.config_overrides[rollout_budget_key]}"
    )
    if max_rollouts_key in job_spec.config_overrides:
        assert job_spec.config_overrides[max_rollouts_key] == rollout_limit, (
            f"Max rollouts mismatch for {benchmark_name}: "
            f"expected {rollout_limit}, got {job_spec.config_overrides[max_rollouts_key]}"
        )
    
    # Assert model overrides are set correctly if specified
    if model_config:
        expected_provider = model_config.get("provider")
        expected_model = model_config.get("model")
        if expected_provider:
            provider_key = "prompt_learning.policy.provider"
            assert provider_key in job_spec.config_overrides, (
                f"Missing {provider_key} override for {benchmark_name}"
            )
            assert job_spec.config_overrides[provider_key] == expected_provider, (
                f"Provider mismatch for {benchmark_name}: "
                f"expected {expected_provider}, got {job_spec.config_overrides[provider_key]}"
            )
        if expected_model:
            model_key = "prompt_learning.policy.model"
            assert model_key in job_spec.config_overrides, (
                f"Missing {model_key} override for {benchmark_name}"
            )
            assert job_spec.config_overrides[model_key] == expected_model, (
                f"Model mismatch for {benchmark_name}: "
                f"expected {expected_model}, got {job_spec.config_overrides[model_key]}"
            )
    
    # CRITICAL: Validate merged config BEFORE submitting to queue
    # This catches errors early (e.g., missing bootstrap_train_seeds, online_pool)
    try:
        from synth_ai.experiment_queue.config_utils import prepare_config_file
        from synth_ai.api.train.validators import validate_prompt_learning_config
        
        # Apply overrides and validate merged config
        prepared = prepare_config_file(config_path, config_overrides)
        try:
            # Load merged config and validate
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            
            with open(prepared.path, "rb") as f:
                merged_config = tomllib.load(f)
            
            # Validate using SDK validator (same as backend will use)
            validate_prompt_learning_config(merged_config, prepared.path)
            
            # Additional MIPRO-specific validation: verify required fields exist after merge
            pl_section = merged_config.get("prompt_learning", {})
            mipro_section = pl_section.get("mipro", {})
            
            bootstrap_seeds = (
                mipro_section.get("bootstrap_train_seeds") or 
                pl_section.get("bootstrap_train_seeds")
            )
            online_pool = (
                mipro_section.get("online_pool") or 
                pl_section.get("online_pool")
            )
            
            if not bootstrap_seeds:
                raise ValueError(
                    f"MIPRO config validation failed for {benchmark_name}: "
                    f"bootstrap_train_seeds is missing after applying overrides. "
                    f"Ensure it's set in TOML or provided via override 'prompt_learning.mipro.bootstrap_train_seeds'"
                )
            if not online_pool:
                raise ValueError(
                    f"MIPRO config validation failed for {benchmark_name}: "
                    f"online_pool is missing after applying overrides. "
                    f"Ensure it's set in TOML or provided via override 'prompt_learning.mipro.online_pool'"
                )
            
            print(f"  ✅ Config validation passed for {benchmark_name}")
        finally:
            prepared.cleanup()
    except Exception as e:
        raise ValueError(
            f"❌ Config validation failed for benchmark '{benchmark_name}' BEFORE submission. "
            f"This prevents invalid jobs from being queued.\n"
            f"Config: {config_path}\n"
            f"Overrides: {config_overrides}\n"
            f"Error: {e}"
        ) from e
    
    return display_name, request, config_path, rollout_limit, model_config


def submit_experiments(yaml_config: Dict) -> List[str]:
    """Submit experiments to the queue in parallel (one per benchmark in YAML config).
    
    Returns:
        List of experiment IDs
    """
    from synth_ai.experiment_queue.service import create_experiment
    
    benchmarks = yaml_config["benchmarks"]
    defaults = yaml_config["defaults"]
    
    experiment_ids = []
    
    print("\n" + "=" * 80)
    print("SUBMITTING MIPRO EXPERIMENTS")
    print("=" * 80)
    
    for benchmark_name, benchmark_config in benchmarks.items():
        try:
            display_name, request, config_path, rollout_limit, model_config = _prepare_experiment_request(
                benchmark_name, benchmark_config, defaults
            )
            
            print(f"\n📋 {display_name}:")
            print(f"   Config: {config_path}")
            print(f"   Rollout limit: {rollout_limit}")
            if model_config:
                provider = model_config.get("provider", "N/A")
                model = model_config.get("model", "N/A")
                print(f"   Model: {provider}/{model}")
            proposer_mode = benchmark_config.get("proposer_mode", "synth")
            print(f"   Proposer mode: {proposer_mode}")
            
            experiment = create_experiment(request)
            experiment_id = experiment.experiment_id
            experiment_ids.append(experiment_id)
            
            print(f"   ✅ Submitted: experiment_id={experiment_id}")
            
        except Exception as e:
            print(f"   ❌ Failed to submit {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Submitted {len(experiment_ids)} experiment(s)")
    return experiment_ids


def poll_experiment_status(experiment_id: str, timeout: int = 3600, poll_interval: float = 5.0) -> Optional[Dict]:
    """Poll experiment status until completion.
    
    Returns:
        Experiment dict with status, or None if timeout
    """
    from synth_ai.experiment_queue.service import fetch_experiment
    from synth_ai.experiment_queue.models import ExperimentStatus
    from synth_ai.experiment_queue.status import ExperimentStatus as StatusObj
    
    start_time = time.time()
    poll_start_time = time.time()
    first_rollout_time = None  # Track when first rollout completes for rate calculation
    
    while time.time() - start_time < timeout:
        try:
            experiment = fetch_experiment(experiment_id)
            if not experiment:
                print(f"❌ Experiment {experiment_id} not found")
                return None
            
            status = experiment.status
            
            # Get status details from first job if available
            status_line = ""
            if experiment.jobs:
                job = experiment.jobs[0]
                has_progress_data = False
                
                if job.status_json:
                    # Validate status_json structure
                    assert isinstance(job.status_json, dict), (
                        f"job.status_json must be dict, got {type(job.status_json).__name__}: {job.status_json}"
                    )
                    
                    try:
                        status_obj = StatusObj.from_dict(job.status_json)
                        assert isinstance(status_obj, StatusObj), (
                            f"from_dict returned wrong type: {type(status_obj).__name__}"
                        )
                        
                        # Extract rollout limit from config to override incorrect total_rollouts
                        rollout_limit_from_config = None
                        try:
                            import tomllib
                            from pathlib import Path
                            config_path = Path(job.config_path)
                            if config_path.exists():
                                with config_path.open("rb") as f:
                                    config_data = tomllib.load(f)
                                pl_config = config_data.get("prompt_learning", {})
                                mipro_config = pl_config.get("mipro", {})
                                rollout_limit_from_config = mipro_config.get("max_rollouts")
                        except Exception:
                            pass
                        
                        # Also try to get rollout_limit from job's config_overrides if available
                        if rollout_limit_from_config is None and job.config_overrides:
                            try:
                                max_rollouts_key = "prompt_learning.mipro.max_rollouts"
                                if max_rollouts_key in job.config_overrides:
                                    rollout_limit_from_config = job.config_overrides[max_rollouts_key]
                            except Exception:
                                pass
                        
                        # Override total_rollouts with config value if available and rollouts_completed exists
                        if rollout_limit_from_config is not None and status_obj.rollouts_completed is not None:
                            # Create a modified status object with correct total_rollouts
                            status_dict = status_obj.to_dict()
                            status_dict["total_rollouts"] = rollout_limit_from_config
                            status_obj = StatusObj.from_dict(status_dict)
                        
                        formatted = status_obj.format_status_line()
                        assert isinstance(formatted, str), (
                            f"format_status_line must return str, got {type(formatted).__name__}: {formatted}"
                        )
                        
                        if formatted and formatted != "No status available":
                            # Check if we have actual progress data (rollouts, ETA, etc.)
                            has_progress_data = (
                                status_obj.rollouts_completed is not None
                                or status_obj.eta_seconds is not None
                                or status_obj.best_score is not None
                            )
                            
                            # ✅ ADD: Check for validation phase in custom_fields or recent events
                            validation_info = None
                            if hasattr(status_obj, 'custom_fields') and status_obj.custom_fields:
                                phase = status_obj.custom_fields.get("phase")
                                if phase == "validation":
                                    validation_candidate = status_obj.custom_fields.get("validation_candidate")
                                    validation_total = status_obj.custom_fields.get("validation_total")
                                    if validation_candidate and validation_total:
                                        validation_info = f" | 🔍 Validation: candidate {validation_candidate}/{validation_total}"
                            
                            # Track first rollout completion time for rate calculation
                            if status_obj.rollouts_completed is not None and status_obj.rollouts_completed > 0:
                                if first_rollout_time is None:
                                    first_rollout_time = time.time()
                            
                            # Calculate rollouts per minute if we have rollouts and elapsed time
                            rollouts_per_min_str = ""
                            if status_obj.rollouts_completed is not None and status_obj.rollouts_completed > 0:
                                # Calculate elapsed time since first rollout
                                if first_rollout_time is not None:
                                    elapsed_since_first = time.time() - first_rollout_time
                                    if elapsed_since_first > 0:
                                        rollouts_per_min = (status_obj.rollouts_completed / elapsed_since_first) * 60
                                        rollouts_per_min_str = f" | {rollouts_per_min:.1f} rollouts/min"
                                # Fallback: use job start time if first_rollout_time not set yet
                                elif job.started_at:
                                    started_at_ts = job.started_at.timestamp()
                                    elapsed = time.time() - started_at_ts
                                    if elapsed > 0:
                                        rollouts_per_min = (status_obj.rollouts_completed / elapsed) * 60
                                        rollouts_per_min_str = f" | {rollouts_per_min:.1f} rollouts/min"
                            
                            # ✅ ADD: Show validation info if available, otherwise show normal status
                            if validation_info:
                                status_line = f" | {formatted}{validation_info}"
                            else:
                                # Always show status_json if it has any data (even just policy/environment)
                                # This ensures we see policy/environment even before progress events arrive
                                status_line = f" | {formatted}{rollouts_per_min_str}"
                    except Exception as e:
                        # Fall back to basic status if parsing fails
                        pass
                
                # If no status_json or status_line is empty, show elapsed time and rollout limit as fallback
                if not status_line and status == ExperimentStatus.RUNNING:
                    # Show elapsed time (prefer job.started_at, fallback to job.created_at, then poll start time)
                    elapsed = 0.0
                    if job.started_at:
                        # Handle timezone-aware datetime
                        started_at_ts = job.started_at.timestamp()
                        current_ts = time.time()
                        elapsed = current_ts - started_at_ts
                        # Clamp to non-negative (in case of clock skew or timezone issues)
                        elapsed = max(0.0, elapsed)
                    elif job.created_at:
                        # Fallback to created_at if started_at is not set (shouldn't happen for RUNNING jobs, but be safe)
                        created_at_ts = job.created_at.timestamp()
                        current_ts = time.time()
                        elapsed = current_ts - created_at_ts
                        elapsed = max(0.0, elapsed)
                    else:
                        # Last resort: use poll start time (shouldn't happen for RUNNING jobs)
                        elapsed = time.time() - poll_start_time
                    
                    if elapsed >= 60:
                        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                    else:
                        elapsed_str = f"{int(elapsed)}s"
                    
                    status_line = f" | Running ({elapsed_str})"
                    
                    # Try to extract rollout limit from config as fallback
                    try:
                        import tomllib
                        from pathlib import Path
                        config_path = Path(job.config_path)
                        if config_path.exists():
                            with config_path.open("rb") as f:
                                config_data = tomllib.load(f)
                            pl_config = config_data.get("prompt_learning", {})
                            mipro_config = pl_config.get("mipro", {})
                            rollout_limit = mipro_config.get("max_rollouts")
                            if rollout_limit:
                                status_line += f" | Limit: {rollout_limit} rollouts"
                    except Exception:
                        pass
            
            print(f"[{experiment_id[:12]}...] Status: {status.value}{status_line}")
            
            if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELED):
                return {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "status": status.value,
                    "error": experiment.error,
                    "jobs": [
                        {
                            "job_id": job.job_id,
                            "status": job.status.value,
                            "error": job.error,
                        }
                        for job in experiment.jobs
                    ],
                }
            
            time.sleep(poll_interval)
        except Exception as e:
            print(f"   ⚠️  Error checking status (will retry): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(poll_interval)
    
    print(f"⏱️  Timeout waiting for experiment {experiment_id}")
    return None


def fetch_backend_job_details(backend_job_id: str) -> Optional[Dict]:
    """Fetch job details including artifacts and snapshots from backend API.
    
    Returns:
        Dict with job details, artifacts, and best_snapshot, or None if fetch fails
    """
    try:
        import requests
    except ImportError:
        return None
    
    import os
    
    backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    if not backend_url.endswith("/api"):
        backend_url = f"{backend_url}/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        return None
    
    url = f"{backend_url}/prompt-learning/online/jobs/{backend_job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception:
        return None


def extract_results(experiment_id: str) -> Dict[str, Any]:
    """Extract results from completed experiment.
    
    Returns:
        Dict with benchmark results
    """
    from synth_ai.experiment_queue.service import fetch_experiment
    
    experiment = fetch_experiment(experiment_id)
    if not experiment:
        return {"experiment_id": experiment_id, "status": "not_found"}
    
    results = {
        "experiment_id": experiment_id,
        "status": experiment.status.value if hasattr(experiment.status, "value") else str(experiment.status),
        "baseline_score": None,
        "best_score": None,
        "total_rollouts": None,
        "total_time": None,
        "eval_seeds_n": None,
        "trials_tried": None,
        "transformations_evaluated": None,
        "lift": None,
        "backend_job_id": None,  # Store for debugging
    }
    
    if experiment.status.value == "completed" if hasattr(experiment.status, "value") else experiment.status == "completed":
        # Extract results from job summaries and backend metadata
        if experiment.jobs:
            job = experiment.jobs[0]
            
            # PRIORITY 1: Check status_json first (queue worker stores stats here)
            status_json = getattr(job, "status_json", None) if hasattr(job, "status_json") else None
            if status_json and isinstance(status_json, dict):
                # Extract from status_json (queue worker stores stats here during execution)
                # Progress poller stores: best_score, baseline_score, total_rollouts, etc.
                if results.get("baseline_score") is None:
                    results["baseline_score"] = status_json.get("baseline_score")
                if results.get("best_score") is None:
                    results["best_score"] = (
                        status_json.get("best_score") or 
                        status_json.get("best_validation_score")
                    )
                if results.get("total_rollouts") is None:
                    results["total_rollouts"] = status_json.get("total_rollouts")
                if results.get("trials_tried") is None:
                    results["trials_tried"] = (
                        status_json.get("trials_tried") or
                        status_json.get("num_trials") or  # MIPRO uses num_trials
                        status_json.get("optimization_trials_evaluated")
                    )
                if results.get("eval_seeds_n") is None:
                    results["eval_seeds_n"] = (
                        status_json.get("eval_seeds_n") or
                        status_json.get("validation_rollouts_executed") or
                        status_json.get("reference_pool_size")  # Size of reference_pool used for validation
                    )
                if results.get("total_time") is None:
                    results["total_time"] = (
                        status_json.get("total_time_seconds") or
                        status_json.get("elapsed_seconds")
                    )
                if results.get("transformations_evaluated") is None:
                    results["transformations_evaluated"] = status_json.get("transformations_evaluated")
            
            # PRIORITY 2: Check job.result (contains ResultSummary data)
            job_result = getattr(job, "result", None) if hasattr(job, "result") else None
            if job_result and isinstance(job_result, dict):
                # Extract from result dict
                if results.get("baseline_score") is None:
                    results["baseline_score"] = job_result.get("baseline_score")
                if results.get("best_score") is None:
                    results["best_score"] = job_result.get("best_score") or job_result.get("best_validation_score")
                if results.get("total_rollouts") is None:
                    results["total_rollouts"] = job_result.get("total_rollouts")
                if results.get("total_time") is None:
                    results["total_time"] = job_result.get("total_time_seconds")
            
            # PRIORITY 3: Fetch backend job details if backend_job_id is available
            backend_data = None
            if hasattr(job, "backend_job_id") and job.backend_job_id:
                results["backend_job_id"] = job.backend_job_id  # Store for debugging
                backend_data = fetch_backend_job_details(job.backend_job_id)
            
            # PRIORITY 4: Check status_json.backend_job_data (stored during execution)
            if not backend_data and status_json and isinstance(status_json, dict):
                backend_data = status_json.get("backend_job_data")
            
            # PRIORITY 5: Check job.backend_job_data attribute (if exists)
            if not backend_data and hasattr(job, "backend_job_data") and job.backend_job_data:
                backend_data = job.backend_job_data
            
            # Extract from backend metadata if available (most complete source)
            if backend_data and isinstance(backend_data, dict):
                # backend_job_data structure: {"job": {"metadata": {"stats": {...}}}}
                backend_job = backend_data.get("job", {})
                backend_metadata = backend_job.get("metadata", {})
                backend_stats = backend_metadata.get("stats", {})
                
                # Also check top-level stats/metadata as fallback
                stats = backend_data.get("stats", {}) or backend_stats
                metadata = backend_data.get("metadata", {}) or backend_metadata
                
                # ✅ ADD: Check snapshot payload directly (MIPRO stores num_trials here)
                snapshot = backend_data.get("snapshot") or backend_data.get("best_snapshot")
                snapshot_payload = None
                if snapshot and isinstance(snapshot, dict):
                    snapshot_payload = snapshot.get("payload", {})
                elif backend_job.get("snapshot"):
                    snapshot_payload = backend_job["snapshot"].get("payload", {}) if isinstance(backend_job["snapshot"], dict) else None
                
                # Extract scores from stats or metadata (only if not already set)
                if results.get("baseline_score") is None:
                    results["baseline_score"] = (
                        backend_stats.get("baseline_score") or 
                        stats.get("baseline_score") or 
                        metadata.get("baseline_score")
                    )
                if results.get("best_score") is None:
                    results["best_score"] = (
                        backend_stats.get("best_score") or 
                        backend_stats.get("best_validation_score") or
                        stats.get("best_score") or 
                        metadata.get("best_score")
                    )
                # ✅ CRITICAL: Extract from snapshot_payload FIRST (most reliable for MIPRO)
                # This MUST run before checking stats_dict because snapshot_payload has the authoritative values
                if snapshot_payload and isinstance(snapshot_payload, dict):
                    # ✅ ALWAYS overwrite with snapshot_payload values (they're authoritative for MIPRO)
                    snapshot_trials = snapshot_payload.get("num_trials")
                    if snapshot_trials is not None:
                        results["trials_tried"] = snapshot_trials
                    
                    # ✅ CRITICAL: Always use snapshot_payload.total_rollouts (MIPRO's authoritative source)
                    # MIPRO tracks all rollouts in _total_rollouts (bootstrap + minibatch + full + validation)
                    snapshot_rollouts = snapshot_payload.get("total_rollouts")
                    if snapshot_rollouts is not None:
                        results["total_rollouts"] = snapshot_rollouts
                        print(f"  [DEBUG] Extracted total_rollouts from snapshot_payload: {snapshot_rollouts}")
                    else:
                        print(f"  [DEBUG] WARNING: snapshot_payload.total_rollouts is None. Available keys: {list(snapshot_payload.keys())[:20]}")
                    
                    snapshot_time = (
                        snapshot_payload.get("total_time_seconds") or
                        snapshot_payload.get("elapsed_seconds")
                    )
                    if snapshot_time is not None:
                        results["total_time"] = snapshot_time
                    
                    # Try to get eval_seeds_n from reference_pool or test_pool size
                    reference_pool = snapshot_payload.get("reference_pool", [])
                    test_pool = snapshot_payload.get("test_pool", [])
                    if reference_pool:
                        results["eval_seeds_n"] = len(reference_pool)
                    elif test_pool:
                        results["eval_seeds_n"] = len(test_pool)
                
                # ✅ CRITICAL: For MIPRO, total_rollouts should already be set from snapshot_payload above
                # Only fall back to stats if snapshot_payload didn't have it
                if results.get("total_rollouts") is None:
                    # MIPRO stores total_rollouts directly in snapshot_payload (not split into optimization/validation)
                    # Backend stats_dict should have total_rollouts copied from snapshot_payload
                    fallback_rollouts = (
                        backend_stats.get("total_rollouts") or
                        stats.get("total_rollouts")
                    )
                    if fallback_rollouts is not None:
                        results["total_rollouts"] = fallback_rollouts
                        print(f"  [DEBUG] Using fallback total_rollouts from stats: {fallback_rollouts}")
                    else:
                        # Last resort: try to extract from status_json or job metadata
                        print(f"  [DEBUG] WARNING: No total_rollouts found in snapshot_payload or stats. Checking status_json...")
                        if status_json and isinstance(status_json, dict):
                            status_rollouts = status_json.get("total_rollouts") or status_json.get("rollouts_completed")
                            if status_rollouts is not None:
                                results["total_rollouts"] = status_rollouts
                                print(f"  [DEBUG] Found total_rollouts in status_json: {status_rollouts}")
                    # NOTE: Don't sum optimization_rollouts_executed + validation_rollouts_executed for MIPRO
                    # because MIPRO doesn't track these separately - it only tracks _total_rollouts
                if results.get("total_time") is None:
                    results["total_time"] = (
                        backend_stats.get("total_time_seconds") or
                        stats.get("total_time_seconds") or
                        metadata.get("total_time_seconds")
                    )
                if results.get("eval_seeds_n") is None:
                    results["eval_seeds_n"] = (
                        backend_stats.get("eval_seeds_n") or
                        backend_stats.get("validation_rollouts_executed") or
                        stats.get("eval_seeds_n")
                    )
                if results.get("trials_tried") is None:
                    results["trials_tried"] = (
                        backend_stats.get("trials_tried") or
                        backend_stats.get("num_trials") or  # MIPRO uses num_trials
                        backend_stats.get("optimization_trials_evaluated") or
                        stats.get("trials_tried") or
                        stats.get("num_trials")
                    )
                if results.get("transformations_evaluated") is None:
                    results["transformations_evaluated"] = (
                        backend_stats.get("transformations_evaluated") or
                        stats.get("transformations_evaluated")
                    )
            
            # DEBUG: Log what we found for troubleshooting
            if results.get("baseline_score") is None and results.get("best_score") is None:
                # No scores found - log available data sources for debugging
                debug_info = {
                    "has_status_json": status_json is not None,
                    "has_job_result": job_result is not None,
                    "has_backend_job_id": hasattr(job, "backend_job_id") and bool(job.backend_job_id),
                    "backend_job_id": getattr(job, "backend_job_id", None),
                    "status_json_keys": list(status_json.keys()) if status_json and isinstance(status_json, dict) else [],
                    "job_result_keys": list(job_result.keys()) if job_result and isinstance(job_result, dict) else [],
                }
                print(f"  ⚠️  No scores found for {experiment_id}. Debug info: {debug_info}")
            
            # Calculate lift
            if results.get("baseline_score") is not None and results.get("best_score") is not None:
                results["lift"] = results["best_score"] - results["baseline_score"]
            else:
                results["lift"] = None
            
            # Extract benchmark name from experiment name
            if hasattr(experiment, "name") and experiment.name:
                # Extract from "MIPRO Banking77" -> "Banking77"
                name_parts = experiment.name.split()
                if len(name_parts) > 1:
                    results["benchmark_name"] = " ".join(name_parts[1:])
                else:
                    results["benchmark_name"] = experiment.name
    
    return results


def main():
    """Main entry point."""
    # Check prerequisites
    if not check_redis():
        print("\n❌ Redis check failed. Please start Redis and try again.")
        sys.exit(1)
    
    if not check_queue_worker():
        print("\n❌ Queue worker check failed. Please start a worker and try again.")
        sys.exit(1)
    
    # Load config
    try:
        yaml_config = load_yaml_config()
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Submit experiments
    try:
        experiment_ids = submit_experiments(yaml_config)
    except Exception as e:
        print(f"\n❌ Failed to submit experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not experiment_ids:
        print("\n❌ No experiments submitted")
        sys.exit(1)
    
    # Poll for completion
    print("\n" + "=" * 80)
    print("POLLING FOR COMPLETION")
    print("=" * 80)
    
    def poll_single_experiment(exp_id: str) -> Optional[Dict]:
        """Poll a single experiment and return result."""
        return poll_experiment_status(exp_id, timeout=7200)  # 2 hour timeout
    
    # Poll all experiments in parallel
    completed_results = []
    with ThreadPoolExecutor(max_workers=len(experiment_ids)) as executor:
        # Submit all polling tasks
        future_to_id = {
            executor.submit(poll_single_experiment, exp_id): exp_id
            for exp_id in experiment_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            exp_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    completed_results.append(result)
            except Exception as e:
                exp_id_str = exp_id[:12] if isinstance(exp_id, str) else str(exp_id)[:12]
                print(f"❌ Error polling experiment {exp_id_str}...: {e}")
    
    print()
    
    # Extract detailed results
    print("Extracting results...")
    all_results = []
    for result in completed_results:
        detailed = extract_results(result["experiment_id"])
        all_results.append(detailed)
    
    # CRITICAL: Check if results are missing and loudly complain
    missing_results = []
    for result in all_results:
        status = result.get("status", "unknown")
        baseline = result.get("baseline_score")
        best = result.get("best_score")
        rollouts = result.get("total_rollouts")
        
        # Check if critical metrics are all missing
        if status == "completed" and baseline is None and best is None and rollouts is None:
            missing_results.append(result)
    
    if missing_results:
        print("\n" + "=" * 120)
        print("🚨 CRITICAL ERROR: MISSING RESULTS DETECTED 🚨")
        print("=" * 120)
        print(f"\n❌ Found {len(missing_results)} completed experiment(s) with NO results extracted!")
        print("\nThis indicates a problem with:")
        print("  1. Result extraction from experiment queue database")
        print("  2. Backend job metadata not being stored properly")
        print("  3. Backend job returning 404 (job may have been cleaned up)")
        print("\nAffected experiments:")
        for result in missing_results:
            exp_id = result.get("experiment_id", "unknown")
            benchmark = result.get("benchmark_name", "unknown")
            backend_job_id = result.get("backend_job_id", "N/A")
            print(f"  - {benchmark} (exp_id: {exp_id[:12]}..., backend_job_id: {backend_job_id})")
        print("\n⚠️  Debugging steps:")
        print("  1. Check experiment queue database for job.status_json and job.result")
        print("  2. Check if backend job still exists (may return 404 if cleaned up)")
        print("  3. Check queue worker logs for errors during job finalization")
        print("  4. Verify backend job completed successfully (not failed early)")
        print("=" * 120 + "\n")
    
    # Print results table
    print("\n" + "=" * 120)
    print("MIPRO EXPERIMENT RESULTS")
    print("=" * 120)
    print(f"{'Benchmark':<20} {'Status':<12} {'Baseline':<12} {'Best':<12} {'Lift':<12} {'Rollouts':<12} {'Time':<12} {'Eval N':<10} {'Trials':<10}")
    print("-" * 120)
    
    for result in all_results:
        benchmark_name = result.get("benchmark_name", result.get("experiment_id", "N/A")[:20])
        status = result.get("status", "unknown")
        baseline = result.get("baseline_score")
        best = result.get("best_score")
        lift = result.get("lift")
        rollouts = result.get("total_rollouts")
        time_sec = result.get("total_time")
        eval_n = result.get("eval_seeds_n")
        trials = result.get("trials_tried")
        
        baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
        best_str = f"{best:.4f}" if best is not None else "N/A"
        lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
        rollouts_str = str(rollouts) if rollouts is not None else "N/A"
        eval_n_str = str(eval_n) if eval_n is not None else "N/A"
        trials_str = str(trials) if trials is not None else "N/A"
        
        # Format time
        if time_sec is not None:
            if time_sec >= 60:
                time_str = f"{time_sec / 60:.1f}m"
            else:
                time_str = f"{time_sec:.1f}s"
        else:
            time_str = "N/A"
        
        print(f"{benchmark_name:<20} {status:<12} {baseline_str:<12} {best_str:<12} {lift_str:<12} {rollouts_str:<12} {time_str:<12} {eval_n_str:<10} {trials_str:<10}")
    
    print("=" * 120)
    
    # ✅ ADD: Print aggregate summary statistics
    print("\n" + "=" * 120)
    print("MIPRO AGGREGATE STATISTICS")
    print("=" * 120)
    print()
    
    successful_results = [r for r in all_results if r.get("status") == "completed"]
    
    if successful_results:
        # Calculate averages across all successful experiments
        baselines = [r.get("baseline_score") for r in successful_results if r.get("baseline_score") is not None]
        best_scores = [r.get("best_score") for r in successful_results if r.get("best_score") is not None]
        lifts = [r.get("lift") for r in successful_results if r.get("lift") is not None]
        rollouts_list = [r.get("total_rollouts") for r in successful_results if r.get("total_rollouts") is not None]
        times_list = [r.get("total_time") for r in successful_results if r.get("total_time") is not None]
        trials_list = [r.get("trials_tried") for r in successful_results if r.get("trials_tried") is not None]
        eval_n_list = [r.get("eval_seeds_n") for r in successful_results if r.get("eval_seeds_n") is not None]
        
        avg_baseline_final = sum(baselines) / len(baselines) if baselines else 0
        avg_best_final = sum(best_scores) / len(best_scores) if best_scores else 0
        avg_lift_final = sum(lifts) / len(lifts) if lifts else 0
        avg_rollouts = sum(rollouts_list) / len(rollouts_list) if rollouts_list else 0
        avg_time = sum(times_list) / len(times_list) if times_list else 0
        avg_trials = sum(trials_list) / len(trials_list) if trials_list else 0
        avg_eval_n = sum(eval_n_list) / len(eval_n_list) if eval_n_list else 0
        
        print(f"Total Experiments: {len(successful_results)}")
        print(f"Average Baseline Score: {avg_baseline_final:.4f} ({avg_baseline_final*100:.2f}%)")
        print(f"Average Best Score: {avg_best_final:.4f} ({avg_best_final*100:.2f}%)")
        print(f"Average Lift: {avg_lift_final:+.4f} ({avg_lift_final*100:+.2f}%)")
        if rollouts_list:
            print(f"Average Rollouts: {avg_rollouts:.1f}")
        if times_list:
            print(f"Average Time: {avg_time:.1f}s ({avg_time/60:.1f}min)")
        if trials_list:
            print(f"Average Trials: {avg_trials:.1f}")
        if eval_n_list:
            print(f"Average Eval N: {avg_eval_n:.1f}")
        print()
        
        # Show min/max lift
        if lifts:
            min_lift = min(lifts)
            max_lift = max(lifts)
            print(f"Lift Range: {min_lift:+.4f} to {max_lift:+.4f}")
            print(f"Best Improvement: {max_lift:+.4f} ({max_lift*100:+.2f}%)")
            print(f"Worst Performance: {min_lift:+.4f} ({min_lift*100:+.2f}%)")
            print()
        
        # Count positive vs negative lifts
        positive_lifts = [l for l in lifts if l > 0]
        negative_lifts = [l for l in lifts if l < 0]
        neutral_lifts = [l for l in lifts if l == 0]
        print(f"Experiments with Positive Lift: {len(positive_lifts)}/{len(lifts)} ({len(positive_lifts)/len(lifts)*100:.1f}%)")
        print(f"Experiments with Negative Lift: {len(negative_lifts)}/{len(lifts)} ({len(negative_lifts)/len(lifts)*100:.1f}%)")
        if neutral_lifts:
            print(f"Experiments with No Change: {len(neutral_lifts)}/{len(lifts)}")
    else:
        print("No successful experiments to aggregate.")
    
    print("=" * 120)
    print()
    
    # Save results to file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = COMPARISONS_DIR / f"mipro_comparison_readout_{timestamp}.txt"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("MIPRO EXPERIMENT RESULTS\n")
            f.write("=" * 120 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            for result in all_results:
                f.write(f"{result}\n")
        print(f"\n📄 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results file: {e}")


if __name__ == "__main__":
    main()

