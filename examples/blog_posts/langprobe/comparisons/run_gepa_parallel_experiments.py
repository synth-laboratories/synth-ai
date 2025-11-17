#!/usr/bin/env python3
"""Run GEPA experiments via experiment queue for Banking77 task.

This script:
- Checks that Redis and queue worker are running
- Submits experiment to the experiment queue for Banking77
- Polls for status until experiment completes
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
CONFIG_FILE = COMPARISONS_DIR / "synth_gepa_config.yaml"
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
    max_trials = benchmark_config.get("max_trials", defaults.get("max_trials", 50))
    max_cost_usd = benchmark_config.get("max_cost_usd", defaults.get("max_cost_usd", 10.0))
    
    gepa_population = benchmark_config.get("gepa_population", defaults.get("gepa_population", {}))
    num_generations = gepa_population.get("num_generations", 2)
    children_per_generation = gepa_population.get("children_per_generation", 5)
    
    # Build config overrides
    config_overrides = {
        # Termination config (for termination conditions)
        "prompt_learning.termination_config.max_rollouts": rollout_limit,
        "prompt_learning.termination_config.max_seconds": time_limit,
        "prompt_learning.termination_config.max_trials": max_trials,
        "prompt_learning.termination_config.max_cost_usd": max_cost_usd,
        # Rollout budget (for GEPA internal tracking - must match max_rollouts)
        # CRITICAL: This must override the base TOML's budget value (300 -> 100)
        "prompt_learning.gepa.rollout.budget": rollout_limit,
        # Population settings
        "prompt_learning.gepa.population.num_generations": num_generations,
        "prompt_learning.gepa.population.children_per_generation": children_per_generation,
    }
    
    # Debug: Print override to verify it's set correctly
    print(f"  [DEBUG] Setting rollout budget override: {config_overrides['prompt_learning.gepa.rollout.budget']} (from YAML limit: {rollout_limit})")
    
    # Apply model overrides if specified
    model_config = benchmark_config.get("model")
    if model_config:
        if "provider" in model_config:
            config_overrides["prompt_learning.policy.provider"] = model_config["provider"]
            print(f"  [DEBUG] Setting provider override: {config_overrides['prompt_learning.policy.provider']}")
        if "model" in model_config:
            config_overrides["prompt_learning.policy.model"] = model_config["model"]
            print(f"  [DEBUG] Setting model override: {config_overrides['prompt_learning.policy.model']}")
    
    # Apply proposer_mode override if specified
    proposer_mode = benchmark_config.get("proposer_mode")
    if proposer_mode:
        config_overrides["prompt_learning.gepa.proposer_mode"] = proposer_mode
        print(f"  [DEBUG] Setting proposer_mode override: {proposer_mode}")
    
    # Apply any other per-benchmark overrides (flatten nested dicts with dot notation)
    for key, value in benchmark_config.items():
        if key not in ("config_path", "rollout_limit", "time_limit_seconds", "max_trials", 
                      "max_cost_usd", "gepa_population", "model", "proposer_mode"):
            # Assume it's a config override path (e.g., "prompt_learning.gepa.mutation.rate")
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
        name=f"GEPA {display_name}",
        description=f"GEPA optimization for {display_name}",
        parallelism=parallelism,
        jobs=[
            ExperimentJobSpec(
                job_type=ExperimentJobType.GEPA,
                config_path=str(config_path),
                config_overrides=config_overrides,
            )
        ],
    )
    
    # ASSERT: Verify critical overrides are set correctly
    job_spec = request.jobs[0]
    assert job_spec.config_overrides is not None, f"config_overrides must be set for {benchmark_name}"
    
    # Assert rollout limit is set correctly
    rollout_budget_key = "prompt_learning.gepa.rollout.budget"
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
    # This catches errors early (e.g., missing train_seeds, val_seeds)
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
            
            # Additional GEPA-specific validation: verify required fields exist after merge
            pl_section = merged_config.get("prompt_learning", {})
            gepa_section = pl_section.get("gepa", {})
            eval_section = gepa_section.get("evaluation", {}) if isinstance(gepa_section, dict) else {}
            
            train_seeds = (
                eval_section.get("train_seeds") or 
                eval_section.get("seeds") or
                pl_section.get("train_seeds")
            )
            val_seeds = (
                eval_section.get("val_seeds") or 
                eval_section.get("validation_seeds") or
                pl_section.get("val_seeds") or
                pl_section.get("validation_seeds")
            )
            
            if not train_seeds:
                raise ValueError(
                    f"GEPA config validation failed for {benchmark_name}: "
                    f"train_seeds is missing after applying overrides. "
                    f"Ensure it's set in TOML at [prompt_learning.gepa.evaluation] or [prompt_learning] level"
                )
            if not val_seeds:
                raise ValueError(
                    f"GEPA config validation failed for {benchmark_name}: "
                    f"val_seeds is missing after applying overrides. "
                    f"Ensure it's set in TOML at [prompt_learning.gepa.evaluation] or [prompt_learning] level"
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
    submission_tasks = []
    
    # Prepare all experiment requests
    for benchmark_name, benchmark_config in benchmarks.items():
        try:
            display_name, request, config_path, rollout_limit, model_config = _prepare_experiment_request(
                benchmark_name, benchmark_config, defaults
            )
            submission_tasks.append((display_name, request, config_path, rollout_limit, model_config))
        except Exception as e:
            print(f"⚠️  Failed to prepare experiment for {benchmark_name}: {e}")
            continue
    
    if not submission_tasks:
        print("⚠️  No experiments to submit")
        return []
    
    # Submit experiments in parallel
    print(f"Submitting {len(submission_tasks)} experiment(s) in parallel...")
    
    def submit_single_experiment(task: Tuple) -> Tuple[str, str, Optional[Exception]]:
        """Submit a single experiment and return (display_name, experiment_id, error)."""
        display_name, request, config_path, rollout_limit, model_config = task
        try:
            experiment = create_experiment(request)
            return (display_name, experiment.experiment_id, None)
        except Exception as e:
            return (display_name, None, e)
    
    # Use ThreadPoolExecutor to submit in parallel
    with ThreadPoolExecutor(max_workers=min(len(submission_tasks), 10)) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(submit_single_experiment, task): task
            for task in submission_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            display_name, request, config_path, rollout_limit, model_config = task
            try:
                display_name_result, experiment_id, error = future.result()
                if error:
                    print(f"❌ Failed to submit experiment for {display_name_result}: {error}")
                elif experiment_id:
                    experiment_ids.append(experiment_id)
                    print(f"✅ Submitted experiment for {display_name_result}: {experiment_id}")
                    print(f"   Config: {config_path}")
                    print(f"   Rollout limit: {rollout_limit}")
                    if model_config:
                        provider = model_config.get('provider', '?')
                        model = model_config.get('model', '?')
                        print(f"   Model override: {provider}/{model}")
                        # Also print the actual config_overrides that were set
                        if request.jobs and request.jobs[0].config_overrides:
                            policy_provider = request.jobs[0].config_overrides.get('prompt_learning.policy.provider')
                            policy_model = request.jobs[0].config_overrides.get('prompt_learning.policy.model')
                            print(f"   Config overrides: provider={policy_provider}, model={policy_model}")
            except Exception as e:
                print(f"❌ Unexpected error submitting experiment for {display_name}: {e}")
    
    print(f"\n✅ Successfully submitted {len(experiment_ids)}/{len(submission_tasks)} experiment(s)")
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
                        
                        # Add steps per rollout info for crafter (if available)
                        steps_info_str = ""
                        if job.config_path:
                            try:
                                import tomllib
                                from pathlib import Path
                                config_path = Path(job.config_path)
                                if config_path.exists():
                                    with config_path.open("rb") as f:
                                        config_data = tomllib.load(f)
                                    pl_config = config_data.get("prompt_learning", {})
                                    gepa_config = pl_config.get("gepa", {})
                                    env_name = gepa_config.get("env_name", "")
                                    
                                    # Check for max_steps in policy config (for crafter)
                                    if env_name == "crafter":
                                        policy_config = pl_config.get("policy", {})
                                        max_steps = policy_config.get("max_steps")
                                        if max_steps:
                                            steps_info_str = f" | {max_steps} steps/rollout"
                                        else:
                                            # Default for crafter is 10 steps
                                            steps_info_str = " | 10 steps/rollout (default)"
                            except Exception:
                                pass
                        
                        # ✅ ADD: Show validation info if available, otherwise show normal status
                        if validation_info:
                            status_line = f" | {formatted}{validation_info}"
                        else:
                            # Always show status_json if it has any data (even just policy/environment)
                            # This ensures we see policy/environment even before progress events arrive
                            status_line = f" | {formatted}{rollouts_per_min_str}{steps_info_str}"
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
                        gepa_config = pl_config.get("gepa", {})
                        rollout_limit = gepa_config.get("rollout_limit")
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
    
    print(f"⏱️  Timeout waiting for experiment {experiment_id}")
    return None


def fetch_backend_job_details(backend_job_id: str) -> Optional[Dict]:
    """Fetch job details including artifacts and snapshots from backend API.
    
    Returns:
        Dict with job details, artifacts, and best_snapshot, or None if fetch fails
    """
    import os
    
    if requests is None:
        print(f"  [WARN] requests library not available, cannot fetch backend artifacts for {backend_job_id}")
        return None
    
    backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    if not backend_url.endswith("/api"):
        backend_url = f"{backend_url}/api"
    
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print(f"  [WARN] SYNTH_API_KEY not set, cannot fetch backend artifacts for {backend_job_id}")
        return None
    
    # Retry logic: try with increasing timeouts
    max_retries = 3
    timeouts = [30.0, 60.0, 120.0]
    
    for attempt in range(max_retries):
        timeout = timeouts[min(attempt, len(timeouts) - 1)]
        try:
            # Fetch job details (includes best_snapshot)
            job_url = f"{backend_url}/prompt-learning/online/jobs/{backend_job_id}"
            headers = {"Authorization": f"Bearer {api_key}"}
            print(f"  [DEBUG] Fetching backend job details (attempt {attempt + 1}/{max_retries}, timeout={timeout}s)...")
            response = requests.get(job_url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                job_data = response.json()
                
                # Also fetch artifacts list
                artifacts_url = f"{backend_url}/prompt-learning/online/jobs/{backend_job_id}/artifacts"
                artifacts_response = requests.get(artifacts_url, headers=headers, timeout=timeout)
                artifacts = []
                if artifacts_response.status_code == 200:
                    artifacts = artifacts_response.json()
                
                best_snapshot = job_data.get("best_snapshot")
                best_snapshot_id = job_data.get("best_snapshot_id")
                # ✅ ADD: Debug logging to understand what's returned
                print(f"  [DEBUG] Backend API response for {backend_job_id}:")
                print(f"    best_snapshot_id={best_snapshot_id}")
                print(f"    best_snapshot type={type(best_snapshot)}, is None={best_snapshot is None}")
                if best_snapshot:
                    print(f"    best_snapshot keys={list(best_snapshot.keys()) if isinstance(best_snapshot, dict) else 'not dict'}")
                    if isinstance(best_snapshot, dict):
                        print(f"    best_snapshot has best_prompt_messages={'best_prompt_messages' in best_snapshot}")
                        print(f"    best_snapshot has archive={'archive' in best_snapshot}")
                metadata = job_data.get("metadata", {})
                stats = metadata.get("stats", {})
                print(f"    metadata.stats keys={list(stats.keys()) if isinstance(stats, dict) else 'not dict'}")
                if isinstance(stats, dict):
                    print(f"    stats.total_tokens={stats.get('total_tokens')}")
                    print(f"    stats.trials_tried={stats.get('trials_tried')}")
                    print(f"    stats.validation_rollouts_executed={stats.get('validation_rollouts_executed')}")
                
                return {
                    "job": job_data,
                    "artifacts": artifacts,
                    "best_snapshot": best_snapshot,
                    "best_snapshot_id": best_snapshot_id,
                }
            else:
                print(f"  [WARN] Backend API returned {response.status_code} for job {backend_job_id}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    print(f"  [DEBUG] Retrying...")
                    continue
                return None
        except requests.exceptions.Timeout as e:
            print(f"  [WARN] Timeout fetching backend job details (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  [DEBUG] Retrying with longer timeout...")
                continue
            return None
        except Exception as e:
            print(f"  [WARN] Failed to fetch backend job details for {backend_job_id}: {e}")
            if attempt < max_retries - 1:
                print(f"  [DEBUG] Retrying...")
                continue
            return None
    
    return None


def extract_results(experiment_id: str) -> Dict:
    """Extract results from a completed experiment, matching original script's metadata."""
    from synth_ai.experiment_queue.service import fetch_experiment
    
    experiment = fetch_experiment(experiment_id)
    if not experiment:
        return {"error": "Experiment not found"}
    
    results = {
        "experiment_id": experiment_id,
        "name": experiment.name,
        "status": experiment.status.value,
    }
    
    # Extract results from each job
    job_results = []
    for job in experiment.jobs:
        job_result = {
            "job_id": job.job_id,
            "status": job.status.value,
            "error": job.error,
        }
        
        # Calculate total_time from database timestamps (most reliable)
        # These timestamps are stored in local SQLite DB
        if job.started_at and job.completed_at:
            time_delta = job.completed_at - job.started_at
            job_result["total_time"] = time_delta.total_seconds()
            print(f"  [DEBUG] Calculated time from DB timestamps: {job_result['total_time']:.1f}s for job {job.job_id}")
        elif job.result and job.result.get("total_time"):
            job_result["total_time"] = job.result.get("total_time")
        
        # Extract from status_json (progress tracking) - most reliable source
        if job.status_json and isinstance(job.status_json, dict):
            status_data = job.status_json
            # Rollouts from progress tracking
            if job_result.get("total_rollouts") is None:
                rollouts_completed = status_data.get("rollouts_completed")
                rollouts_total = status_data.get("total_rollouts") or status_data.get("rollouts_total")
                if rollouts_completed is not None:
                    job_result["total_rollouts"] = rollouts_completed
                    print(f"  [DEBUG] Found {rollouts_completed} rollouts from status_json for job {job.job_id}")
                elif rollouts_total is not None:
                    job_result["total_rollouts"] = rollouts_total
            
            # Trials from progress tracking
            if job_result.get("trials_tried") is None:
                trials_completed = status_data.get("trials_completed")
                if trials_completed is not None:
                    job_result["trials_tried"] = trials_completed
                    print(f"  [DEBUG] Found {trials_completed} trials from status_json for job {job.job_id}")
        
        # Count trials from database (fallback)
        # Trials are stored in local SQLite DB and eagerly loaded by fetch_experiment
        if job_result.get("trials_tried") is None:
            try:
                if hasattr(job, 'trials') and job.trials:
                    completed_trials = [t for t in job.trials if t.status.value == "completed"]
                    job_result["trials_tried"] = len(completed_trials)
                    # Debug: log trial count
                    if len(completed_trials) > 0:
                        print(f"  [DEBUG] Found {len(completed_trials)} completed trials in DB for job {job.job_id}")
            except Exception as e:
                # Fallback if trials not accessible
                print(f"  [DEBUG] Could not access trials from DB: {e}")
                pass
        
        # Fetch backend job details if backend_job_id is available
        backend_job_data = None
        if job.backend_job_id:
            print(f"  [DEBUG] Fetching backend job details for backend_job_id={job.backend_job_id}")
            backend_job_data = fetch_backend_job_details(job.backend_job_id)
            if backend_job_data:
                best_snapshot = backend_job_data.get('best_snapshot')
                artifacts = backend_job_data.get('artifacts', [])
                print(f"  [DEBUG] Successfully fetched backend job data (has best_snapshot={best_snapshot is not None}, artifacts={len(artifacts)})")
                # ✅ ADD: More detailed debugging
                if best_snapshot:
                    print(f"  [DEBUG] best_snapshot is dict={isinstance(best_snapshot, dict)}")
                    if isinstance(best_snapshot, dict):
                        print(f"  [DEBUG] best_snapshot keys: {list(best_snapshot.keys())[:20]}")
                        print(f"  [DEBUG] best_snapshot has best_prompt_messages={'best_prompt_messages' in best_snapshot}")
                        print(f"  [DEBUG] best_snapshot has archive={'archive' in best_snapshot}")
                else:
                    print(f"  [DEBUG] WARNING: best_snapshot is None for job {job.job_id}")
        
        # Extract from job.result (ResultSummary dict) - ONLY if job succeeded
        # Failed jobs should not have scores extracted (they may contain stale/cached data)
        if job.result and job.status.value == "completed":
            result_dict = job.result
            stats = result_dict.get("stats", {})
            artifacts = result_dict.get("artifacts", {})
            learning_curve = result_dict.get("learning_curve", [])
            
            # Basic scores
            baseline_score = result_dict.get("baseline_score")
            best_score = result_dict.get("best_score")
            
            # candidate1_score is best_score (first candidate)
            candidate1_score = best_score
            candidate1_lift = None
            if baseline_score is not None and candidate1_score is not None:
                candidate1_lift = candidate1_score - baseline_score
            
            job_result["baseline_score"] = baseline_score
            job_result["candidate1_score"] = candidate1_score
            job_result["candidate1_lift"] = candidate1_lift
            job_result["best_score"] = best_score  # Keep for compatibility
            
            # Rollouts: prefer status_json (already set above), then stats, then result_dict, then learning curve
            # Don't use config limit - use actual executed rollouts
            if job_result.get("total_rollouts") is None:
                total_rollouts = None
                if stats.get("total_rollouts"):
                    total_rollouts = stats.get("total_rollouts")
                elif result_dict.get("total_rollouts"):
                    total_rollouts = result_dict.get("total_rollouts")
                elif learning_curve:
                    # Get max rollout_count from learning curve points
                    max_rollout = 0
                    for point in learning_curve:
                        if isinstance(point, dict) and point.get("rollout_count"):
                            max_rollout = max(max_rollout, point.get("rollout_count", 0))
                    if max_rollout > 0:
                        total_rollouts = max_rollout
                
                # Fallback: count rollouts from trials if available (each trial = 1 rollout)
                # Trials are stored in local SQLite DB
                if total_rollouts is None:
                    try:
                        if hasattr(job, 'trials') and job.trials:
                            # Count completed trials as rollouts
                            completed_trials = [t for t in job.trials if t.status.value == "completed"]
                            if completed_trials:
                                total_rollouts = len(completed_trials)
                                print(f"  [DEBUG] Counted {total_rollouts} rollouts from DB trials for job {job.job_id}")
                    except Exception as e:
                        print(f"  [DEBUG] Could not count rollouts from trials: {e}")
                        pass
                
                job_result["total_rollouts"] = total_rollouts
            
            # Time: prefer calculated from DB timestamps (already set above), fallback to result_dict
            if job_result.get("total_time") is None:
                job_result["total_time"] = result_dict.get("total_time")
            
            # Extract from stats dict
            job_result["total_cost"] = stats.get("total_cost_usd") or stats.get("total_cost")
            
            # Total tokens: prefer status_json, then backend stats, then stats, then sum from trials
            total_tokens = None
            if job.status_json and isinstance(job.status_json, dict):
                status_data = job.status_json
                total_tokens = status_data.get("rollout_tokens_used") or status_data.get("total_tokens")
            
            # ✅ ADD: Check backend metadata stats first (most reliable)
            if total_tokens is None and backend_job_data:
                backend_job = backend_job_data.get("job", {})
                backend_metadata = backend_job.get("metadata", {})
                backend_stats = backend_metadata.get("stats", {})
                total_tokens = backend_stats.get("total_tokens")
                if total_tokens is not None:
                    print(f"  [DEBUG] Found total_tokens={total_tokens} from backend metadata.stats for job {job.job_id}")
                else:
                    # ✅ ADD: Try to calculate from individual token fields if total_tokens not set
                    rollouts_prompt = backend_stats.get("rollouts_prompt_tokens", 0) or 0
                    rollouts_completion = backend_stats.get("rollouts_completion_tokens", 0) or 0
                    rollouts_unknown = backend_stats.get("rollouts_unknown_tokens", 0) or 0
                    mutation_prompt = backend_stats.get("mutation_prompt_tokens", 0) or 0
                    mutation_completion = backend_stats.get("mutation_completion_tokens", 0) or 0
                    mutation_unknown = backend_stats.get("mutation_unknown_tokens", 0) or 0
                    calculated_tokens = rollouts_prompt + rollouts_completion + rollouts_unknown + mutation_prompt + mutation_completion + mutation_unknown
                    if calculated_tokens > 0:
                        total_tokens = calculated_tokens
                        print(f"  [DEBUG] Calculated total_tokens={total_tokens} from individual token fields for job {job.job_id}")
            
            if total_tokens is None:
                total_tokens = stats.get("total_tokens") or stats.get("tokens")
            
            if total_tokens is None:
                # Try to sum tokens from trials if available
                try:
                    if hasattr(job, 'trials') and job.trials:
                        # Sum tokens from trial metadata
                        trial_tokens = 0
                        for trial in job.trials:
                            if trial.metadata_json and isinstance(trial.metadata_json, dict):
                                trial_tokens += trial.metadata_json.get("tokens", 0) or 0
                        if trial_tokens > 0:
                            total_tokens = trial_tokens
                            print(f"  [DEBUG] Summed {total_tokens} tokens from DB trials for job {job.job_id}")
                except Exception as e:
                    print(f"  [DEBUG] Could not sum tokens from trials: {e}")
                    pass
            
            job_result["total_tokens"] = total_tokens
            
            # ✅ ADD: Extract eval_seeds_n and trials_tried from backend metadata stats for completed jobs
            # (Same extraction logic as failed jobs, but for successful jobs)
            if backend_job_data:
                backend_job = backend_job_data.get("job", {})
                backend_metadata = backend_job.get("metadata", {})
                backend_stats = backend_metadata.get("stats", {})
                
                # Extract eval_seeds_n from backend metadata stats (validation_rollouts_executed = eval N)
                if backend_stats.get("validation_rollouts_executed") is not None:
                    job_result["eval_seeds_n"] = backend_stats["validation_rollouts_executed"]
                    print(f"  [DEBUG] Found eval_n={backend_stats['validation_rollouts_executed']} from backend metadata.stats.validation_rollouts_executed for completed job {job.job_id}")
                
                # Extract trials_tried from backend stats (if not already set)
                if job_result.get("trials_tried") is None:
                    if backend_stats.get("trials_tried") is not None:
                        job_result["trials_tried"] = backend_stats["trials_tried"]
                        print(f"  [DEBUG] Found trials_tried={backend_stats['trials_tried']} from backend metadata.stats for completed job {job.job_id}")
                    elif backend_stats.get("optimization_trials_evaluated") is not None:
                        job_result["trials_tried"] = backend_stats["optimization_trials_evaluated"]
                        print(f"  [DEBUG] Found trials_tried={backend_stats['optimization_trials_evaluated']} from backend metadata.stats.optimization_trials_evaluated for completed job {job.job_id}")
        elif job.status.value == "failed":
            # Job failed - don't extract scores (may be stale data)
            # Set all scores to None explicitly
            job_result["baseline_score"] = None
            job_result["candidate1_score"] = None
            job_result["candidate1_lift"] = None
            job_result["best_score"] = None
            error_msg = job.error or "Unknown error"
            print(f"  [ERROR] Job {job.job_id} FAILED: {error_msg}")
            print(f"  [DEBUG] Job {job.job_id} failed - skipping score extraction to avoid stale data")
            
            # For failed jobs, only extract metadata that doesn't depend on job.result
            # Try to get eval/trial counts from backend job data if available
            if backend_job_data:
                backend_job = backend_job_data.get("job", {})
                backend_metadata = backend_job.get("metadata", {})
                backend_stats = backend_metadata.get("stats", {})
                
                # Extract from backend metadata stats (validation_rollouts_executed = eval N)
                if backend_stats.get("validation_rollouts_executed") is not None:
                    job_result["eval_seeds_n"] = backend_stats["validation_rollouts_executed"]
                    print(f"  [DEBUG] Found eval_n={backend_stats['validation_rollouts_executed']} from backend metadata.stats.validation_rollouts_executed")
                
                # Extract trials_tried from backend stats
                if backend_stats.get("trials_tried") is not None:
                    job_result["trials_tried"] = backend_stats["trials_tried"]
                    print(f"  [DEBUG] Found trials_tried={backend_stats['trials_tried']} from backend metadata")
                elif backend_stats.get("optimization_trials_evaluated") is not None:
                    job_result["trials_tried"] = backend_stats["optimization_trials_evaluated"]
                    print(f"  [DEBUG] Found trials_tried={backend_stats['optimization_trials_evaluated']} from backend metadata.stats.optimization_trials_evaluated")
        
        # ✅ ADD: Extract proposal method (proposer_mode) from config_overrides
        proposal_method = None
        if job.config_overrides:
            proposal_method = job.config_overrides.get("prompt_learning.gepa.proposer_mode")
            if proposal_method:
                job_result["proposal_method"] = proposal_method
                print(f"  [DEBUG] Extracted proposal_method={proposal_method} from config_overrides for job {job.job_id}")
        
        # Fallback: check backend metadata if available
        if not proposal_method and backend_job_data:
            backend_job = backend_job_data.get("job", {})
            backend_metadata = backend_job.get("metadata", {})
            proposal_method = backend_metadata.get("proposer_mode")
            if proposal_method:
                job_result["proposal_method"] = proposal_method
                print(f"  [DEBUG] Extracted proposal_method={proposal_method} from backend metadata for job {job.job_id}")
        
        # Default to "synth" if not found (since this script uses synth_gepa_config.yaml)
        if not job_result.get("proposal_method"):
            job_result["proposal_method"] = "synth"
        
        # If no result dict, try to get policy_model from config_overrides FIRST (most accurate)
        # Then fallback to original config file
        if not job_result.get("policy_model"):
            # PRIORITY 1: Check config_overrides (this is what was actually used)
            if job.config_overrides:
                model_override = job.config_overrides.get("prompt_learning.policy.model")
                if model_override:
                    job_result["policy_model"] = model_override
                    print(f"  [DEBUG] Extracted policy_model={model_override} from config_overrides for job {job.job_id}")
            
            # PRIORITY 2: Fallback to original config file (if override not found)
            if not job_result.get("policy_model"):
                try:
                    import tomllib
                    config_path = Path(job.config_path)
                    if config_path.exists():
                        with config_path.open("rb") as f:
                            config_data = tomllib.load(f)
                        pl_config = config_data.get("prompt_learning", {})
                        gepa_config = pl_config.get("gepa", {})
                        policy_config = gepa_config.get("policy") or pl_config.get("policy", {})
                        if isinstance(policy_config, dict):
                            policy_model = policy_config.get("model")
                            if policy_model:
                                job_result["policy_model"] = policy_model
                                print(f"  [DEBUG] Extracted policy_model={policy_model} from config file for job {job.job_id}")
                except Exception as e:
                    print(f"  [DEBUG] Could not extract policy_model from config: {e}")
                    pass
        
        # Store result dict and backend data for variant extraction
        if job.result:
            job_result["result"] = job.result
        if backend_job_data:
            job_result["backend_job_data"] = backend_job_data
        
        job_results.append(job_result)
    
    results["jobs"] = job_results
    return results


def extract_variants(job_result: Dict) -> List[Dict]:
    """Extract evaluated prompt variants/templates from job results.
    
    Returns:
        List of dicts with keys: score, eval_score, variant_text, variant_type, metadata
    """
    variants = []
    
    # First, try to extract from backend artifacts/snapshots (most reliable)
    backend_job_data = job_result.get("backend_job_data")
    result_dict = job_result.get("result")
    print(f"  [DEBUG] extract_variants called for job_result with keys: {list(job_result.keys())}")
    print(f"  [DEBUG] backend_job_data is None: {backend_job_data is None}, type: {type(backend_job_data)}")
    print(f"  [DEBUG] result_dict is None: {result_dict is None}, type: {type(result_dict)}")
    if result_dict:
        print(f"  [DEBUG] result_dict keys: {list(result_dict.keys()) if isinstance(result_dict, dict) else 'not dict'}")
        if isinstance(result_dict, dict):
            artifacts = result_dict.get("artifacts", {})
            learning_curve = result_dict.get("learning_curve", [])
            print(f"  [DEBUG] result_dict has artifacts: {bool(artifacts)}, learning_curve length: {len(learning_curve) if isinstance(learning_curve, list) else 0}")
            if artifacts:
                print(f"  [DEBUG] artifacts keys: {list(artifacts.keys()) if isinstance(artifacts, dict) else 'not dict'}")
                print(f"  [DEBUG] artifacts content: {str(artifacts)[:500]}")  # First 500 chars
    if backend_job_data:
        # Extract from best_snapshot
        best_snapshot = backend_job_data.get("best_snapshot")
        if best_snapshot and isinstance(best_snapshot, dict):
            # ✅ ADD: Try best_prompt_messages first (new field), then fallback to messages
            # Note: best_snapshot IS the payload (from snapshot_row.get("payload"))
            messages = best_snapshot.get("best_prompt_messages") or best_snapshot.get("messages", [])
            if messages:
                variant_text = "\n".join([
                    f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                    for msg in messages if isinstance(msg, dict)
                ])
                variants.append({
                    "score": job_result.get("best_score"),
                    "eval_score": job_result.get("candidate1_score"),
                    "variant_text": variant_text,
                    "variant_type": "best_snapshot",
                    "metadata": {"source": "backend_best_snapshot"},
                })
                print(f"  [DEBUG] Extracted variant from backend best_snapshot (len={len(variant_text)})")
        
        # ✅ ADD: Extract variants from archive in best_snapshot payload
        if best_snapshot and isinstance(best_snapshot, dict):
            archive = best_snapshot.get("archive", [])
            if isinstance(archive, list):
                for item in archive:
                    if isinstance(item, dict):
                        score = item.get("score", {}).get("accuracy") if isinstance(item.get("score"), dict) else item.get("score")
                        messages = item.get("messages", [])
                        if messages:
                            variant_text = "\n".join([
                                f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                                for msg in messages if isinstance(msg, dict)
                            ])
                            variants.append({
                                "score": score,
                                "eval_score": None,
                                "variant_text": variant_text,
                                "variant_type": "archive_template",
                                "metadata": {"source": "backend_archive", "archive_item": item},
                            })
                            print(f"  [DEBUG] Extracted variant from backend archive (len={len(variant_text)})")
        
        # ✅ ADD: Extract attempted_candidates from backend metadata (REALLY IMPORTANT!)
        backend_job = backend_job_data.get("job", {})
        backend_metadata = backend_job.get("metadata", {})
        # ✅ ADD: Debug logging to see what's in metadata
        print(f"  [DEBUG] Backend metadata keys: {list(backend_metadata.keys()) if isinstance(backend_metadata, dict) else 'not dict'}")
        print(f"  [DEBUG] Backend metadata full content: {str(backend_metadata)[:1000]}")  # First 1000 chars
        attempted_candidates = backend_metadata.get("attempted_candidates", [])
        print(f"  [DEBUG] attempted_candidates type={type(attempted_candidates)}, len={len(attempted_candidates) if isinstance(attempted_candidates, list) else 'N/A'}")
        optimized_candidates = backend_metadata.get("optimized_candidates", [])
        print(f"  [DEBUG] optimized_candidates type={type(optimized_candidates)}, len={len(optimized_candidates) if isinstance(optimized_candidates, list) else 'N/A'}")
        # ✅ ADD: Assertions to ensure attempted_candidates are valid
        assert isinstance(attempted_candidates, (list, type(None))), f"attempted_candidates must be list or None, got {type(attempted_candidates)}"
        if attempted_candidates:
            assert len(attempted_candidates) > 0, f"attempted_candidates must not be empty if present, got len={len(attempted_candidates)}"
            print(f"  [DEBUG] Found {len(attempted_candidates)} attempted_candidates in backend metadata")
            for idx, candidate in enumerate(attempted_candidates):
                try:
                    # ✅ ADD: Assertions to ensure candidate is valid
                    assert isinstance(candidate, dict), f"attempted_candidate[{idx}] must be dict, got {type(candidate)}"
                    assert "object" in candidate, f"attempted_candidate[{idx}] must have 'object' field"
                    assert "accuracy" in candidate, f"attempted_candidate[{idx}] must have 'accuracy' field"
                    
                    # Note: AttemptedCandidate.to_dict() returns "type" not "candidate_type"
                    candidate_type = candidate.get("type") or candidate.get("candidate_type")  # Support both field names
                    obj = candidate.get("object", {})
                    variant_text = None
                    
                    if candidate_type == "template":
                        # ✅ Templates: Try messages field first, then fallback to object.sections
                        messages = candidate.get("messages", [])
                        if not messages and isinstance(obj, dict):
                            # ✅ ADD: Check for nested data structure (object.data.sections)
                            if "data" in obj and isinstance(obj["data"], dict):
                                sections = obj["data"].get("sections", [])
                            else:
                                sections = obj.get("sections", [])
                            if sections:
                                messages = [
                                    {"role": s.get("role", "?"), "content": s.get("content", "")}
                                    for s in sections if isinstance(s, dict)
                                ]
                        
                        if messages:
                            assert isinstance(messages, list), f"messages must be list for candidate[{idx}], got {type(messages)}"
                            assert len(messages) > 0, f"messages must not be empty for candidate[{idx}]"
                            variant_text = "\n".join([
                                f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                                for msg in messages if isinstance(msg, dict)
                            ])
                        else:
                            print(f"  [DEBUG] WARNING: attempted_candidate[{idx}] (template) has no messages")
                    
                    elif candidate_type == "transformation":
                        # ✅ Transformations: Extract text_replacements and example_injections (THIS IS WHAT WE NEED!)
                        if isinstance(obj, dict):
                            text_replacements = obj.get("text_replacements", [])
                            example_injections = obj.get("example_injections", [])
                            
                            parts = []
                            if text_replacements:
                                parts.append("TEXT REPLACEMENTS:")
                                for repl_idx, repl in enumerate(text_replacements):
                                    if isinstance(repl, dict):
                                        old_text = repl.get("old_text", "")
                                        new_text = repl.get("new_text", "")
                                        role = repl.get("apply_to_role", "?")
                                        parts.append(f"  [{repl_idx+1}] Role: {role}")
                                        # ✅ Always show both Old and New, even if empty (to see what changed)
                                        parts.append(f"      Old: {old_text if old_text else '(empty)'}")
                                        parts.append(f"      New: {new_text if new_text else '(empty)'}")
                            
                            if example_injections:
                                parts.append("\nEXAMPLE INJECTIONS:")
                                for inj_idx, inj in enumerate(example_injections):
                                    if isinstance(inj, dict):
                                        examples = inj.get("examples", [])
                                        after_role = inj.get("insert_after_role", "?")
                                        parts.append(f"  [{inj_idx+1}] Insert after role: {after_role}")
                                        parts.append(f"      Examples ({len(examples)}):")
                                        for ex_idx, ex in enumerate(examples):  # Show all examples, no limit
                                            if isinstance(ex, dict):
                                                ex_role = ex.get("role", "?")
                                                ex_content = ex.get("content", "")
                                                parts.append(f"        [{ex_idx+1}] {ex_role}: {ex_content}")
                            
                            if parts:
                                variant_text = "\n".join(parts)
                            else:
                                print(f"  [DEBUG] WARNING: attempted_candidate[{idx}] (transformation) has no text_replacements or example_injections")
                        else:
                            print(f"  [DEBUG] WARNING: attempted_candidate[{idx}] (transformation) object is not a dict: {type(obj)}")
                    
                    if variant_text:
                        assert len(variant_text) > 0, f"variant_text must not be empty for candidate[{idx}]"
                        accuracy = candidate.get("accuracy")
                        assert accuracy is not None, f"accuracy must not be None for candidate[{idx}]"
                        prompt_length = candidate.get("prompt_length")
                        tool_call_rate = candidate.get("tool_call_rate")
                        variants.append({
                            "score": accuracy,
                            "eval_score": None,
                            "variant_text": variant_text,
                            "variant_type": f"attempted_{candidate_type}",
                            "metadata": {
                                "source": "backend_attempted_candidates",
                                "index": idx,
                                "prompt_length": prompt_length,
                                "tool_call_rate": tool_call_rate,
                                "candidate": candidate,
                            },
                        })
                        print(f"  [DEBUG] Extracted attempted candidate #{idx} (type={candidate_type}, accuracy={accuracy}, len={len(variant_text)})")
                    else:
                        print(f"  [DEBUG] Skipped attempted_candidate[{idx}] (type={candidate_type}) - no extractable content")
                except AssertionError as e:
                    print(f"  [DEBUG] Assertion failed for attempted_candidate[{idx}]: {e}")
                    # Continue to next candidate instead of failing
                except Exception as e:
                    print(f"  [DEBUG] Error extracting attempted_candidate[{idx}]: {e}")
                    # Continue to next candidate instead of failing
        
        # ✅ ADD: Extract optimized_candidates from backend metadata
        optimized_candidates = backend_metadata.get("optimized_candidates", [])
        # ✅ ADD: Assertions to ensure optimized_candidates are valid
        assert isinstance(optimized_candidates, (list, type(None))), f"optimized_candidates must be list or None, got {type(optimized_candidates)}"
        if optimized_candidates:
            assert len(optimized_candidates) > 0, f"optimized_candidates must not be empty if present, got len={len(optimized_candidates)}"
            print(f"  [DEBUG] Found {len(optimized_candidates)} optimized_candidates in backend metadata")
            for idx, candidate in enumerate(optimized_candidates):
                try:
                    # ✅ ADD: Assertions to ensure candidate is valid
                    assert isinstance(candidate, dict), f"optimized_candidate[{idx}] must be dict, got {type(candidate)}"
                    assert "object" in candidate, f"optimized_candidate[{idx}] must have 'object' field"
                    assert "score" in candidate, f"optimized_candidate[{idx}] must have 'score' field"
                    
                    payload_kind = candidate.get("payload_kind")
                    obj = candidate.get("object", {})
                    variant_text = None
                    
                    if payload_kind == "template":
                        # ✅ Templates: Try messages field first, then fallback to object.sections
                        messages = candidate.get("messages", [])
                        if not messages and isinstance(obj, dict):
                            # ✅ ADD: Check for nested data structure (object.data.sections)
                            if "data" in obj and isinstance(obj["data"], dict):
                                sections = obj["data"].get("sections", [])
                            else:
                                sections = obj.get("sections", [])
                            if sections:
                                messages = [
                                    {"role": s.get("role", "?"), "content": s.get("content", "")}
                                    for s in sections if isinstance(s, dict)
                                ]
                        
                        if messages:
                            assert isinstance(messages, list), f"messages must be list for optimized_candidate[{idx}], got {type(messages)}"
                            assert len(messages) > 0, f"messages must not be empty for optimized_candidate[{idx}]"
                            variant_text = "\n".join([
                                f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                                for msg in messages if isinstance(msg, dict)
                            ])
                        else:
                            print(f"  [DEBUG] WARNING: optimized_candidate[{idx}] (template) has no messages")
                    
                    elif payload_kind == "transformation":
                        # ✅ Transformations: Extract text_replacements and example_injections (THIS IS WHAT WE NEED!)
                        if isinstance(obj, dict):
                            text_replacements = obj.get("text_replacements", [])
                            example_injections = obj.get("example_injections", [])
                            
                            parts = []
                            if text_replacements:
                                parts.append("TEXT REPLACEMENTS:")
                                for repl_idx, repl in enumerate(text_replacements):
                                    if isinstance(repl, dict):
                                        old_text = repl.get("old_text", "")
                                        new_text = repl.get("new_text", "")
                                        role = repl.get("apply_to_role", "?")
                                        parts.append(f"  [{repl_idx+1}] Role: {role}")
                                        # ✅ Always show both Old and New, even if empty (to see what changed)
                                        parts.append(f"      Old: {old_text if old_text else '(empty)'}")
                                        parts.append(f"      New: {new_text if new_text else '(empty)'}")
                            
                            if example_injections:
                                parts.append("\nEXAMPLE INJECTIONS:")
                                for inj_idx, inj in enumerate(example_injections):
                                    if isinstance(inj, dict):
                                        examples = inj.get("examples", [])
                                        after_role = inj.get("insert_after_role", "?")
                                        parts.append(f"  [{inj_idx+1}] Insert after role: {after_role}")
                                        parts.append(f"      Examples ({len(examples)}):")
                                        for ex_idx, ex in enumerate(examples):  # Show all examples, no limit
                                            if isinstance(ex, dict):
                                                ex_role = ex.get("role", "?")
                                                ex_content = ex.get("content", "")
                                                parts.append(f"        [{ex_idx+1}] {ex_role}: {ex_content}")
                            
                            if parts:
                                variant_text = "\n".join(parts)
                            else:
                                print(f"  [DEBUG] WARNING: optimized_candidate[{idx}] (transformation) has no text_replacements or example_injections")
                        else:
                            print(f"  [DEBUG] WARNING: optimized_candidate[{idx}] (transformation) object is not a dict: {type(obj)}")
                    
                    if variant_text:
                        assert len(variant_text) > 0, f"variant_text must not be empty for optimized_candidate[{idx}]"
                        score = candidate.get("score", {})
                        accuracy = score.get("accuracy") if isinstance(score, dict) else score
                        assert accuracy is not None, f"accuracy must not be None for optimized_candidate[{idx}]"
                        variants.append({
                            "score": accuracy,
                            "eval_score": None,
                            "variant_text": variant_text,
                            "variant_type": f"optimized_{payload_kind}",
                            "metadata": {
                                "source": "backend_optimized_candidates",
                                "index": idx,
                                "rank": candidate.get("rank"),
                                "candidate": candidate,
                            },
                        })
                        print(f"  [DEBUG] Extracted optimized candidate #{idx} (type={payload_kind}, accuracy={accuracy}, len={len(variant_text)})")
                    else:
                        print(f"  [DEBUG] Skipped optimized_candidate[{idx}] (type={payload_kind}) - no extractable content")
                except AssertionError as e:
                    print(f"  [DEBUG] Assertion failed for optimized_candidate[{idx}]: {e}")
                    # Continue to next candidate instead of failing
                except Exception as e:
                    print(f"  [DEBUG] Error extracting optimized_candidate[{idx}]: {e}")
                    # Continue to next candidate instead of failing
        
        # Extract from artifacts (snapshots)
        artifacts = backend_job_data.get("artifacts", [])
        for artifact in artifacts:
            if isinstance(artifact, dict):
                snapshot_id = artifact.get("snapshot_id")
                label = artifact.get("label", "")
                if snapshot_id and label:
                    # Could fetch snapshot details here if needed
                    variants.append({
                        "score": None,
                        "eval_score": None,
                        "variant_text": f"Artifact: {label} (snapshot_id={snapshot_id})",
                        "variant_type": "artifact",
                        "metadata": {"snapshot_id": snapshot_id, "label": label},
                    })
    
    # Extract from learning curve points (fallback)
    if job_result.get("result") and isinstance(job_result["result"], dict):
        learning_curve = job_result["result"].get("learning_curve", [])
        if isinstance(learning_curve, list):
            for point in learning_curve:
                if isinstance(point, dict):
                    score = point.get("performance") or point.get("score")
                    metadata = point.get("metadata", {})
                    
                    # Try to extract template/transformation from metadata
                    variant_text = None
                    variant_type = None
                    
                    # Check for template in metadata
                    if "template" in metadata:
                        template = metadata["template"]
                        if isinstance(template, dict):
                            # Extract sections content
                            sections = template.get("sections", [])
                            if sections:
                                variant_text = "\n".join([
                                    f"[{s.get('role', '?')}]: {s.get('content', '')}"
                                    for s in sections if isinstance(s, dict)
                                ])
                                variant_type = "template"
                    
                    # Check for transformation in metadata
                    if not variant_text and "transformation" in metadata:
                        transformation = metadata["transformation"]
                        if isinstance(transformation, dict):
                            text_replacements = transformation.get("text_replacements", [])
                            example_injections = transformation.get("example_injections", [])
                            parts = []
                            if text_replacements:
                                parts.append("Text Replacements:")
                                for tr in text_replacements:
                                    parts.append(f"  '{tr.get('old_text', '')}' -> '{tr.get('new_text', '')}'")
                            if example_injections:
                                parts.append("Example Injections:")
                                for ei in example_injections:
                                    examples = ei.get("examples", [])
                                    parts.append(f"  {len(examples)} examples")
                            if parts:
                                variant_text = "\n".join(parts)
                                variant_type = "transformation"
                    
                    # Check for archive payload in metadata
                    if not variant_text and "archive_payload" in metadata:
                        payload = metadata["archive_payload"]
                        if isinstance(payload, dict):
                            obj = payload.get("object", {})
                            if isinstance(obj, dict):
                                if obj.get("type") == "template":
                                    template_data = obj.get("data", {})
                                    sections = template_data.get("sections", [])
                                    if sections:
                                        variant_text = "\n".join([
                                            f"[{s.get('role', '?')}]: {s.get('content', '')}"
                                            for s in sections if isinstance(s, dict)
                                        ])
                                        variant_type = "template"
                                elif obj.get("type") == "transformation":
                                    transformation_data = obj.get("data", {})
                                    text_replacements = transformation_data.get("text_replacements", [])
                                    example_injections = transformation_data.get("example_injections", [])
                                    parts = []
                                    if text_replacements:
                                        parts.append("Text Replacements:")
                                        for tr in text_replacements:
                                            parts.append(f"  '{tr.get('old_text', '')}' -> '{tr.get('new_text', '')}'")
                                    if example_injections:
                                        parts.append("Example Injections:")
                                        for ei in example_injections:
                                            examples = ei.get("examples", [])
                                            parts.append(f"  {len(examples)} examples")
                                    if parts:
                                        variant_text = "\n".join(parts)
                                        variant_type = "transformation"
                    
                    # Fallback: try to extract from system_name or other metadata
                    if not variant_text:
                        system_name = metadata.get("system_name")
                        template_id = metadata.get("template_id")
                        if system_name or template_id:
                            variant_text = f"System: {system_name or template_id}"
                            variant_type = "unknown"
                    
                    if score is not None or variant_text:
                        variants.append({
                            "score": score,
                            "eval_score": metadata.get("eval_score") or metadata.get("validation_score"),
                            "variant_text": variant_text or "N/A",
                            "variant_type": variant_type or "unknown",
                            "metadata": metadata,
                            "rollout_count": point.get("rollout_count"),
                        })
    
    # Extract from artifacts (archive summaries, best_prompt files)
    if job_result.get("result") and isinstance(job_result["result"], dict):
        artifacts = job_result["result"].get("artifacts", {})
        print(f"  [DEBUG] Checking artifacts for variants. Artifacts keys: {list(artifacts.keys()) if isinstance(artifacts, dict) else 'not dict'}")
        
        # Check for archive summary
        if "archive_summary" in artifacts:
            print(f"  [DEBUG] Found archive_summary in artifacts")
            archive_summary = artifacts["archive_summary"]
            if isinstance(archive_summary, list):
                for item in archive_summary:
                    if isinstance(item, dict):
                        score = item.get("score")
                        obj = item.get("object", {})
                        variant_text = None
                        variant_type = None
                        
                        if isinstance(obj, dict):
                            if obj.get("type") == "template":
                                # ✅ ADD: Try messages field first (new field), then fallback to sections
                                messages = item.get("messages", [])
                                if messages:
                                    variant_text = "\n".join([
                                        f"[{msg.get('role', '?')}]: {msg.get('content', '')}"
                                        for msg in messages if isinstance(msg, dict)
                                    ])
                                    variant_type = "template"
                                else:
                                    template_data = obj.get("data", {})
                                    sections = template_data.get("sections", [])
                                    if sections:
                                        variant_text = "\n".join([
                                            f"[{s.get('role', '?')}]: {s.get('content', '')}"
                                            for s in sections if isinstance(s, dict)
                                        ])
                                        variant_type = "template"
                            elif obj.get("type") == "transformation":
                                transformation_data = obj.get("data", {})
                                text_replacements = transformation_data.get("text_replacements", [])
                                example_injections = transformation_data.get("example_injections", [])
                                parts = []
                                if text_replacements:
                                    parts.append("Text Replacements:")
                                    for tr in text_replacements:
                                        parts.append(f"  '{tr.get('old_text', '')}' -> '{tr.get('new_text', '')}'")
                                if example_injections:
                                    parts.append("Example Injections:")
                                    for ei in example_injections:
                                        examples = ei.get("examples", [])
                                        parts.append(f"  {len(examples)} examples")
                                if parts:
                                    variant_text = "\n".join(parts)
                                    variant_type = "transformation"
                        
                        if score is not None or variant_text:
                            variants.append({
                                "score": score,
                                "eval_score": None,
                                "variant_text": variant_text or "N/A",
                                "variant_type": variant_type or "unknown",
                                "metadata": item.get("trace", {}),
                                "rollout_count": None,
                            })
        
        # Check for best_prompt_path and try to load it
        best_prompt_path = artifacts.get("best_prompt_path")
        if best_prompt_path:
            try:
                import json
                prompt_path = Path(best_prompt_path)
                if prompt_path.exists():
                    with open(prompt_path, "r") as f:
                        prompt_data = json.load(f)
                    
                    # Extract template from best prompt file
                    if isinstance(prompt_data, dict):
                        sections = prompt_data.get("sections", [])
                        if sections:
                            variant_text = "\n".join([
                                f"[{s.get('role', '?')}]: {s.get('content', '')}"
                                for s in sections if isinstance(s, dict)
                            ])
                            variants.append({
                                "score": job_result.get("best_score") or job_result.get("candidate1_score"),
                                "eval_score": None,
                                "variant_text": variant_text,
                                "variant_type": "template",
                                "metadata": prompt_data.get("metadata", {}),
                                "rollout_count": None,
                            })
            except Exception as e:
                print(f"  [DEBUG] Could not load best_prompt file: {e}")
    
    return variants


def print_aggregate_results(all_results: List[Dict]) -> None:
    """Print aggregate results table matching original script format."""
    # Print evaluated variants for each task FIRST
    print("\n" + "=" * 150)
    print("EVALUATED VARIANTS BY TASK")
    print("=" * 150)
    
    for result in all_results:
        task_name = result.get("name", "Unknown").replace("GEPA ", "")
        job = result.get("jobs", [{}])[0] if result.get("jobs") else {}
        
        # Extract variants - pass the full job_result dict INCLUDING backend_job_data
        job_result_for_variants = {
            "result": job.get("result"),
            "best_score": job.get("best_score"),
            "candidate1_score": job.get("candidate1_score"),
            "backend_job_data": job.get("backend_job_data"),  # ✅ ADD: Include backend_job_data so attempted_candidates can be extracted
        }
        variants = extract_variants(job_result_for_variants)
        
        if not variants:
            print(f"\n{task_name}: No variants found in results")
            continue
        
        # Sort by score (descending, best first), then by eval_score if available
        variants.sort(key=lambda v: (
            v.get("eval_score") if v.get("eval_score") is not None else v.get("score") or -1,
            v.get("score") or -1
        ), reverse=True)
        
        print(f"\n{task_name} ({len(variants)} variant(s)):")
        print("-" * 150)
        
        for idx, variant in enumerate(variants, 1):
            score = variant.get("score")
            eval_score = variant.get("eval_score")
            variant_text = variant.get("variant_text", "N/A")
            variant_type = variant.get("variant_type", "unknown")
            rollout_count = variant.get("rollout_count")
            
            # ✅ REMOVED: Truncation - show full content as requested
            # No truncation - display complete variant text
            
            score_str = f"{score:.4f}" if score is not None else "N/A"
            eval_score_str = f"{eval_score:.4f}" if eval_score is not None else "N/A"
            rollout_str = f"rollout={rollout_count}" if rollout_count is not None else ""
            
            print(f"\n  Variant {idx} ({variant_type}):")
            print(f"    Score: {score_str} | Eval Score: {eval_score_str} | {rollout_str}")
            print(f"    Content:")
            # Indent each line of variant_text
            for line in variant_text.split("\n"):
                print(f"      {line}")
    
    print("\n" + "=" * 150)
    
    # Print aggregate stats table AFTER variants
    print("\n" + "=" * 150)
    print("AGGREGATE STATS ACROSS ALL TASKS (synth_gepa)")
    print("=" * 150)
    print()
    
    # ✅ ADD: Prominently display transformation counts summary BEFORE the table
    print("🔢 TRANSFORMATIONS EVALUATED (Key Metric - Check for Early Termination Bug):")
    print("-" * 150)
    transformation_counts: Dict[str, int] = {}
    for result in all_results:
        job = result.get("jobs", [{}])[0] if result.get("jobs") else {}
        if job.get("status") == "completed":
            task_name = result.get("name", "Unknown").replace("GEPA ", "")
            trials = job.get("trials_tried")
            proposal_method = job.get("proposal_method", "synth")
            if trials is not None:
                key = f"{task_name} ({proposal_method})"
                transformation_counts[key] = trials
                # Highlight if suspiciously low (< 10 transformations)
                status_icon = "⚠️ " if trials < 10 else "✅ "
                print(f"  {status_icon}{key:<40} {trials:>3} transformations")
    print("-" * 150)
    print()
    
    print(f"{'Task':<20} {'Policy Model':<25} {'Proposal':<10} {'Baseline':<12} {'Candidate 1':<14} {'Lift':<12} {'Rollouts':<10} {'Transformations':<15} {'Tokens':<12} {'Time':<10} {'Eval N':<8}")
    print("-" * 150)
    
    # Group results by proposal method (strategy)
    results_by_strategy: Dict[str, List[Dict]] = {}
    
    for result in all_results:
        task_name = result.get("name", "Unknown").replace("GEPA ", "")
        
        # Extract from first job
        job = result.get("jobs", [{}])[0] if result.get("jobs") else {}
        job_status = job.get("status", "unknown")
        
        # Skip failed jobs in aggregate table (they have no valid scores - may be stale data)
        if job_status == "failed":
            error_msg = job.get("error") or "Unknown error"
            # Truncate long error messages for table display
            error_display = error_msg[:20] + "..." if len(error_msg) > 20 else error_msg
            print(f"{task_name:<20} {'FAILED':<25} {'':<10} {'FAILED':<12} {'FAILED':<14} {'':<12} {'':<10} {'':<10} {'':<12} {'':<10} {'':<8}")
            # Print full error message below the table row
            print(f"  └─ Error: {error_msg}")
            continue
        
        baseline = job.get("baseline_score")
        candidate1 = job.get("candidate1_score")
        lift = job.get("candidate1_lift")
        rollouts = job.get("total_rollouts")
        trials = job.get("trials_tried")
        tokens = job.get("total_tokens")
        total_time = job.get("total_time")
        eval_n = job.get("eval_seeds_n")
        policy_model = job.get("policy_model", "N/A")
        
        # ✅ ADD: Extract proposal method (proposer_mode) - already extracted in extract_results
        proposal_method = job.get("proposal_method", "synth")  # Default to "synth" if not found
        
        # Group by strategy
        if proposal_method not in results_by_strategy:
            results_by_strategy[proposal_method] = []
        results_by_strategy[proposal_method].append({
            "task_name": task_name,
            "baseline": baseline,
            "candidate1": candidate1,
            "lift": lift,
            "rollouts": rollouts,
            "trials": trials,
            "tokens": tokens,
            "total_time": total_time,
            "eval_n": eval_n,
            "policy_model": policy_model,
            "proposal_method": proposal_method,
        })
        
        baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
        candidate1_str = f"{candidate1:.4f}" if candidate1 is not None else "N/A"
        lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
        rollouts_str = str(rollouts) if rollouts is not None else "N/A"
        # ✅ PROMINENT: Highlight transformation count - add warning if suspiciously low
        if trials is not None:
            trials_str = f"{trials}" if trials >= 10 else f"⚠️ {trials} ⚠️"
        else:
            trials_str = "N/A"
        tokens_str = str(tokens) if tokens is not None else "N/A"
        time_str = f"{total_time:.1f}s" if total_time is not None else "N/A"
        eval_n_str = str(eval_n) if eval_n is not None else "N/A"
        policy_model_str = str(policy_model) if policy_model else "N/A"
        proposal_str = str(proposal_method) if proposal_method else "synth"
        
        print(f"{task_name:<20} {policy_model_str:<25} {proposal_str:<10} {baseline_str:<12} {candidate1_str:<14} {lift_str:<12} {rollouts_str:<10} {trials_str:<15} {tokens_str:<12} {time_str:<10} {eval_n_str:<8}")
    
    # Print totals and averages (only count successful jobs)
    print("-" * 150)
    successful_results = [
        r for r in all_results 
        if r.get("jobs") and r.get("jobs", [{}])[0].get("status") == "completed"
    ]
    if successful_results:
        avg_baseline = sum(r.get("jobs", [{}])[0].get("baseline_score") or 0 for r in successful_results) / len(successful_results)
        avg_candidate1 = sum(r.get("jobs", [{}])[0].get("candidate1_score") or 0 for r in successful_results) / len(successful_results)
        avg_lift = sum(r.get("jobs", [{}])[0].get("candidate1_lift") or 0 for r in successful_results) / len(successful_results)
    else:
        avg_baseline = 0
        avg_candidate1 = 0
        avg_lift = 0
    
    total_trials = sum(r.get("trials", 0) or 0 for strategy_results in results_by_strategy.values() for r in strategy_results)
    total_tokens = sum(r.get("tokens", 0) or 0 for strategy_results in results_by_strategy.values() for r in strategy_results)
    total_time_seconds = sum(r.get("total_time", 0) or 0 for strategy_results in results_by_strategy.values() for r in strategy_results)
    
    total_time_str = f"{total_time_seconds/60:.1f}m" if total_time_seconds > 60 else f"{total_time_seconds:.1f}s"
    
    print(f"{'TOTAL':<20} {'':<25} {'':<12} {'':<14} {'':<12} {'':<10} {total_trials:<10} {total_tokens:<12} {total_time_str:<10} {'':<8}")
    print(f"{'AVERAGE':<20} {'':<25} {avg_baseline:.4f}     {avg_candidate1:.4f}      {avg_lift:+.4f}")
    print("-" * 150)
    
    # ✅ ADD: Print aggregate summary statistics
    print("\n" + "=" * 150)
    print("GEPA AGGREGATE STATISTICS")
    print("=" * 150)
    print()
    
    if successful_results:
        # Calculate averages across all successful experiments
        baselines = [r.get("jobs", [{}])[0].get("baseline_score") for r in successful_results if r.get("jobs", [{}])[0].get("baseline_score") is not None]
        candidates = [r.get("jobs", [{}])[0].get("candidate1_score") for r in successful_results if r.get("jobs", [{}])[0].get("candidate1_score") is not None]
        lifts = [r.get("jobs", [{}])[0].get("candidate1_lift") for r in successful_results if r.get("jobs", [{}])[0].get("candidate1_lift") is not None]
        rollouts_list = [r.get("jobs", [{}])[0].get("total_rollouts") for r in successful_results if r.get("jobs", [{}])[0].get("total_rollouts") is not None]
        times_list = [r.get("jobs", [{}])[0].get("total_time") for r in successful_results if r.get("jobs", [{}])[0].get("total_time") is not None]
        trials_list = [r.get("jobs", [{}])[0].get("trials_tried") for r in successful_results if r.get("jobs", [{}])[0].get("trials_tried") is not None]
        
        avg_baseline_final = sum(baselines) / len(baselines) if baselines else 0
        avg_candidate_final = sum(candidates) / len(candidates) if candidates else 0
        avg_lift_final = sum(lifts) / len(lifts) if lifts else 0
        avg_rollouts = sum(rollouts_list) / len(rollouts_list) if rollouts_list else 0
        avg_time = sum(times_list) / len(times_list) if times_list else 0
        avg_trials = sum(trials_list) / len(trials_list) if trials_list else 0
        
        print(f"Total Experiments: {len(successful_results)}")
        print(f"Average Baseline Score: {avg_baseline_final:.4f} ({avg_baseline_final*100:.2f}%)")
        print(f"Average Best Score: {avg_candidate_final:.4f} ({avg_candidate_final*100:.2f}%)")
        print(f"Average Lift: {avg_lift_final:+.4f} ({avg_lift_final*100:+.2f}%)")
        if rollouts_list:
            print(f"Average Rollouts: {avg_rollouts:.1f}")
        if times_list:
            print(f"Average Time: {avg_time:.1f}s ({avg_time/60:.1f}min)")
        if trials_list:
            print(f"Average Transformations: {avg_trials:.1f}")
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
    
    print("=" * 150)
    print()
    
    # ✅ ADD: Print breakdown by strategy (proposer mode)
    print("=" * 150)
    print("BREAKDOWN BY STRATEGY (Proposer Mode)")
    print("=" * 150)
    print()
    
    for strategy in sorted(results_by_strategy.keys()):
        strategy_results = results_by_strategy[strategy]
        if not strategy_results:
            continue
        
        print(f"\n{strategy.upper()} Strategy ({len(strategy_results)} task(s)):")
        print("-" * 150)
        print(f"{'Task':<20} {'Policy Model':<25} {'Baseline':<12} {'Candidate 1':<14} {'Lift':<12} {'Rollouts':<10} {'Transformations':<15} {'Time':<10}")
        print("-" * 150)
        
        strategy_total_trials = 0
        strategy_total_tokens = 0
        strategy_total_time = 0
        strategy_baselines = []
        strategy_candidates = []
        strategy_lifts = []
        
        for r in strategy_results:
            task_name = r["task_name"]
            baseline = r["baseline"]
            candidate1 = r["candidate1"]
            lift = r["lift"]
            rollouts = r["rollouts"]
            total_time = r["total_time"]
            policy_model = r["policy_model"]
            
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            candidate1_str = f"{candidate1:.4f}" if candidate1 is not None else "N/A"
            lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
            rollouts_str = str(rollouts) if rollouts is not None else "N/A"
            # ✅ PROMINENT: Highlight transformation count in strategy breakdown too
            trials = r["trials"]
            if trials is not None:
                trials_str = f"{trials}" if trials >= 10 else f"⚠️ {trials} ⚠️"
            else:
                trials_str = "N/A"
            time_str = f"{total_time:.1f}s" if total_time is not None else "N/A"
            policy_model_str = str(policy_model) if policy_model else "N/A"
            
            print(f"{task_name:<20} {policy_model_str:<25} {baseline_str:<12} {candidate1_str:<14} {lift_str:<12} {rollouts_str:<10} {trials_str:<15} {time_str:<10}")
            
            if r["trials"]:
                strategy_total_trials += r["trials"]
            if r["tokens"]:
                strategy_total_tokens += r["tokens"] or 0
            if r["total_time"]:
                strategy_total_time += r["total_time"] or 0
            if baseline is not None:
                strategy_baselines.append(baseline)
            if candidate1 is not None:
                strategy_candidates.append(candidate1)
            if lift is not None:
                strategy_lifts.append(lift)
        
        print("-" * 150)
        strategy_avg_baseline = sum(strategy_baselines) / len(strategy_baselines) if strategy_baselines else 0
        strategy_avg_candidate = sum(strategy_candidates) / len(strategy_candidates) if strategy_candidates else 0
        strategy_avg_lift = sum(strategy_lifts) / len(strategy_lifts) if strategy_lifts else 0
        strategy_time_str = f"{strategy_total_time/60:.1f}m" if strategy_total_time > 60 else f"{strategy_total_time:.1f}s"
        
        # Calculate average transformations for this strategy
        strategy_transformations = [r["trials"] for r in strategy_results if r["trials"] is not None]
        avg_transformations = sum(strategy_transformations) / len(strategy_transformations) if strategy_transformations else 0
        avg_transformations_str = f"{avg_transformations:.1f}" if avg_transformations > 0 else "N/A"
        
        # Calculate average rollouts for this strategy
        strategy_rollouts = [r["rollouts"] for r in strategy_results if r["rollouts"] is not None]
        avg_rollouts = sum(strategy_rollouts) / len(strategy_rollouts) if strategy_rollouts else 0
        avg_rollouts_str = f"{avg_rollouts:.1f}" if avg_rollouts > 0 else "N/A"
        
        print(f"{'AVERAGE':<20} {'':<25} {strategy_avg_baseline:.4f}     {strategy_avg_candidate:.4f}      {strategy_avg_lift:+.4f}     {avg_rollouts_str:<10} {avg_transformations_str:<15} {strategy_time_str:<10}")
        print("-" * 150)
        
        # Print strategy summary stats
        if strategy_lifts:
            positive_count = len([l for l in strategy_lifts if l > 0])
            negative_count = len([l for l in strategy_lifts if l < 0])
            min_lift = min(strategy_lifts)
            max_lift = max(strategy_lifts)
            print(f"  Summary: {positive_count}/{len(strategy_lifts)} positive lifts | Range: {min_lift:+.4f} to {max_lift:+.4f}")
        print()
    
    print("\n" + "=" * 150)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run GEPA experiments via experiment queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (worker concurrency=5)
  python run_gepa_parallel_experiments.py
  
  # Run with custom worker concurrency (allows more parallel experiments)
  python run_gepa_parallel_experiments.py --worker-concurrency 10
  
  # Set parallelism per experiment in YAML config:
  # benchmarks:
  #   banking77_synth:
  #     parallelism: 2  # Allow 2 jobs from this experiment to run simultaneously
        """
    )
    parser.add_argument(
        "--worker-concurrency",
        type=int,
        default=None,
        help="Override worker concurrency (default: 5, or EXPERIMENT_QUEUE_WORKER_CONCURRENCY env var). "
             "Set this to the number of experiments you want to run in parallel."
    )
    parser.add_argument(
        "--auto-concurrency",
        action="store_true",
        help="Automatically set worker concurrency to match number of experiments (requires restarting worker)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GEPA Parallel Experiments via Queue")
    print("=" * 80)
    print()
    
    # Check prerequisites
    print("Checking prerequisites...")
    if not check_redis():
        sys.exit(1)
    
    # Load config early to determine number of experiments
    try:
        yaml_config = load_yaml_config()
        benchmarks = yaml_config["benchmarks"]
        num_experiments = len(benchmarks)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Determine worker concurrency
    worker_concurrency = args.worker_concurrency
    if args.auto_concurrency:
        worker_concurrency = num_experiments
        print(f"🔧 Auto-setting worker concurrency to {worker_concurrency} (number of experiments)")
    
    # Check/start worker with appropriate concurrency
    if worker_concurrency is not None:
        print(f"🔧 Requested worker concurrency: {worker_concurrency}")
        # Check if worker is running
        try:
            from synth_ai.cli.queue import _get_running_workers
            workers = _get_running_workers()
            if workers:
                # Worker is running - warn user they may need to restart
                print(f"⚠️  Worker is already running. To change concurrency to {worker_concurrency}:")
                print(f"   1. Stop current worker: synth-ai queue stop")
                print(f"   2. Start with new concurrency: synth-ai queue start --concurrency {worker_concurrency}")
                print(f"   Or set EXPERIMENT_QUEUE_WORKER_CONCURRENCY={worker_concurrency} and restart")
                print()
                # Only prompt if running interactively (stdin is a TTY)
                if sys.stdin.isatty():
                    response = input("Continue with current worker? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Exiting. Please restart worker with correct concurrency.")
                        sys.exit(1)
                else:
                    print("⚠️  Non-interactive mode: continuing with current worker concurrency")
            else:
                # No worker running - start one with requested concurrency
                print(f"🚀 Starting worker with concurrency={worker_concurrency}...")
                import subprocess
                result = subprocess.run(
                    ["synth-ai", "queue", "start", "--concurrency", str(worker_concurrency), "--background"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"❌ Failed to start worker: {result.stderr}")
                    print(f"   Please start manually: synth-ai queue start --concurrency {worker_concurrency}")
                    sys.exit(1)
                print("✅ Worker started")
                import time
                time.sleep(2)  # Give worker time to start
        except Exception as e:
            print(f"⚠️  Could not manage worker: {e}")
            print("   Please ensure worker is running with appropriate concurrency")
    
    if not check_queue_worker():
        sys.exit(1)
    
    print()
    
    # Config already loaded above
    benchmarks = yaml_config["benchmarks"]
    defaults = yaml_config["defaults"]
    
    print(f"✅ Loaded config from {CONFIG_FILE}")
    print(f"  Found {len(benchmarks)} benchmark(s):")
    for benchmark_name, benchmark_config in benchmarks.items():
        rollout_limit = benchmark_config.get("rollout_limit", "N/A")
        parallelism = benchmark_config.get("parallelism", 1)
        model_config = benchmark_config.get("model", {})
        model_str = f"{model_config.get('provider', '?')}/{model_config.get('model', '?')}" if model_config else "N/A"
        parallelism_str = f", parallelism={parallelism}" if parallelism > 1 else ""
        print(f"    - {benchmark_name}: {rollout_limit} rollouts, model={model_str}{parallelism_str}")
    print(f"  Global defaults: time_limit={defaults.get('time_limit_seconds', 600)}s, max_trials={defaults.get('max_trials', 50)}, max_cost=${defaults.get('max_cost_usd', 10.0)}")
    if worker_concurrency:
        print(f"  Worker concurrency: {worker_concurrency} (allows {worker_concurrency} experiments to run simultaneously)")
    print()
    
    # Submit experiments
    print("Submitting experiments to queue...")
    experiment_ids = submit_experiments(yaml_config)
    
    if not experiment_ids:
        print("❌ No experiments submitted")
        sys.exit(1)
    
    print(f"\n✅ Submitted {len(experiment_ids)} experiments")
    print()
    
    # Poll for completion (in parallel)
    print("Polling for experiment completion...")
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
                print(f"❌ Error polling experiment {exp_id[:12]}...: {e}")
    
    print()
    
    # Extract detailed results
    print("Extracting results...")
    all_results = []
    for result in completed_results:
        detailed = extract_results(result["experiment_id"])
        all_results.append(detailed)
    
    # Save results to files BEFORE printing (so we can capture output)
    import time as time_module
    timestamp = time_module.strftime("%Y%m%d_%H%M%S")
    readout_file = COMPARISONS_DIR / f"synth_gepa_comparison_readout_{timestamp}.txt"
    json_file = COMPARISONS_DIR / f"synth_gepa_comparison_results_{timestamp}.json"
    
    try:
        # Save readout (text summary) - capture what will be printed
        import io
        from contextlib import redirect_stdout
        
        # Capture output
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            print_aggregate_results(all_results)
        readout_content = output_buffer.getvalue()
        
        # Write to file
        with open(readout_file, "w") as f:
            f.write(readout_content)
        
        # Also print to console
        print(readout_content)
        print(f"\n✅ Saved readout to: {readout_file}")
        
        # Save full JSON results
        import json
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✅ Saved JSON results to: {json_file}")
    except Exception as e:
        # Fallback: print anyway even if file save fails
        print_aggregate_results(all_results)
        print(f"\n⚠️  Could not save results files: {e}")
    
    print("\n✅ All experiments completed")


if __name__ == "__main__":
    main()

