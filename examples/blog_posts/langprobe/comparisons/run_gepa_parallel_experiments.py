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
    resolved_benchmarks = {}
    for benchmark_name, benchmark_config in benchmarks.items():
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
    
    # Apply any other per-benchmark overrides (flatten nested dicts with dot notation)
    for key, value in benchmark_config.items():
        if key not in ("config_path", "rollout_limit", "time_limit_seconds", "max_trials", 
                      "max_cost_usd", "gepa_population", "model"):
            # Assume it's a config override path (e.g., "prompt_learning.gepa.mutation.rate")
            if isinstance(value, dict):
                # Flatten nested dicts
                for nested_key, nested_value in value.items():
                    config_overrides[f"{key}.{nested_key}"] = nested_value
            else:
                config_overrides[key] = value
    
    # Format benchmark name for display (capitalize first letter)
    display_name = benchmark_name[0].upper() + benchmark_name[1:] if benchmark_name else benchmark_name
    
    request = ExperimentSubmitRequest(
        name=f"GEPA {display_name}",
        description=f"GEPA optimization for {display_name}",
        parallelism=1,  # One job per experiment
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
                        # Always show status_json if it has any data (even just policy/environment)
                        # This ensures we see policy/environment even before progress events arrive
                        status_line = f" | {formatted}"
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
    
    try:
        # Fetch job details (includes best_snapshot)
        job_url = f"{backend_url}/prompt-learning/online/jobs/{backend_job_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(job_url, headers=headers, timeout=10.0)
        if response.status_code == 200:
            job_data = response.json()
            
            # Also fetch artifacts list
            artifacts_url = f"{backend_url}/prompt-learning/online/jobs/{backend_job_id}/artifacts"
            artifacts_response = requests.get(artifacts_url, headers=headers, timeout=10.0)
            artifacts = []
            if artifacts_response.status_code == 200:
                artifacts = artifacts_response.json()
            
            return {
                "job": job_data,
                "artifacts": artifacts,
                "best_snapshot": job_data.get("best_snapshot"),
                "best_snapshot_id": job_data.get("best_snapshot_id"),
            }
        else:
            print(f"  [WARN] Backend API returned {response.status_code} for job {backend_job_id}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  [WARN] Failed to fetch backend job details for {backend_job_id}: {e}")
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
                print(f"  [DEBUG] Successfully fetched backend job data (has best_snapshot={backend_job_data.get('best_snapshot') is not None}, artifacts={len(backend_job_data.get('artifacts', []))})")
        
        # Extract from job.result (ResultSummary dict)
        if job.result:
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
            
            # Total tokens: prefer status_json, then stats, then sum from trials
            total_tokens = None
            if job.status_json and isinstance(job.status_json, dict):
                status_data = job.status_json
                total_tokens = status_data.get("rollout_tokens_used") or status_data.get("total_tokens")
            
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
            
            # Extract eval_seeds_n from stats, artifacts, or config
            eval_seeds_n = (
                stats.get("eval_seeds_n") 
                or stats.get("eval_n")
                or stats.get("validation_seeds_n")
                or artifacts.get("eval_seeds_n")
            )
            
            # If eval_seeds_n not found, try to extract from learning curve metadata
            if not eval_seeds_n and learning_curve:
                # Check first point metadata
                if learning_curve and learning_curve[0].get("metadata"):
                    metadata = learning_curve[0]["metadata"]
                    eval_seeds_n = metadata.get("eval_seeds_n") or metadata.get("eval_n")
            
            # If still not found, try to get from config
            if not eval_seeds_n:
                try:
                    import tomllib
                    config_path = Path(job.config_path)
                    if config_path.exists():
                        with config_path.open("rb") as f:
                            config_data = tomllib.load(f)
                        pl_config = config_data.get("prompt_learning", {})
                        gepa_config = pl_config.get("gepa", {})
                        eval_config = gepa_config.get("eval") or pl_config.get("eval", {})
                        if isinstance(eval_config, dict):
                            eval_seeds = eval_config.get("seeds") or eval_config.get("eval_seeds")
                            if isinstance(eval_seeds, list):
                                eval_seeds_n = len(eval_seeds)
                            elif isinstance(eval_seeds, int):
                                eval_seeds_n = eval_seeds
                except Exception:
                    pass
            
            job_result["eval_seeds_n"] = eval_seeds_n
            
            # Extract policy_model from stats or artifacts
            policy_model = (
                stats.get("policy_model")
                or artifacts.get("policy_model")
                or stats.get("model")
            )
            job_result["policy_model"] = policy_model
            
            # Count trials_tried from database if not already set
            if job_result.get("trials_tried") is None:
                if learning_curve:
                    job_result["trials_tried"] = len(learning_curve)
                elif stats.get("trials_tried"):
                    job_result["trials_tried"] = stats.get("trials_tried")
            
            # Try to get eval/trial counts from backend job data
            # Backend stores actual eval runs and optimization trials separately
            if backend_job_data:
                backend_job = backend_job_data.get("job", {})
                # Check if backend provides eval/trial counts in metadata or stats
                backend_metadata = backend_job.get("metadata", {})
                backend_stats = backend_metadata.get("stats", {})
                
                # Try to extract from backend metadata
                if backend_stats.get("eval_n") is not None:
                    job_result["eval_seeds_n"] = backend_stats["eval_n"]
                    print(f"  [DEBUG] Found eval_n={backend_stats['eval_n']} from backend metadata")
                if backend_stats.get("trials_tried") is not None and job_result.get("trials_tried") is None:
                    job_result["trials_tried"] = backend_stats["trials_tried"]
                    print(f"  [DEBUG] Found trials_tried={backend_stats['trials_tried']} from backend metadata")
            
            # Count eval N and trial N from DB trials filtered by phase metadata
            # NOTE: Trials in experiment queue DB are learning curve checkpoints, not actual eval/optimization trials
            # They don't have phase metadata, so we can't distinguish eval vs optimization from DB alone
            # The backend stores the actual trial/eval data separately
            eval_n_from_db = None
            trial_n_from_db = None
            try:
                if hasattr(job, 'trials') and job.trials:
                    completed_trials = [t for t in job.trials if t.status.value == "completed"]
                    
                    # Debug: print trial metadata to understand structure
                    if completed_trials:
                        sample_trial = completed_trials[0]
                        print(f"  [DEBUG] Sample trial metadata keys: {list(sample_trial.metadata_json.keys()) if sample_trial.metadata_json else 'None'}")
                        print(f"  [DEBUG] Sample trial metadata: {sample_trial.metadata_json}")
                    
                    # Count eval trials (validation runs) - check for phase in metadata
                    eval_phases = ("validation_baseline", "validation_topk", "eval")
                    eval_trials = [
                        t for t in completed_trials
                        if t.metadata_json and isinstance(t.metadata_json, dict)
                        and any(t.metadata_json.get("phase", "").startswith(phase) for phase in eval_phases)
                    ]
                    if eval_trials:
                        eval_n_from_db = len(eval_trials)
                        print(f"  [DEBUG] Found {eval_n_from_db} eval trials from DB (phase=validation_*/eval) for job {job.job_id}")
                    
                    # Count optimization trials (pattern_eval, optimization, or no phase/default)
                    optimization_phases = ("pattern_eval", "optimization", "transformation_limits", "limits")
                    optimization_trials = [
                        t for t in completed_trials
                        if not t.metadata_json  # No metadata = default optimization trial
                        or not isinstance(t.metadata_json, dict)
                        or t.metadata_json.get("phase") in optimization_phases
                        or not t.metadata_json.get("phase")  # No phase = default optimization trial
                        or not any(t.metadata_json.get("phase", "").startswith(phase) for phase in eval_phases)  # Not an eval phase
                    ]
                    if optimization_trials:
                        trial_n_from_db = len(optimization_trials)
                        print(f"  [DEBUG] Found {trial_n_from_db} optimization trials from DB for job {job.job_id}")
            except Exception as e:
                print(f"  [DEBUG] Could not count eval/trial N from DB trials: {e}")
            
            # Use DB counts if available and backend didn't provide them
            if eval_n_from_db is not None and job_result.get("eval_seeds_n") is None:
                job_result["eval_seeds_n"] = eval_n_from_db
            if trial_n_from_db is not None and job_result.get("trials_tried") is None:
                job_result["trials_tried"] = trial_n_from_db
        
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
    if backend_job_data:
        # Extract from best_snapshot
        best_snapshot = backend_job_data.get("best_snapshot")
        if best_snapshot and isinstance(best_snapshot, dict):
            messages = best_snapshot.get("messages", [])
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
        
        # Check for archive summary
        if "archive_summary" in artifacts:
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
    print("\n" + "=" * 150)
    print("AGGREGATE STATS ACROSS ALL TASKS (synth_gepa)")
    print("=" * 150)
    print()
    print(f"{'Task':<20} {'Policy Model':<25} {'Baseline':<12} {'Candidate 1':<14} {'Lift':<12} {'Rollouts':<10} {'Trials':<10} {'Tokens':<12} {'Time':<10} {'Eval N':<8}")
    print("-" * 150)
    
    total_trials = 0
    total_tokens = 0
    total_time_seconds = 0
    
    for result in all_results:
        task_name = result.get("name", "Unknown").replace("GEPA ", "")
        
        # Extract from first job
        job = result.get("jobs", [{}])[0] if result.get("jobs") else {}
        baseline = job.get("baseline_score")
        candidate1 = job.get("candidate1_score")
        lift = job.get("candidate1_lift")
        rollouts = job.get("total_rollouts")
        trials = job.get("trials_tried")
        tokens = job.get("total_tokens")
        total_time = job.get("total_time")
        eval_n = job.get("eval_seeds_n")
        policy_model = job.get("policy_model", "N/A")
        
        # Accumulate totals
        if trials:
            total_trials += trials
        if tokens:
            total_tokens += tokens or 0
        if total_time:
            total_time_seconds += total_time
        
        baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
        candidate1_str = f"{candidate1:.4f}" if candidate1 is not None else "N/A"
        lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
        rollouts_str = str(rollouts) if rollouts is not None else "N/A"
        trials_str = str(trials) if trials is not None else "N/A"
        tokens_str = str(tokens) if tokens is not None else "N/A"
        time_str = f"{total_time:.1f}s" if total_time is not None else "N/A"
        eval_n_str = str(eval_n) if eval_n is not None else "N/A"
        policy_model_str = str(policy_model) if policy_model else "N/A"
        
        print(f"{task_name:<20} {policy_model_str:<25} {baseline_str:<12} {candidate1_str:<14} {lift_str:<12} {rollouts_str:<10} {trials_str:<10} {tokens_str:<12} {time_str:<10} {eval_n_str:<8}")
    
    # Print totals and averages
    print("-" * 150)
    avg_baseline = sum(r.get("jobs", [{}])[0].get("baseline_score") or 0 for r in all_results if r.get("jobs")) / len(all_results) if all_results else 0
    avg_candidate1 = sum(r.get("jobs", [{}])[0].get("candidate1_score") or 0 for r in all_results if r.get("jobs")) / len(all_results) if all_results else 0
    avg_lift = sum(r.get("jobs", [{}])[0].get("candidate1_lift") or 0 for r in all_results if r.get("jobs")) / len(all_results) if all_results else 0
    
    total_time_str = f"{total_time_seconds/60:.1f}m" if total_time_seconds > 60 else f"{total_time_seconds:.1f}s"
    
    print(f"{'TOTAL':<20} {'':<25} {'':<12} {'':<14} {'':<12} {'':<10} {total_trials:<10} {total_tokens:<12} {total_time_str:<10} {'':<8}")
    print(f"{'AVERAGE':<20} {'':<25} {avg_baseline:.4f}     {avg_candidate1:.4f}      {avg_lift:+.4f}")
    print("-" * 150)
    
    # Print evaluated variants for each task
    print("\n" + "=" * 150)
    print("EVALUATED VARIANTS BY TASK")
    print("=" * 150)
    
    for result in all_results:
        task_name = result.get("name", "Unknown").replace("GEPA ", "")
        job = result.get("jobs", [{}])[0] if result.get("jobs") else {}
        
        # Extract variants - pass the full job_result dict
        job_result_for_variants = {
            "result": job.get("result"),
            "best_score": job.get("best_score"),
            "candidate1_score": job.get("candidate1_score"),
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
            
            # Truncate long variant text (show first 200 chars, then truncate)
            max_text_len = 200
            if len(variant_text) > max_text_len:
                # Try to truncate at a newline if possible
                truncated = variant_text[:max_text_len]
                last_newline = truncated.rfind("\n")
                if last_newline > max_text_len * 0.7:  # If newline is reasonably close to max
                    variant_text = truncated[:last_newline] + "\n      ... (truncated)"
                else:
                    variant_text = truncated + "..."
            
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


def main():
    """Main entry point."""
    print("=" * 80)
    print("GEPA Parallel Experiments via Queue")
    print("=" * 80)
    print()
    
    # Check prerequisites
    print("Checking prerequisites...")
    if not check_redis():
        sys.exit(1)
    
    if not check_queue_worker():
        sys.exit(1)
    
    print()
    
    # Load YAML config
    try:
        yaml_config = load_yaml_config()
        benchmarks = yaml_config["benchmarks"]
        defaults = yaml_config["defaults"]
        
        print(f"✅ Loaded config from {CONFIG_FILE}")
        print(f"  Found {len(benchmarks)} benchmark(s):")
        for benchmark_name, benchmark_config in benchmarks.items():
            rollout_limit = benchmark_config.get("rollout_limit", "N/A")
            model_config = benchmark_config.get("model", {})
            model_str = f"{model_config.get('provider', '?')}/{model_config.get('model', '?')}" if model_config else "N/A"
            print(f"    - {benchmark_name}: {rollout_limit} rollouts, model={model_str}")
        print(f"  Global defaults: time_limit={defaults.get('time_limit_seconds', 600)}s, max_trials={defaults.get('max_trials', 50)}, max_cost=${defaults.get('max_cost_usd', 10.0)}")
        print()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
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
    
    # Print aggregate table
    print_aggregate_results(all_results)
    
    print("\n✅ All experiments completed")


if __name__ == "__main__":
    main()

