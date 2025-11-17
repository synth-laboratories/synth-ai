#!/usr/bin/env python3
"""Run DSPy-AI/GEPA-AI library optimizations using synth_gepa_config.yaml.

This script runs the actual DSPy and GEPA-AI library implementations as a reference
comparison, using the same YAML configuration as run_gepa_parallel_experiments.py.

For benchmarks with proposer_mode="dspy", runs DSPy's GEPA optimizer.
For benchmarks with proposer_mode="gepa-ai", runs GEPA-AI library optimizer.
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Load configuration from YAML file (same as run_gepa_parallel_experiments.py)
COMPARISONS_DIR = Path(__file__).parent
CONFIG_FILE = COMPARISONS_DIR / "synth_gepa_config.yaml"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def load_yaml_config() -> Dict:
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config from: {CONFIG_FILE}")
    return config


# Mapping from benchmark names to their DSPy adapter scripts
BENCHMARK_ADAPTERS = {
    "banking77": {
        "dspy": "task_specific/banking77/run_dspy_gepa_banking77.py",
        "gepa_ai": "task_specific/banking77/run_gepa_ai_banking77.py",
        "adapter_module": "task_specific.banking77.dspy_banking77_adapter",
        "gepa_ai_adapter_module": "task_specific.banking77.gepa_ai_banking77_adapter",
    },
    "heartdisease": {
        "dspy": "task_specific/heartdisease/run_dspy_gepa_heartdisease.py",
        "gepa_ai": "task_specific/heartdisease/gepa_ai_heartdisease_adapter",
        "adapter_module": "task_specific.heartdisease.dspy_heartdisease_adapter",
        "gepa_ai_adapter_module": "task_specific.heartdisease.gepa_ai_heartdisease_adapter",
    },
    "hover": {
        "dspy": "task_specific/hover/dspy_hover_adapter",
        "gepa_ai": "task_specific/hover/gepa_ai_hover_adapter",
        "adapter_module": "task_specific.hover.dspy_hover_adapter",
        "gepa_ai_adapter_module": "task_specific.hover.gepa_ai_hover_adapter",
    },
    "pupa": {
        "dspy": "task_specific/pupa/dspy_pupa_adapter",
        "gepa_ai": "task_specific/pupa/gepa_ai_pupa_adapter",
        "adapter_module": "task_specific.pupa.dspy_pupa_adapter",
        "gepa_ai_adapter_module": "task_specific.pupa.gepa_ai_pupa_adapter",
    },
    "ifbench": {
        "dspy": "task_specific/ifbench/dspy_ifbench_adapter",
        "gepa_ai": "task_specific/ifbench/gepa_ai_ifbench_adapter",
        "adapter_module": "task_specific.ifbench.dspy_ifbench_adapter",
        "gepa_ai_adapter_module": "task_specific.ifbench.gepa_ai_ifbench_adapter",
    },
    "hotpotqa": {
        "dspy": "task_specific/hotpotqa/run_dspy_gepa_hotpotqa.py",
        "gepa_ai": "task_specific/hotpotqa/gepa_ai_hotpotqa_adapter",
        "adapter_module": "task_specific.hotpotqa.dspy_hotpotqa_adapter",
        "gepa_ai_adapter_module": "task_specific.hotpotqa.gepa_ai_hotpotqa_adapter",
    },
    "crafter": {
        "dspy": "task_specific/crafter/dspy_crafter_adapter",
        "gepa_ai": "task_specific/crafter/gepa_ai_crafter_adapter",
        "adapter_module": "task_specific.crafter.dspy_crafter_adapter",
        "gepa_ai_adapter_module": "task_specific.crafter.gepa_ai_crafter_adapter",
    },
    "verilog": {
        "dspy": "task_specific/verilog/dspy_verilog_adapter",
        "gepa_ai": "task_specific/verilog/gepa_ai_verilog_adapter",
        "adapter_module": "task_specific.verilog.dspy_verilog_adapter",
        "gepa_ai_adapter_module": "task_specific.verilog.gepa_ai_verilog_adapter",
    },
}


async def run_dspy_library_optimization(
    benchmark_name: str,
    benchmark_config: Dict,
    yaml_config: Dict,
) -> Dict[str, Any]:
    """Run DSPy library GEPA optimization for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "banking77_dspy")
        benchmark_config: Benchmark config from YAML
        yaml_config: Full YAML config
        
    Returns:
        Results dictionary
    """
    # Extract base benchmark name (remove _dspy/_gepa_ai suffix)
    base_name = benchmark_name.replace("_dspy", "").replace("_gepa_ai", "").replace("_synth", "")
    
    # Get adapter info
    adapter_info = BENCHMARK_ADAPTERS.get(base_name)
    if not adapter_info or not adapter_info.get("dspy"):
        return {
            "benchmark": benchmark_name,
            "status": "skipped",
            "reason": f"No DSPy adapter found for {base_name}",
        }
    
    # Get adapter module path
    adapter_module = adapter_info.get("adapter_module")
    if not adapter_module:
        return {
            "benchmark": benchmark_name,
            "status": "skipped",
            "reason": f"No adapter module found for {base_name}",
        }
    
    # Extract config values
    rollout_limit = benchmark_config.get("rollout_limit", 500)
    proposer_mode = benchmark_config.get("proposer_mode", "dspy")
    
    # Extract train_seeds and val_seeds from config overrides
    train_seeds = benchmark_config.get("prompt_learning.gepa.evaluation.train_seeds")
    val_seeds = benchmark_config.get("prompt_learning.gepa.evaluation.val_seeds")
    
    # Extract reflection_minibatch_size (subsample size for quick evaluation)
    reflection_minibatch_size = benchmark_config.get("prompt_learning.gepa.reflection_minibatch_size", 3)
    if isinstance(reflection_minibatch_size, str):
        reflection_minibatch_size = int(reflection_minibatch_size)
    
    # Extract model config
    model_config = benchmark_config.get("model", {})
    provider = model_config.get("provider", "groq")
    model = model_config.get("model", "llama-3.1-8b-instant")
    
    # Get task app URL from config path (need to read TOML)
    config_path_str = benchmark_config.get("config_path")
    if not config_path_str:
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": "Missing config_path in benchmark config",
        }
    
    config_path = REPO_ROOT / config_path_str
    if not config_path.exists():
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": f"Config file not found: {config_path}",
        }
    
    # Read TOML to get task_app_url
    # Yield control to event loop first to allow other tasks to start
    await asyncio.sleep(0)
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {
                "benchmark": benchmark_name,
                "status": "error",
                "error": "tomllib/tomli required to read config",
            }
    
    # Run file I/O in thread pool to avoid blocking event loop
    def _load_toml():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    
    toml_config = await asyncio.to_thread(_load_toml)
    
    task_app_url = (
        toml_config.get("prompt_learning", {})
        .get("task_app_url", "http://127.0.0.1:8102")
    )
    
    print(f"\n🚀 Running DSPy library optimization for {benchmark_name}")
    print(f"   Rollout budget: {rollout_limit}")
    print(f"   Model: {provider}/{model}")
    print(f"   Train seeds: {len(train_seeds) if train_seeds else 'default'}")
    print(f"   Val seeds: {len(val_seeds) if val_seeds else 'default'}")
    print(f"   Reflection minibatch size: {reflection_minibatch_size}")
    
    # Import adapter function dynamically
    # Add langprobe directory to path for imports
    langprobe_dir = REPO_ROOT / "examples" / "blog_posts" / "langprobe"
    if str(langprobe_dir) not in sys.path:
        sys.path.insert(0, str(langprobe_dir))
    
    try:
        # Import the adapter module
        module = __import__(adapter_module, fromlist=[f"run_dspy_gepa_{base_name}"])
        # Try to get the function - some adapters use run_dspy_gepa_{base_name}, others use module-level functions
        try:
            run_func = getattr(module, f"run_dspy_gepa_{base_name}")
        except AttributeError:
            # Try alternative function name pattern
            run_func = getattr(module, f"run_dspy_gepa_{base_name.replace('_', '')}", None)
            if run_func is None:
                # Try just the module name pattern
                func_name = f"run_dspy_gepa_{base_name}"
                # Some adapters might have different naming - try common patterns
                for alt_name in [f"run_dspy_gepa_{base_name}", f"run_{base_name}_dspy_gepa"]:
                    if hasattr(module, alt_name):
                        run_func = getattr(module, alt_name)
                        break
                if run_func is None:
                    raise AttributeError(f"Could not find function run_dspy_gepa_{base_name} or alternatives in {adapter_module}")
    except (ImportError, AttributeError) as e:
        import traceback
        traceback.print_exc()
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": f"Failed to import adapter: {e}",
        }
    
    # Run optimization
    try:
        start_time = time.time()
        
        # Build model string from provider and model
        model_str = f"{provider}/{model}" if provider else model
        
        results = await run_func(
            task_app_url=task_app_url,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_limit,
            reflection_minibatch_size=reflection_minibatch_size,
            output_dir=None,  # Use default output dir
            model=model_str,  # Pass model to adapter
        )
        
        elapsed_time = time.time() - start_time
        
        # Format time nicely for display
        if elapsed_time >= 60:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed_time:.1f}s"
        
        print()
        print("=" * 80)
        print(f"✅ [{benchmark_name}] DSPy optimization COMPLETED in {time_str}")
        if results.get("baseline_score") is not None and results.get("best_score") is not None:
            baseline = results.get("baseline_score", 0.0)
            best = results.get("best_score", 0.0)
            improvement = ((best - baseline) / baseline * 100) if baseline > 0 else 0
            print(f"   Baseline: {baseline:.4f} → Best: {best:.4f} ({improvement:+.1f}% improvement)")
        print("=" * 80)
        print()
        
        return {
            "benchmark": benchmark_name,
            "status": "completed",
            "proposer_mode": proposer_mode,
            "model": f"{provider}/{model}",
            "baseline_score": results.get("baseline_score", 0.0),
            "best_score": results.get("best_score", 0.0),
            "val_score": results.get("val_score"),
            "total_rollouts": results.get("total_rollouts", 0),
            "actual_rollouts": results.get("actual_rollouts"),
            "total_time": elapsed_time,
            "prompt_file": results.get("prompt_file"),
            "results_file": results.get("results_file"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": str(e),
        }


async def run_gepa_ai_library_optimization(
    benchmark_name: str,
    benchmark_config: Dict,
    yaml_config: Dict,
) -> Dict[str, Any]:
    """Run GEPA-AI library optimization for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "banking77_gepa_ai")
        benchmark_config: Benchmark config from YAML
        yaml_config: Full YAML config
        
    Returns:
        Results dictionary
    """
    # Extract base benchmark name (remove _dspy/_gepa_ai suffix)
    base_name = benchmark_name.replace("_dspy", "").replace("_gepa_ai", "").replace("_synth", "")
    
    # Get adapter info
    adapter_info = BENCHMARK_ADAPTERS.get(base_name)
    if not adapter_info or not adapter_info.get("gepa_ai_adapter_module"):
        return {
            "benchmark": benchmark_name,
            "status": "skipped",
            "reason": f"No GEPA-AI adapter module found for {base_name}",
        }
    
    # Extract config values
    rollout_limit = benchmark_config.get("rollout_limit", 500)
    proposer_mode = benchmark_config.get("proposer_mode", "gepa-ai")
    
    # Extract train_seeds and val_seeds from config overrides
    train_seeds = benchmark_config.get("prompt_learning.gepa.evaluation.train_seeds")
    val_seeds = benchmark_config.get("prompt_learning.gepa.evaluation.val_seeds")
    
    # Extract reflection_minibatch_size (subsample size for quick evaluation)
    reflection_minibatch_size = benchmark_config.get("prompt_learning.gepa.reflection_minibatch_size", 3)
    if isinstance(reflection_minibatch_size, str):
        reflection_minibatch_size = int(reflection_minibatch_size)
    
    # Extract model config
    model_config = benchmark_config.get("model", {})
    provider = model_config.get("provider", "groq")
    model = model_config.get("model", "llama-3.1-8b-instant")
    
    # Get task app URL from config path (need to read TOML)
    config_path_str = benchmark_config.get("config_path")
    if not config_path_str:
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": "Missing config_path in benchmark config",
        }
    
    config_path = REPO_ROOT / config_path_str
    if not config_path.exists():
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": f"Config file not found: {config_path}",
        }
    
    # Read TOML to get task_app_url
    # Yield control to event loop first to allow other tasks to start
    await asyncio.sleep(0)
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {
                "benchmark": benchmark_name,
                "status": "error",
                "error": "tomllib/tomli required to read config",
            }
    
    # Run file I/O in thread pool to avoid blocking event loop
    def _load_toml():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    
    toml_config = await asyncio.to_thread(_load_toml)
    
    task_app_url = (
        toml_config.get("prompt_learning", {})
        .get("task_app_url", "http://127.0.0.1:8102")
    )
    
    print(f"\n🚀 Running GEPA-AI library optimization for {benchmark_name}")
    print(f"   Rollout budget: {rollout_limit}")
    print(f"   Model: {provider}/{model}")
    print(f"   Train seeds: {len(train_seeds) if train_seeds else 'default'}")
    print(f"   Val seeds: {len(val_seeds) if val_seeds else 'default'}")
    print(f"   Reflection minibatch size: {reflection_minibatch_size}")
    
    # Import adapter function dynamically
    # Add langprobe directory to path for imports
    langprobe_dir = REPO_ROOT / "examples" / "blog_posts" / "langprobe"
    if str(langprobe_dir) not in sys.path:
        sys.path.insert(0, str(langprobe_dir))
    
    try:
        # Import the GEPA-AI adapter module
        adapter_module = adapter_info.get("gepa_ai_adapter_module")
        if not adapter_module:
            return {
                "benchmark": benchmark_name,
                "status": "skipped",
                "error": f"No GEPA-AI adapter module found for {base_name}",
            }
        
        module = __import__(adapter_module, fromlist=[f"run_gepa_ai_{base_name}"])
        # Try common function name patterns
        func_name = f"run_gepa_ai_{base_name}"
        run_func = None
        if hasattr(module, func_name):
            run_func = getattr(module, func_name)
        else:
            # Try alternative naming patterns
            for alt_name in [
                f"run_gepa_ai_{base_name}",
                f"run_gepa_ai_{base_name.replace('_', '')}",
                f"run_{base_name}_gepa_ai",
            ]:
                if hasattr(module, alt_name):
                    run_func = getattr(module, alt_name)
                    break
            if run_func is None:
                raise AttributeError(f"Could not find function {func_name} or alternatives in {adapter_module}")
    except (ImportError, AttributeError) as e:
        import traceback
        traceback.print_exc()
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": f"Failed to import GEPA-AI adapter: {e}",
        }
    
    # Run optimization
    try:
        start_time = time.time()
        
        # Build model string from provider and model
        model_str = f"{provider}/{model}" if provider else model
        
        results = await run_func(
            task_app_url=task_app_url,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_limit,
            reflection_minibatch_size=reflection_minibatch_size,
            output_dir=None,  # Use default output dir
            model=model_str,  # Pass model to adapter
        )
        
        elapsed_time = time.time() - start_time
        
        # Format time nicely for display
        if elapsed_time >= 60:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed_time:.1f}s"
        
        print()
        print("=" * 80)
        print(f"✅ [{benchmark_name}] GEPA-AI optimization COMPLETED in {time_str}")
        if results.get("baseline_score") is not None and results.get("best_score") is not None:
            baseline = results.get("baseline_score", 0.0)
            best = results.get("best_score", 0.0)
            improvement = ((best - baseline) / baseline * 100) if baseline > 0 else 0
            print(f"   Baseline: {baseline:.4f} → Best: {best:.4f} ({improvement:+.1f}% improvement)")
        print("=" * 80)
        print()
        
        return {
            "benchmark": benchmark_name,
            "status": "completed",
            "proposer_mode": proposer_mode,
            "model": f"{provider}/{model}",
            "baseline_score": results.get("baseline_score", 0.0),
            "best_score": results.get("best_score", 0.0),
            "val_score": results.get("val_score"),
            "total_rollouts": results.get("total_rollouts", 0),
            "actual_rollouts": results.get("actual_rollouts"),
            "total_time": elapsed_time,
            "prompt_file": results.get("prompt_file"),
            "results_file": results.get("results_file"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": str(e),
        }


async def run_library_optimization(
    benchmark_name: str,
    benchmark_config: Dict,
    yaml_config: Dict,
) -> Dict[str, Any]:
    """Run library optimization based on proposer_mode.
    
    Args:
        benchmark_name: Name of the benchmark
        benchmark_config: Benchmark config from YAML
        yaml_config: Full YAML config
        
    Returns:
        Results dictionary
    """
    import time
    start_time = time.time()
    
    proposer_mode = benchmark_config.get("proposer_mode", "synth")
    
    # Yield control immediately to allow all tasks to start in parallel
    await asyncio.sleep(0)
    
    print(f"⏱️  [{benchmark_name}] Starting {proposer_mode} optimization...")
    
    try:
        if proposer_mode == "dspy":
            result = await run_dspy_library_optimization(benchmark_name, benchmark_config, yaml_config)
        elif proposer_mode == "gepa-ai":
            result = await run_gepa_ai_library_optimization(benchmark_name, benchmark_config, yaml_config)
        else:
            result = {
                "benchmark": benchmark_name,
                "status": "skipped",
                "reason": f"proposer_mode '{proposer_mode}' not supported for library reference runs",
            }
        
        elapsed = time.time() - start_time
        # Format time nicely
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed:.1f}s"
        
        print()
        print("=" * 80)
        print(f"✅ [{benchmark_name}] COMPLETED in {time_str}")
        print("=" * 80)
        print()
        
        # Add timing to result if not already present
        if isinstance(result, dict):
            result["total_time"] = elapsed
        
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        # Format time nicely
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{elapsed:.1f}s"
        
        print()
        print("=" * 80)
        print(f"❌ [{benchmark_name}] FAILED after {time_str}: {e}")
        print("=" * 80)
        print()
        
        import traceback
        traceback.print_exc()
        return {
            "benchmark": benchmark_name,
            "status": "error",
            "error": str(e),
            "total_time": elapsed,
        }


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run DSPy-AI/GEPA-AI library optimizations using synth_gepa_config.yaml"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run only this benchmark (default: run all)",
    )
    args = parser.parse_args()
    
    # Load config
    yaml_config = load_yaml_config()
    benchmarks = yaml_config.get("benchmarks", {})
    
    if not benchmarks:
        print("❌ No benchmarks found in config")
        return
    
    # Filter benchmarks if specified
    if args.benchmark:
        if args.benchmark not in benchmarks:
            print(f"❌ Benchmark '{args.benchmark}' not found in config")
            print(f"   Available benchmarks: {list(benchmarks.keys())}")
            return
        benchmarks = {args.benchmark: benchmarks[args.benchmark]}
    
    # Filter to only benchmarks with proposer_mode="dspy" or "gepa-ai" and run=true
    library_benchmarks = {}
    for name, config in benchmarks.items():
        # Check if run field is explicitly False (skip if False, include if True or missing)
        run_flag = config.get("run", True)  # Default to True if not specified
        if not run_flag:
            continue
        
        proposer_mode = config.get("proposer_mode", "synth")
        if proposer_mode in ("dspy", "gepa-ai"):
            library_benchmarks[name] = config
    
    if not library_benchmarks:
        print("⚠️  No benchmarks with proposer_mode='dspy' or 'gepa-ai' found")
        print("   Available proposer_modes:", set(c.get("proposer_mode", "synth") for c in benchmarks.values()))
        return
    
    print("=" * 80)
    print("LIBRARY REFERENCE EXPERIMENTS (DSPy-AI/GEPA-AI)")
    print("=" * 80)
    print(f"Running {len(library_benchmarks)} benchmark(s) IN PARALLEL")
    print("  (Different models prevent rate limiting conflicts)")
    print()
    
    # Show which benchmarks will run
    for name, config in library_benchmarks.items():
        proposer_mode = config.get("proposer_mode", "synth")
        model_config = config.get("model", {})
        provider = model_config.get("provider", "N/A")
        model = model_config.get("model", "N/A")
        print(f"  • {name} ({proposer_mode}): {provider}/{model}")
    print()
    
    # Run optimizations in parallel (all at once)
    # Create tasks and build proper data structures for tracking
    tasks: Dict[asyncio.Task, str] = {}  # task -> benchmark_name mapping
    name_to_task: Dict[str, asyncio.Task] = {}  # benchmark_name -> task mapping
    
    for name, config in library_benchmarks.items():
        coro = run_library_optimization(name, config, yaml_config)
        task = asyncio.create_task(coro)
        tasks[task] = name  # Reverse mapping: task -> name
        name_to_task[name] = task  # Forward mapping: name -> task
    
    # Yield control to event loop to ensure all tasks start immediately
    await asyncio.sleep(0)
    
    print("🚀 Starting all optimizations in parallel...")
    print()
    
    # Track results as they complete
    completed_results: Dict[str, Dict[str, Any]] = {}
    pending_benchmarks = set(library_benchmarks.keys())
    last_update_time = time.time()
    UPDATE_INTERVAL = 180.0  # 3 minutes
    
    def print_results_table():
        """Print current results table."""
        print("\n" + "=" * 140)
        print("RESULTS SUMMARY (Updated)")
        print("=" * 140)
        print(f"{'Benchmark':<30} {'Status':<15} {'Model':<25} {'Baseline':<12} {'Best':<12} {'Lift':<12} {'Rollouts':<12} {'Time':<12}")
        print("-" * 140)
        
        # Show completed benchmarks
        for name in sorted(completed_results.keys()):
            result = completed_results[name]
            benchmark = result.get("benchmark", name)
            status = result.get("status", "unknown")
            model = result.get("model", "N/A")
            baseline = result.get("baseline_score")
            best = result.get("best_score")
            rollouts = result.get("total_rollouts") or result.get("actual_rollouts")
            elapsed = result.get("total_time")
            
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            best_str = f"{best:.4f}" if best is not None else "N/A"
            rollouts_str = str(rollouts) if rollouts is not None else "N/A"
            
            # Calculate lift (improvement) - consistent notation
            if baseline is not None and best is not None:
                if baseline > 0:
                    lift_pct = ((best - baseline) / baseline) * 100
                    # Use consistent notation: always show sign, no double plus
                    lift_str = f"{lift_pct:+.1f}%"
                elif best > baseline:
                    lift_str = f"+{best - baseline:.4f}"
                else:
                    lift_str = f"{best - baseline:.4f}"
            else:
                lift_str = "N/A"
            
            if elapsed is not None:
                if elapsed >= 60:
                    time_str = f"{elapsed / 60:.1f}m"
                else:
                    time_str = f"{elapsed:.1f}s"
            else:
                time_str = "N/A"
            
            if status == "error":
                error = result.get("error", "Unknown error")
                print(f"{benchmark:<30} {'ERROR':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}")
                print(f"  Error: {error[:80]}")
            elif status == "skipped":
                reason = result.get("reason", "Unknown reason")
                print(f"{benchmark:<30} {'SKIPPED':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}")
                print(f"  Reason: {reason[:80]}")
            else:
                print(f"{benchmark:<30} {status:<15} {model:<25} {baseline_str:<12} {best_str:<12} {lift_str:<12} {rollouts_str:<12} {time_str:<12}")
        
        # Show pending benchmarks
        if pending_benchmarks:
            print("-" * 140)
            print("⏳ Still Running:")
            for name in sorted(pending_benchmarks):
                config = library_benchmarks[name]
                proposer_mode = config.get("proposer_mode", "synth")
                model_config = config.get("model", {})
                provider = model_config.get("provider", "N/A")
                model = model_config.get("model", "N/A")
                print(f"  • {name} ({proposer_mode}): {provider}/{model}")
        
        print("=" * 140)
    
    # Process results as they complete
    async def process_completions():
        """Process completed benchmarks and show updates."""
        try:
            # Use asyncio.wait() with FIRST_COMPLETED to process tasks as they finish
            # This ensures all tasks run in parallel and we process results incrementally
            # Convert dict_keys to a set for asyncio.wait()
            remaining_tasks = set(tasks.keys())
            
            while remaining_tasks:
                # Wait for at least one task to complete
                done, pending = await asyncio.wait(
                    remaining_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process all completed tasks
                for done_task in done:
                    # done_task should be a Task object from asyncio.wait()
                    # Verify it's actually a Task
                    if not isinstance(done_task, asyncio.Task):
                        print(f"⚠️  WARNING: Expected Task object, got {type(done_task)}: {done_task}", flush=True)
                        continue
                    
                    # Get benchmark name from our mapping using identity check
                    benchmark_name = None
                    for task, name in tasks.items():
                        if task is done_task:
                            benchmark_name = name
                            break
                    
                    if benchmark_name is None:
                        # This shouldn't happen, but provide better error message
                        raise RuntimeError(
                            f"CRITICAL BUG: Completed task not found in tasks mapping. "
                            f"Task type: {type(done_task)}, Task id: {id(done_task)}, "
                            f"Available task ids: {[id(t) for t in tasks.keys()]}, "
                            f"Available names: {list(tasks.values())}"
                        )
                    
                    try:
                        # Get the result from the completed task
                        result = done_task.result()
                        
                        if isinstance(result, Exception):
                            completed_results[benchmark_name] = {
                                "benchmark": benchmark_name,
                                "status": "error",
                                "error": str(result),
                            }
                        else:
                            completed_results[benchmark_name] = result
                        
                        # Remove from pending
                        pending_benchmarks.discard(benchmark_name)
                        remaining_tasks.remove(done_task)
                        
                        # Show table when a benchmark completes
                        print_results_table()
                        
                        if len(pending_benchmarks) == 0:
                            print("\n✅ All benchmarks completed!")
                            # Cancel periodic_updates task if it's still running
                            # (it will check and exit on its own, but this ensures quick exit)
                            return
                        else:
                            print(f"\n⏳ Waiting for {len(pending_benchmarks)} remaining benchmark(s)...")
                            print(f"   Next update in ~{UPDATE_INTERVAL/60:.1f} minutes or when next completes\n")
                    
                    except Exception as e:
                        print(f"❌ ERROR processing result for {benchmark_name}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        completed_results[benchmark_name] = {
                            "benchmark": benchmark_name,
                            "status": "error",
                            "error": str(e),
                        }
                        pending_benchmarks.discard(benchmark_name)
                        remaining_tasks.discard(done_task)
                        print_results_table()
                        
                        if len(pending_benchmarks) == 0:
                            print("\n✅ All benchmarks completed (some with errors)!")
                            return
        except Exception as e:
            print(f"❌ CRITICAL ERROR in process_completions: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Fallback: check all tasks manually
            for name, task in name_to_task.items():
                if name not in completed_results:
                    if task.done():
                        try:
                            result = task.result()
                            completed_results[name] = result
                            pending_benchmarks.discard(name)
                        except Exception as task_error:
                            completed_results[name] = {
                                "benchmark": name,
                                "status": "error",
                                "error": str(task_error),
                            }
                            pending_benchmarks.discard(name)
    
    async def periodic_updates():
        """Show periodic updates every 3 minutes."""
        while len(pending_benchmarks) > 0:
            # Sleep in smaller chunks so we can exit quickly when all benchmarks complete
            sleep_interval = min(UPDATE_INTERVAL, 10.0)  # Check every 10 seconds max
            elapsed = 0.0
            while elapsed < UPDATE_INTERVAL and len(pending_benchmarks) > 0:
                await asyncio.sleep(sleep_interval)
                elapsed += sleep_interval
                # Exit early if all benchmarks completed
                if len(pending_benchmarks) == 0:
                    return
            
            # Check again after sleep - benchmarks may have completed
            if len(pending_benchmarks) > 0:
                print_results_table()
                print(f"\n⏳ Still waiting for {len(pending_benchmarks)} remaining benchmark(s)...")
                print(f"   Next update in ~{UPDATE_INTERVAL/60:.1f} minutes or when next completes\n")
            # If all benchmarks completed while we were sleeping, exit
            if len(pending_benchmarks) == 0:
                break
    
    # Run both tasks concurrently
    # process_completions will exit when all benchmarks are done
    # periodic_updates will also exit when pending_benchmarks is empty
    periodic_task = asyncio.create_task(periodic_updates())
    try:
        await process_completions()
    finally:
        # Cancel periodic_updates if it's still running (all benchmarks completed)
        if not periodic_task.done():
            periodic_task.cancel()
            try:
                await periodic_task
            except asyncio.CancelledError:
                pass
    
    # Safety check: ensure all tasks are processed
    for name, task in name_to_task.items():
        if name not in completed_results:
            if task.done():
                try:
                    result = task.result()
                    completed_results[name] = result
                    pending_benchmarks.discard(name)
                except Exception as e:
                    completed_results[name] = {
                        "benchmark": name,
                        "status": "error",
                        "error": str(e),
                    }
                    pending_benchmarks.discard(name)
            else:
                # Task still running - wait for it
                try:
                    result = await task
                    completed_results[name] = result
                    pending_benchmarks.discard(name)
                except Exception as e:
                    completed_results[name] = {
                        "benchmark": name,
                        "status": "error",
                        "error": str(e),
                    }
                    pending_benchmarks.discard(name)
    
    # Final results table
    results = [completed_results[name] for name in sorted(completed_results.keys())]
    
    # Print results table
    print("\n" + "=" * 140)
    print("RESULTS SUMMARY")
    print("=" * 140)
    print(f"{'Benchmark':<30} {'Status':<15} {'Model':<25} {'Baseline':<12} {'Best':<12} {'Lift':<12} {'Rollouts':<12} {'Time':<12}")
    print("-" * 140)
    
    # Track gains and final performance for summary table
    gepa_ai_gains: list[float] = []
    dspy_gains: list[float] = []
    gepa_ai_final_scores: list[float] = []
    dspy_final_scores: list[float] = []
    
    for result in results:
        benchmark = result.get("benchmark", "Unknown")
        status = result.get("status", "unknown")
        model = result.get("model", "N/A")
        baseline = result.get("baseline_score")
        best = result.get("best_score")
        rollouts = result.get("total_rollouts") or result.get("actual_rollouts")
        elapsed = result.get("total_time")
        
        baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
        best_str = f"{best:.4f}" if best is not None else "N/A"
        rollouts_str = str(rollouts) if rollouts is not None else "N/A"
        
        # Calculate lift (delta)
        lift_delta = None
        if baseline is not None and best is not None:
            lift_delta = best - baseline
            if baseline > 0:
                lift_pct = ((best - baseline) / baseline) * 100
                # Use consistent notation: always show sign, no double plus
                lift_str = f"{lift_pct:+.1f}%"
            elif best > baseline:
                lift_str = f"+{lift_delta:.4f}"
            else:
                lift_str = f"{lift_delta:.4f}"
        else:
            lift_str = "N/A"
        
        # Track gains and final performance for summary (only for successful runs)
        if status == "completed" and lift_delta is not None:
            if "gepa_ai" in benchmark.lower():
                gepa_ai_gains.append(lift_delta)
                gepa_ai_final_scores.append(best)
            elif "dspy" in benchmark.lower():
                dspy_gains.append(lift_delta)
                dspy_final_scores.append(best)
        
        if elapsed is not None:
            if elapsed >= 60:
                time_str = f"{elapsed / 60:.1f}m"
            else:
                time_str = f"{elapsed:.1f}s"
        else:
            time_str = "N/A"
        
        if status == "error":
            error = result.get("error", "Unknown error")
            print(f"{benchmark:<30} {'ERROR':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}")
            print(f"  Error: {error}")
        elif status == "skipped":
            reason = result.get("reason", "Unknown reason")
            print(f"{benchmark:<30} {'SKIPPED':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}")
            print(f"  Reason: {reason}")
        else:
            print(f"{benchmark:<30} {status:<15} {model:<25} {baseline_str:<12} {best_str:<12} {lift_str:<12} {rollouts_str:<12} {time_str:<12}")
    
    print("=" * 140)
    
    # Print summary table with average gains and final performance
    if gepa_ai_gains or dspy_gains:
        print("\n" + "=" * 100)
        print("AVERAGE GAINS SUMMARY")
        print("=" * 100)
        print(f"{'Optimizer':<20} {'Avg Gain (Δ)':<18} {'Avg Final Score':<18} {'Count':<10}")
        print("-" * 100)
        
        if gepa_ai_gains:
            avg_gain = sum(gepa_ai_gains) / len(gepa_ai_gains)
            avg_final = sum(gepa_ai_final_scores) / len(gepa_ai_final_scores)
            print(f"{'GEPA-AI':<20} {avg_gain:+.4f}{'':<10} {avg_final:.4f}{'':<10} {len(gepa_ai_gains):<10}")
        
        if dspy_gains:
            avg_gain = sum(dspy_gains) / len(dspy_gains)
            avg_final = sum(dspy_final_scores) / len(dspy_final_scores)
            print(f"{'DSPy':<20} {avg_gain:+.4f}{'':<10} {avg_final:.4f}{'':<10} {len(dspy_gains):<10}")
        
        print("=" * 100)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = COMPARISONS_DIR / f"library_reference_results_{timestamp}.json"
    readout_file = COMPARISONS_DIR / f"library_reference_readout_{timestamp}.txt"
    
    output_data = {
        "timestamp": timestamp,
        "config_file": str(CONFIG_FILE),
        "results": results,
    }
    
    try:
        # Save JSON
        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {results_file}")
        
        # ✅ ADD: Save comprehensive text readout (like synth versions)
        with open(readout_file, "w") as f:
            f.write("=" * 120 + "\n")
            f.write("LIBRARY REFERENCE EXPERIMENTS RESULTS (DSPy-AI/GEPA-AI)\n")
            f.write("=" * 120 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Config File: {CONFIG_FILE}\n")
            f.write("=" * 120 + "\n\n")
            
            # Summary table
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 140 + "\n")
            f.write(f"{'Benchmark':<30} {'Status':<15} {'Model':<25} {'Baseline':<12} {'Best':<12} {'Lift':<12} {'Rollouts':<12} {'Time':<12}\n")
            f.write("-" * 140 + "\n")
            
            for result in results:
                benchmark = result.get("benchmark", "Unknown")
                status = result.get("status", "unknown")
                model = result.get("model", "N/A")
                baseline = result.get("baseline_score")
                best = result.get("best_score")
                rollouts = result.get("total_rollouts") or result.get("actual_rollouts")
                elapsed = result.get("total_time")
                
                baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
                best_str = f"{best:.4f}" if best is not None else "N/A"
                rollouts_str = str(rollouts) if rollouts is not None else "N/A"
                
                # Calculate lift (delta) for file output - consistent notation
                lift_delta = None
                if baseline is not None and best is not None:
                    lift_delta = best - baseline
                    if baseline > 0:
                        lift_pct = ((best - baseline) / baseline) * 100
                        # Use consistent notation: always show sign, no double plus
                        lift_str = f"{lift_pct:+.1f}%"
                    elif best > baseline:
                        lift_str = f"+{lift_delta:.4f}"
                    else:
                        lift_str = f"{lift_delta:.4f}"
                else:
                    lift_str = "N/A"
                
                if elapsed is not None:
                    if elapsed >= 60:
                        time_str = f"{elapsed / 60:.1f}m"
                    else:
                        time_str = f"{elapsed:.1f}s"
                else:
                    time_str = "N/A"
                
                if status == "error":
                    error = result.get("error", "Unknown error")
                    f.write(f"{benchmark:<30} {'ERROR':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}\n")
                    f.write(f"  Error: {error}\n")
                elif status == "skipped":
                    reason = result.get("reason", "Unknown reason")
                    f.write(f"{benchmark:<30} {'SKIPPED':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12} {'':<12}\n")
                    f.write(f"  Reason: {reason}\n")
                else:
                    f.write(f"{benchmark:<30} {status:<15} {model:<25} {baseline_str:<12} {best_str:<12} {lift_str:<12} {rollouts_str:<12} {time_str:<12}\n")
            
            f.write("=" * 140 + "\n")
            
            # Add summary table to file
            file_gepa_ai_gains: list[float] = []
            file_dspy_gains: list[float] = []
            file_gepa_ai_final_scores: list[float] = []
            file_dspy_final_scores: list[float] = []
            
            for result in results:
                status = result.get("status", "unknown")
                baseline = result.get("baseline_score")
                best = result.get("best_score")
                benchmark = result.get("benchmark", "Unknown")
                
                if status == "completed" and baseline is not None and best is not None:
                    lift_delta = best - baseline
                    if "gepa_ai" in benchmark.lower():
                        file_gepa_ai_gains.append(lift_delta)
                        file_gepa_ai_final_scores.append(best)
                    elif "dspy" in benchmark.lower():
                        file_dspy_gains.append(lift_delta)
                        file_dspy_final_scores.append(best)
            
            if file_gepa_ai_gains or file_dspy_gains:
                f.write("\n" + "=" * 100 + "\n")
                f.write("AVERAGE GAINS SUMMARY\n")
                f.write("=" * 100 + "\n")
                f.write(f"{'Optimizer':<20} {'Avg Gain (Δ)':<18} {'Avg Final Score':<18} {'Count':<10}\n")
                f.write("-" * 100 + "\n")
                
                if file_gepa_ai_gains:
                    avg_gain = sum(file_gepa_ai_gains) / len(file_gepa_ai_gains)
                    avg_final = sum(file_gepa_ai_final_scores) / len(file_gepa_ai_final_scores)
                    f.write(f"{'GEPA-AI':<20} {avg_gain:+.4f}{'':<10} {avg_final:.4f}{'':<10} {len(file_gepa_ai_gains):<10}\n")
                
                if file_dspy_gains:
                    avg_gain = sum(file_dspy_gains) / len(file_dspy_gains)
                    avg_final = sum(file_dspy_final_scores) / len(file_dspy_final_scores)
                    f.write(f"{'DSPy':<20} {avg_gain:+.4f}{'':<10} {avg_final:.4f}{'':<10} {len(file_dspy_gains):<10}\n")
                
                f.write("=" * 100 + "\n\n")
            
            # Detailed results for each benchmark
            for result in results:
                benchmark = result.get("benchmark", "Unknown")
                status = result.get("status", "unknown")
                
                if status in ("error", "skipped"):
                    continue
                
                f.write("=" * 120 + "\n")
                f.write(f"📊 DETAILED RESULTS: {benchmark}\n")
                f.write("=" * 120 + "\n\n")
                
                # Basic info
                baseline = result.get("baseline_score")
                best = result.get("best_score")
                rollouts = result.get("total_rollouts") or result.get("actual_rollouts")
                elapsed = result.get("total_time")
                
                if baseline is not None:
                    f.write(f"Baseline Score: {baseline:.4f} ({baseline*100:.1f}%)\n")
                if best is not None:
                    f.write(f"Best Score:     {best:.4f} ({best*100:.1f}%)\n")
                if baseline is not None and best is not None:
                    improvement = ((best - baseline) / baseline) * 100 if baseline > 0 else 0
                    f.write(f"Improvement:    {improvement:+.1f}% relative ({(best - baseline)*100:+.1f} pp absolute)\n")
                if rollouts is not None:
                    f.write(f"Rollouts:       {rollouts}\n")
                if elapsed is not None:
                    f.write(f"Total Time:     {elapsed:.1f}s ({elapsed/60:.1f}m)\n")
                
                # File references
                prompt_file = result.get("prompt_file")
                results_file_bench = result.get("results_file")
                readout_file_bench = result.get("readout_file")
                log_file_bench = result.get("log_file")
                
                f.write("\n📁 Output Files:\n")
                if prompt_file:
                    f.write(f"  Optimized Prompt: {prompt_file}\n")
                if results_file_bench:
                    f.write(f"  Detailed Results (JSON): {results_file_bench}\n")
                if readout_file_bench:
                    f.write(f"  Comprehensive Readout: {readout_file_bench}\n")
                if log_file_bench:
                    f.write(f"  Verbose Log: {log_file_bench}\n")
                
                f.write("\n")
            
            f.write("=" * 120 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 120 + "\n")
        
        print(f"📄 Saved comprehensive readout to: {readout_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save results files: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

