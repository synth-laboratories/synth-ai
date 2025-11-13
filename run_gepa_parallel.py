#!/usr/bin/env python3
"""Run GEPA in parallel for Heart Disease, HotPotQA, and Banking77 tasks.

This script:
- Sets rollout limit to 200 and time limit to 3 minutes
- Runs all 3 tasks in parallel
- Shows only aggregate stats table (masks everything else)
- Shows only candidate 1 lift (not candidate 2)
"""

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Paths to config files
REPO_ROOT = Path(__file__).parent
CONFIGS = {
    "Heart Disease": REPO_ROOT / "examples/blog_posts/langprobe/task_specific/heartdisease/heartdisease_gepa.toml",
    "HotPotQA": REPO_ROOT / "examples/blog_posts/langprobe/task_specific/hotpotqa/hotpotqa_gepa.toml",
    "Banking77": REPO_ROOT / "examples/blog_posts/langprobe/task_specific/banking77/banking77_gepa.toml",
}


def modify_config_for_limits(config_path: Path, rollout_limit: int = 50, time_limit_seconds: int = 30) -> Path:
    """Create a temporary modified config with rollout and time limits."""
    # Read original config as text
    with open(config_path, "r") as f:
        config_text = f.read()
    
    # Resolve env_file_path relative to original config directory
    config_dir = config_path.parent.resolve()
    repo_root = config_path.parent
    # Go up to find repo root (where .env is)
    for _ in range(6):  # Max depth
        if (repo_root / ".env").exists():
            break
        repo_root = repo_root.parent
    else:
        # Fallback: use synth-ai repo root
        repo_root = Path(__file__).parent
    
    env_file_abs = (repo_root / ".env").resolve()
    
    # Resolve results_folder to absolute path (relative to original config directory)
    results_folder_abs = None
    results_folder_updated = False
    
    # Add/modify termination_config section
    lines = config_text.split("\n")
    new_lines = []
    in_termination_config = False
    termination_config_added = False
    in_prompt_learning = False
    env_file_path_updated = False
    
    for i, line in enumerate(lines):
        # Update env_file_path to absolute path
        if "[prompt_learning]" in line:
            in_prompt_learning = True
            new_lines.append(line)
            continue
        
        if in_prompt_learning and "env_file_path" in line and "=" in line and not env_file_path_updated:
            # Replace relative path with absolute path
            new_lines.append(f'env_file_path = "{env_file_abs}"')
            env_file_path_updated = True
            continue
        
        if in_prompt_learning and "results_folder" in line and "=" in line and not results_folder_updated:
            # Extract current results_folder value
            match = re.search(r'results_folder\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                results_folder_rel = match.group(1).strip()
                # Resolve relative to original config directory
                if not Path(results_folder_rel).is_absolute():
                    results_folder_abs = (config_dir / results_folder_rel).resolve()
                else:
                    results_folder_abs = Path(results_folder_rel).expanduser().resolve()
                # Replace with absolute path
                new_lines.append(f'results_folder = "{results_folder_abs}"')
                results_folder_updated = True
                continue
        
        if in_prompt_learning and line.strip().startswith("[") and "[prompt_learning" not in line:
            in_prompt_learning = False
        
        if "[prompt_learning.termination_config]" in line:
            in_termination_config = True
            new_lines.append(line)
            continue
        
        if in_termination_config:
            if line.strip().startswith("[") and not line.strip().startswith("[prompt_learning"):
                # End of termination_config section
                # Add our limits if not already present
                if "max_rollouts" not in "\n".join(new_lines[-10:]):
                    new_lines.append(f"max_rollouts = {rollout_limit}")
                if "max_seconds" not in "\n".join(new_lines[-10:]):
                    new_lines.append(f"max_seconds = {time_limit_seconds}")
                in_termination_config = False
                termination_config_added = True
            elif "max_rollouts" in line or "max_seconds" in line:
                # Replace existing values
                if "max_rollouts" in line:
                    new_lines.append(f"max_rollouts = {rollout_limit}")
                elif "max_seconds" in line:
                    new_lines.append(f"max_seconds = {time_limit_seconds}")
                continue
        
        new_lines.append(line)
    
    # Ensure env_file_path is set if not found
    if not env_file_path_updated:
        # Find [prompt_learning] section and add env_file_path after it
        for i, line in enumerate(new_lines):
            if "[prompt_learning]" in line:
                # Insert after the section header
                new_lines.insert(i + 1, f'env_file_path = "{env_file_abs}"')
                break
    
    # Ensure results_folder is set to absolute path if not found or not updated
    if not results_folder_updated:
        # Try to find existing results_folder in original config
        match = re.search(r'results_folder\s*=\s*["\']?([^"\']+)["\']?', config_text)
        if match:
            results_folder_rel = match.group(1).strip()
            if not Path(results_folder_rel).is_absolute():
                results_folder_abs = (config_dir / results_folder_rel).resolve()
            else:
                results_folder_abs = Path(results_folder_rel).expanduser().resolve()
        else:
            # Default: results folder relative to original config
            results_folder_abs = (config_dir / "results").resolve()
        
        # Find [prompt_learning] section and add/update results_folder
        for i, line in enumerate(new_lines):
            if "[prompt_learning]" in line:
                # Check if results_folder already exists in next few lines
                found = False
                for j in range(i + 1, min(i + 10, len(new_lines))):
                    if "results_folder" in new_lines[j] and "=" in new_lines[j]:
                        # Update existing line
                        new_lines[j] = f'results_folder = "{results_folder_abs}"'
                        found = True
                        break
                if not found:
                    # Insert after env_file_path or section header
                    insert_idx = i + 1
                    if env_file_path_updated:
                        # Find env_file_path line
                        for j in range(i + 1, min(i + 10, len(new_lines))):
                            if "env_file_path" in new_lines[j]:
                                insert_idx = j + 1
                                break
                    new_lines.insert(insert_idx, f'results_folder = "{results_folder_abs}"')
                break
    
    # If termination_config section doesn't exist, add it
    if not termination_config_added:
        # Find where to insert it (after gepa section or at end)
        insert_idx = len(new_lines)
        for i, line in enumerate(new_lines):
            if "[prompt_learning.gepa.token]" in line or "[display]" in line:
                insert_idx = i
                break
        
        # Insert termination_config section
        new_lines.insert(insert_idx, "")
        new_lines.insert(insert_idx + 1, "[prompt_learning.termination_config]")
        new_lines.insert(insert_idx + 2, f"max_rollouts = {rollout_limit}")
        new_lines.insert(insert_idx + 3, f"max_seconds = {time_limit_seconds}")
        # Keep existing max_cost_usd and max_trials if present
        if "max_cost_usd" not in config_text:
            new_lines.insert(insert_idx + 4, "max_cost_usd = 3.0")
        if "max_trials" not in config_text:
            new_lines.insert(insert_idx + 5, "max_trials = 1000")
    
    # Also update rollout budget
    for i, line in enumerate(new_lines):
        if "[prompt_learning.gepa.rollout]" in line:
            # Find budget line and update it
            for j in range(i + 1, min(i + 5, len(new_lines))):
                if "budget" in new_lines[j] and "=" in new_lines[j]:
                    new_lines[j] = f"budget = {rollout_limit}"
                    break
            break
    
    # Ensure display section exists and suppresses output
    display_section_exists = False
    for i, line in enumerate(new_lines):
        if "[display]" in line:
            display_section_exists = True
            # Update display settings to suppress output
            for j in range(i + 1, min(i + 10, len(new_lines))):
                next_line = new_lines[j]
                if next_line.strip().startswith("[") and "[display" not in next_line:
                    break
                if "tui" in next_line and "=" in next_line:
                    new_lines[j] = "tui = false"
                elif "show_curve" in next_line and "=" in next_line:
                    new_lines[j] = "show_curve = false"
                elif "verbose_summary" in next_line and "=" in next_line:
                    new_lines[j] = "verbose_summary = false"
            break
    
    # Add display section if it doesn't exist
    if not display_section_exists:
        # Insert before END or at end
        insert_idx = len(new_lines)
        for i, line in enumerate(new_lines):
            if line.strip().startswith("[") and "[prompt_learning" not in line:
                insert_idx = i
                break
        new_lines.insert(insert_idx, "")
        new_lines.insert(insert_idx + 1, "[display]")
        new_lines.insert(insert_idx + 2, "tui = false")
        new_lines.insert(insert_idx + 3, "show_curve = false")
        new_lines.insert(insert_idx + 4, "verbose_summary = false")
        new_lines.insert(insert_idx + 5, "local_backend = true")
    
    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write("\n".join(new_lines))
    temp_file.close()
    
    # Debug: verify limits were set
    config_content = "\n".join(new_lines)
    if f"max_rollouts = {rollout_limit}" not in config_content:
        print(f"⚠️  WARNING: max_rollouts not found in modified config!")
    if f"max_seconds = {time_limit_seconds}" not in config_content:
        print(f"⚠️  WARNING: max_seconds not found in modified config!")
    
    return temp_path


def extract_results_from_file(task_name: str, job_id: str, config_path: Optional[Path] = None) -> Dict:
    """Extract results from results file by job_id. Never reads old files."""
    print(f"[{task_name}] Extracting results from file for job_id: {job_id}")
    
    if not job_id:
        print(f"[{task_name}] ERROR: No job_id provided")
        return {"error": "No job_id provided"}
    
    if not config_path:
        # Try to find config by task name
        config_path = CONFIGS.get(task_name)
    
    if not config_path or not config_path.exists():
        print(f"[{task_name}] ERROR: Config not found: {config_path}")
        return {"error": "Config not found"}
    
    # Extract policy model from config
    policy_model = None
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        
        if tomllib:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            pl_config = config.get("prompt_learning", {})
            policy_config = pl_config.get("policy", {})
            policy_model = policy_config.get("model")
            if policy_model:
                print(f"[{task_name}] Found policy_model: {policy_model}")
        else:
            # Fallback: try to parse manually
            with open(config_path, "r") as f:
                content = f.read()
            match = re.search(r'\[prompt_learning\.policy\].*?model\s*=\s*["\']?([^"\'\n]+)', content, re.DOTALL)
            if match:
                policy_model = match.group(1).strip()
                print(f"[{task_name}] Found policy_model (manual parse): {policy_model}")
    except Exception as e:
        print(f"[{task_name}] WARNING: Could not extract policy model: {e}")
    
    results_folder = config_path.parent / "results"
    if not results_folder.exists():
        print(f"[{task_name}] ERROR: Results folder not found: {results_folder}")
        return {"error": "Results folder not found"}
    
    # Find results file by job_id (format: gepa_results_pl_xxxxx_timestamp.txt)
    import re
    result_file = None
    all_files = list(results_folder.glob("gepa_results_*.txt"))
    print(f"[{task_name}] Searching {len(all_files)} results files for job_id {job_id}")
    
    for file in all_files:
        # Extract job_id from filename
        match = re.search(r'pl_[a-f0-9]+', file.name)
        if match:
            file_job_id = match.group(0)
            print(f"[{task_name}] Checking file {file.name} with job_id {file_job_id}")
            if file_job_id == job_id:
                result_file = file
                print(f"[{task_name}] Found matching file: {file.name}")
                break
    
    if not result_file:
        print(f"[{task_name}] ERROR: No results file found for job_id {job_id}")
        print(f"[{task_name}] Available files: {[f.name for f in all_files[:5]]}")
        return {"error": f"No results file found for job_id {job_id}"}
    
    # Parse results file
    try:
        print(f"[{task_name}] Reading results file: {result_file}")
        with open(result_file, "r") as f:
            content = f.read()
        
        print(f"[{task_name}] File size: {len(content)} chars")
        
        # Extract baseline and candidate scores
        baseline_score = None
        candidate1_score = None
        candidate1_lift = None
        total_cost = None
        total_rollouts = None
        total_time = None
        total_tokens = None
        eval_seeds_n = None
        
        # Look for baseline score (format: "📊 Baseline Score: 0.2600 (26.0%)")
        import re
        baseline_match = re.search(r'Baseline Score:\s*([\d.]+)', content)
        if baseline_match:
            baseline_score = float(baseline_match.group(1))
            print(f"[{task_name}] Found baseline_score: {baseline_score}")
        else:
            print(f"[{task_name}] WARNING: No baseline score found")
        
        # Extract cost (format: "Total Cost: $0.1234")
        cost_match = re.search(r'Total Cost:\s*\$?([\d.]+)', content)
        if cost_match:
            total_cost = float(cost_match.group(1))
            print(f"[{task_name}] Found total_cost: ${total_cost}")
        
        # Extract rollouts (format: "Rollouts: 50" or "Total Rollouts: 50")
        rollouts_match = re.search(r'(?:Total\s+)?Rollouts:\s*(\d+)', content)
        if rollouts_match:
            total_rollouts = int(rollouts_match.group(1))
            print(f"[{task_name}] Found total_rollouts: {total_rollouts}")
        
        # Extract time (format: "Time: 45.2s" or "Total Time: 45.2s (0.8 min)")
        time_match = re.search(r'(?:Total\s+)?Time:\s*([\d.]+)s', content)
        if time_match:
            total_time = float(time_match.group(1))
            print(f"[{task_name}] Found total_time: {total_time}s")
        
        # Extract tokens (format: "Tokens: 1.2345M" or "Total Tokens: 1234567")
        tokens_match = re.search(r'(?:Total\s+)?Tokens:\s*([\d.]+)([KMkm]?)', content)
        if tokens_match:
            tokens_val = float(tokens_match.group(1))
            unit = tokens_match.group(2).upper() if tokens_match.group(2) else ""
            if unit == "K":
                total_tokens = int(tokens_val * 1000)
            elif unit == "M":
                total_tokens = int(tokens_val * 1000000)
            else:
                total_tokens = int(tokens_val)
            print(f"[{task_name}] Found total_tokens: {total_tokens}")
        
        # Extract eval seeds N (look for "Heldout Evaluation" section with N= or seeds count)
        eval_n_match = re.search(r'(?:Heldout Evaluation|Validation).*?N\s*=\s*(\d+)', content, re.DOTALL | re.IGNORECASE)
        if eval_n_match:
            eval_seeds_n = int(eval_n_match.group(1))
            print(f"[{task_name}] Found eval_seeds_n: {eval_seeds_n}")
        else:
            # Try to find in baseline section
            baseline_n_match = re.search(r'Baseline.*?N\s*=\s*(\d+)', content, re.DOTALL | re.IGNORECASE)
            if baseline_n_match:
                eval_seeds_n = int(baseline_n_match.group(1))
                print(f"[{task_name}] Found eval_seeds_n from baseline: {eval_seeds_n}")
        
        # Look for validation section with Candidate 1 in FINAL SUMMARY
        # Format: "Candidate 1 Accuracy: 0.5712 (Δ-0.0334 vs baseline)" or "Candidate 1 Accuracy: 0.5712 (Δ-0.0334, -5.7% vs baseline)"
        validation_section = re.search(r'FINAL SUMMARY.*?Candidate 1.*?Accuracy:\s*([\d.]+).*?\(Δ([+-]?[\d.]+)', content, re.DOTALL)
        if validation_section:
            candidate1_score = float(validation_section.group(1))
            candidate1_lift = float(validation_section.group(2))
            print(f"[{task_name}] Found candidate1_score from FINAL SUMMARY: {candidate1_score}, lift: {candidate1_lift}")
        else:
            # Also try Heldout Evaluation section format
            heldout_section = re.search(r'Heldout Evaluation.*?Candidate 1.*?Accuracy:\s*([\d.]+).*?\(Δ([+-]?[\d.]+)', content, re.DOTALL)
            if heldout_section:
                candidate1_score = float(heldout_section.group(1))
                candidate1_lift = float(heldout_section.group(2))
                print(f"[{task_name}] Found candidate1_score from Heldout Evaluation: {candidate1_score}, lift: {candidate1_lift}")
            else:
                # Fallback: look for best score if no validation section
                best_match = re.search(r'Best Score:\s*([\d.]+)', content)
                if best_match:
                    candidate1_score = float(best_match.group(1))
                    if baseline_score is not None:
                        candidate1_lift = candidate1_score - baseline_score
                    print(f"[{task_name}] Found candidate1_score from Best Score: {candidate1_score}, lift: {candidate1_lift}")
                else:
                    print(f"[{task_name}] WARNING: No candidate score found")
        
        result = {
            "baseline_score": baseline_score,
            "candidate1_score": candidate1_score,
            "candidate1_lift": candidate1_lift,
            "total_cost": total_cost,
            "total_rollouts": total_rollouts,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "eval_seeds_n": eval_seeds_n,
            "policy_model": policy_model,
        }
        print(f"[{task_name}] Extracted results: {result}")
        return result
    except Exception as e:
        print(f"[{task_name}] ERROR parsing results file: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to parse results file: {e}"}


async def run_gepa_job(task_name: str, config_path: Path) -> Dict:
    """Run a single GEPA job and return results."""
    import time
    
    print(f"[{task_name}] Starting job...")
    
    # Record start time to find new results files
    start_time = time.time()
    
    # Create modified config with limits (small for quick testing)
    temp_config = modify_config_for_limits(config_path, rollout_limit=50, time_limit_seconds=30)
    assert temp_config.exists(), f"Temp config file not created: {temp_config}"
    print(f"[{task_name}] Created temp config: {temp_config}")
    
    # Get timestamp before running to identify new results files
    results_folder = config_path.parent / "results"
    existing_files = set()
    if results_folder.exists():
        existing_files = {f.name for f in results_folder.glob("gepa_results_*.txt")}
        print(f"[{task_name}] Found {len(existing_files)} existing results files")
    
    try:
        # Run the job (suppress all output by setting display options in TOML)
        cmd = [
            sys.executable, "-m", "synth_ai", "train",
            "--config", str(temp_config),
            "--local-backend",
        ]
        
        print(f"[{task_name}] Running command: {' '.join(cmd[:3])} ...")
        
        # Run and capture output (capture stdout to get job_id, capture stderr for debugging)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,  # Capture stderr for debugging
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"[{task_name}] Job failed with return code {process.returncode}")
            print(f"[{task_name}] Stderr: {error_msg[:500]}")
            return {
                "task": task_name,
                "status": "failed",
                "error": f"Job failed: {error_msg[:200]}",
            }
        
        print(f"[{task_name}] Job completed successfully")
        
        # Extract job_id from stdout (look for JSON response with job_id)
        job_id = None
        if stdout:
            import re
            output = stdout.decode()
            print(f"[{task_name}] stdout length: {len(output)} chars")
            print(f"[{task_name}] stdout preview: {output[:200]}")
            
            # Look for job_id in response JSON
            match = re.search(r'"job_id"\s*:\s*"([^"]+)"', output)
            if match:
                job_id = match.group(1)
                print(f"[{task_name}] Found job_id from JSON: {job_id}")
            else:
                # Try pattern pl_xxxxx
                match = re.search(r'pl_[a-f0-9]+', output)
                if match:
                    job_id = match.group(0)
                    print(f"[{task_name}] Found job_id from pattern: {job_id}")
                else:
                    print(f"[{task_name}] No job_id found in stdout")
        
        if not job_id:
            return {
                "task": task_name,
                "status": "failed",
                "error": "Could not extract job_id from stdout",
            }
        
        # CLI command waits for job completion, so results file should exist now
        # But give it a moment to ensure file I/O is complete
        import time
        time.sleep(0.5)  # Small delay to ensure file is written
        
        # Check if results file exists (CLI saves it at the end)
        result_file_found = False
        if results_folder.exists() and job_id:
            import re
            # Retry a few times in case file is still being written
            for attempt in range(3):
                all_results_files = list(results_folder.glob("gepa_results_*.txt"))
                print(f"[{task_name}] Attempt {attempt+1}: Checking {len(all_results_files)} results files in {results_folder}")
                print(f"[{task_name}] Looking for job_id: {job_id}")
                
                for file in all_results_files:
                    # Extract job_id from filename (format: gepa_results_pl_xxxxx_timestamp.txt)
                    match = re.search(r'pl_[a-f0-9]+', file.name)
                    if match:
                        file_job_id = match.group(0)
                        if file_job_id == job_id:
                            result_file_found = True
                            print(f"[{task_name}] ✅ Found matching results file: {file.name}")
                            break
                
                if result_file_found:
                    break
                
                if attempt < 2:
                    print(f"[{task_name}] File not found yet, waiting 0.5s...")
                    time.sleep(0.5)
            
            if not result_file_found:
                print(f"[{task_name}] ❌ No matching results file found after retries")
                all_results_files = list(results_folder.glob("gepa_results_*.txt"))
                print(f"[{task_name}] All files: {[f.name for f in all_results_files[:10]]}")
        elif not results_folder.exists():
            print(f"[{task_name}] ⚠️  Results folder doesn't exist: {results_folder}")
        elif not job_id:
            print(f"[{task_name}] ⚠️  No job_id to search for")
        
        if not result_file_found and job_id:
            print(f"[{task_name}] Results file not found (will extract from events)")
        
        print(f"[{task_name}] Job completed with job_id: {job_id}")
        return {
            "task": task_name,
            "status": "completed",
            "job_id": job_id,
            "config_path": config_path,
        }
    finally:
        # Clean up temp config
        try:
            temp_config.unlink()
            print(f"[{task_name}] Cleaned up temp config")
        except Exception as e:
            print(f"[{task_name}] Failed to clean up temp config: {e}")


def extract_results_from_events(job_id: str, backend_base: str = "http://localhost:8000/api", api_key: Optional[str] = None) -> Dict:
    """Extract results from job events."""
    import os
    import requests
    import time
    
    print(f"[extract_events] Extracting results from events for job_id: {job_id}")
    
    if not api_key:
        api_key = os.getenv("SYNTH_API_KEY") or os.getenv("ENVIRONMENT_API_KEY")
    
    if not api_key:
        print(f"[extract_events] ERROR: No API key found")
        return {"error": "No API key found"}
    
    # Wait for job to complete (poll status)
    url_status = f"{backend_base}/prompt-learning/online/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"[extract_events] Polling job status at {url_status}")
    max_wait = 600  # 10 minutes max wait
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(url_status, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                job_data = resp.json()
                status = job_data.get("status", "")
                print(f"[extract_events] Job status: {status}")
                if status in ("succeeded", "failed"):
                    print(f"[extract_events] Job completed with status: {status}")
                    break
            else:
                print(f"[extract_events] Status check failed with code {resp.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"[extract_events] Exception polling status: {e}")
            time.sleep(2)
    
    # Fetch events
    url_events = f"{backend_base}/prompt-learning/online/jobs/{job_id}/events?limit=1000"
    
    print(f"[extract_events] Fetching events from {url_events}")
    try:
        resp = requests.get(url_events, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            print(f"[extract_events] Failed to fetch events: status {resp.status_code}")
            return {"error": f"Failed to fetch events: status {resp.status_code}"}
        
        events = resp.json()
        if isinstance(events, dict):
            events = events.get("events", [])
        
        print(f"[extract_events] Fetched {len(events)} events")
        
        # Extract key metrics
        baseline_score = None
        candidate1_score = None
        candidate1_lift = None
        total_cost = None
        total_rollouts = None
        total_time = None
        
        # Log event types found
        event_types = [e.get("type", "unknown") for e in events if isinstance(e, dict)]
        print(f"[extract_events] Event types found: {set(event_types)}")
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            
            if event_type == "prompt.learning.validation.summary":
                print(f"[extract_events] Found validation.summary event")
                validation = event_data
                baseline = validation.get("baseline", {})
                results = validation.get("results", [])
                
                if baseline:
                    baseline_score = baseline.get("accuracy")
                    print(f"[extract_events] Baseline score: {baseline_score}")
                
                if results and len(results) > 0:
                    # Only get candidate 1
                    candidate1 = results[0]
                    candidate1_score = candidate1.get("accuracy")
                    print(f"[extract_events] Candidate 1 score: {candidate1_score}")
                    if baseline_score is not None and candidate1_score is not None:
                        candidate1_lift = candidate1_score - baseline_score
                        print(f"[extract_events] Candidate 1 lift: {candidate1_lift}")
                else:
                    print(f"[extract_events] No results found in validation summary")
            
            elif event_type == "prompt.learning.completed":
                print(f"[extract_events] Found completed event")
                total_cost = event_data.get("total_cost_usd")
                total_rollouts = event_data.get("total_rollouts")
                print(f"[extract_events] Total cost: {total_cost}, rollouts: {total_rollouts}")
                # Calculate time from created_at and finished_at
                created_at = event.get("created_at")
                finished_at = event.get("finished_at")
                if created_at and finished_at:
                    from datetime import datetime
                    try:
                        start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                        total_time = (end - start).total_seconds()
                        print(f"[extract_events] Total time: {total_time}s")
                    except Exception as e:
                        print(f"[extract_events] Failed to parse time: {e}")
        
        result = {
            "baseline_score": baseline_score,
            "candidate1_score": candidate1_score,
            "candidate1_lift": candidate1_lift,
            "total_cost": total_cost,
            "total_rollouts": total_rollouts,
            "total_time": total_time,
        }
        print(f"[extract_events] Final extracted result: {result}")
        return result
    except Exception as e:
        print(f"[extract_events] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def main():
    """Run all GEPA jobs in parallel and show aggregate stats."""
    print("=" * 80)
    print("Starting parallel GEPA jobs for 3 tasks...")
    print("=" * 80)
    
    # Run all jobs in parallel
    tasks = []
    for task_name, config_path in CONFIGS.items():
        if not config_path.exists():
            print(f"⚠️  Config not found: {config_path}")
            continue
        tasks.append(run_gepa_job(task_name, config_path))
    
    if not tasks:
        print("No valid configs found!")
        return
    
    print(f"\n🚀 Running {len(tasks)} jobs in parallel...\n")
    results = await asyncio.gather(*tasks)
    print(f"\n✅ All {len(tasks)} jobs completed\n")
    
    # Extract detailed results from events or files
    print("\n" + "=" * 80)
    print("Extracting results from completed jobs...")
    print("=" * 80)
    
    detailed_results = []
    for result in results:
        task_name = result.get("task", "Unknown")
        job_id = result.get("job_id")
        config_path = result.get("config_path")
        status = result.get("status")
        
        print(f"\n[{task_name}] Processing result: status={status}, job_id={job_id}")
        
        if status == "completed" and job_id:
            # Try to extract from events first
            print(f"[{task_name}] Trying to extract from events...")
            details = extract_results_from_events(job_id)
            print(f"[{task_name}] Events extraction result: {details}")
            
            if "error" not in details and details.get("baseline_score") is not None:
                details["task"] = task_name
                detailed_results.append(details)
                print(f"[{task_name}] Successfully extracted from events")
            else:
                # Fallback: extract from results file by job_id
                print(f"[{task_name}] Falling back to file extraction...")
                details = extract_results_from_file(task_name, job_id, config_path)
                details["task"] = task_name
                detailed_results.append(details)
                print(f"[{task_name}] File extraction result: {details}")
        else:
            error_msg = result.get("error", "Failed to complete or no job_id")
            print(f"[{task_name}] Job failed: {error_msg}")
            detailed_results.append({
                "task": task_name,
                "error": error_msg,
            })
    
    # Calculate aggregates
    valid_results = [r for r in detailed_results if "error" not in r and r.get("baseline_score") is not None]
    
    print("\n" + "=" * 80)
    print(f"Found {len(valid_results)} valid results out of {len(detailed_results)} total")
    print("=" * 80)
    
    if not valid_results:
        # Debug: show what we got
        print("\nNo valid results to aggregate!")
        print("\nDebug info:")
        for r in detailed_results:
            print(f"  {r.get('task')}: {r}")
        return
    
    # Aggregate stats
    total_cost = sum(r.get("total_cost", 0) or 0 for r in valid_results)
    total_rollouts = sum(r.get("total_rollouts", 0) or 0 for r in valid_results)
    total_time = sum(r.get("total_time", 0) or 0 for r in valid_results)
    total_tokens = sum(r.get("total_tokens", 0) or 0 for r in valid_results)
    
    avg_baseline = sum(r.get("baseline_score", 0) or 0 for r in valid_results) / len(valid_results)
    avg_candidate1 = sum(r.get("candidate1_score", 0) or 0 for r in valid_results) / len(valid_results)
    avg_lift = sum(r.get("candidate1_lift", 0) or 0 for r in valid_results) / len(valid_results)
    
    # Generate aggregate table output
    from datetime import datetime
    output_lines = []
    output_lines.append("\n" + "=" * 140)
    output_lines.append("AGGREGATE STATS ACROSS ALL TASKS")
    output_lines.append("=" * 140)
    output_lines.append("")
    output_lines.append(f"{'Task':<20} {'Policy Model':<25} {'Baseline':<12} {'Candidate 1':<14} {'Lift':<12} {'Rollouts':<10} {'Tokens':<12} {'Time':<10} {'Eval N':<8}")
    output_lines.append("-" * 140)
    
    for result in detailed_results:
        task = result.get("task", "Unknown")
        if "error" in result:
            output_lines.append(f"{task:<20} {'ERROR':<25} {'ERROR':<12} {'ERROR':<14} {'ERROR':<12} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8}")
        else:
            baseline = result.get("baseline_score")
            candidate1 = result.get("candidate1_score")
            lift = result.get("candidate1_lift")
            rollouts = result.get("total_rollouts", 0) or 0
            tokens = result.get("total_tokens", 0) or 0
            time_sec = result.get("total_time", 0) or 0
            eval_n = result.get("eval_seeds_n", 0) or 0
            policy_model = result.get("policy_model", "N/A")
            
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            candidate1_str = f"{candidate1:.4f}" if candidate1 is not None else "N/A"
            lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
            rollouts_str = str(rollouts) if rollouts > 0 else "N/A"
            tokens_str = f"{tokens/1e6:.2f}M" if tokens >= 1e6 else (f"{tokens/1e3:.1f}K" if tokens >= 1e3 else (str(tokens) if tokens > 0 else "N/A"))
            time_str = f"{time_sec:.1f}s" if time_sec < 60 else f"{time_sec/60:.1f}m" if time_sec > 0 else "N/A"
            eval_n_str = str(eval_n) if eval_n > 0 else "N/A"
            policy_model_str = str(policy_model) if policy_model else "N/A"
            
            output_lines.append(f"{task:<20} {policy_model_str:<25} {baseline_str:<12} {candidate1_str:<14} {lift_str:<12} {rollouts_str:<10} {tokens_str:<12} {time_str:<10} {eval_n_str:<8}")
    
    output_lines.append("-" * 140)
    tokens_str = f"{total_tokens/1e6:.2f}M" if total_tokens >= 1e6 else (f"{total_tokens/1e3:.1f}K" if total_tokens >= 1e3 else str(total_tokens))
    time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time/60:.1f}m"
    output_lines.append(f"{'TOTAL':<20} {'':<25} {'':<12} {'':<14} {'':<12} {total_rollouts:<10} {tokens_str:<12} {time_str:<10} {'':<8}")
    output_lines.append(f"{'AVERAGE':<20} {'':<25} {avg_baseline:.4f}     {avg_candidate1:.4f}     {avg_lift:+.4f} {'':<10} {'':<12} {'':<10} {'':<8}")
    output_lines.append("")
    output_lines.append(f"Total Cost: ${total_cost:.4f}")
    output_lines.append("=" * 140)
    
    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file
    comparisons_dir = REPO_ROOT / "examples" / "blog_posts" / "langprobe" / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = comparisons_dir / f"comparison_readout_{timestamp}.txt"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n📄 Comparison results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save comparison results file: {e}")


if __name__ == "__main__":
    asyncio.run(main())

