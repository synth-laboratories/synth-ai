"""Task app validation utilities."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

import click
import httpx

from synth_ai.sdk.task.contracts import TaskAppEndpoints  # type: ignore[attr-defined]


def validate_rollout_response_for_rl(response_data: dict[str, Any], *, warn_only: bool = False) -> list[str]:
    """Validate that a task app rollout response has required fields for RL training.
    
    The backend RL trainer requires:
    1. pipeline_metadata["inference_url"] at top level (with ?cid= for trace correlation)
    2. Each step's info.meta["inference_url"] must be present (nested structure!)
    
    Args:
        response_data: The rollout response dict from task app
        warn_only: If True, return warnings instead of raising exceptions
        
    Returns:
        List of validation warnings/errors
        
    Raises:
        ValueError: If critical fields are missing (unless warn_only=True)
    """
    issues = []
    
    # Check pipeline_metadata
    pipeline_metadata = response_data.get("pipeline_metadata")
    if not isinstance(pipeline_metadata, dict):
        issues.append("Missing or invalid 'pipeline_metadata' (required for RL training)")
    else:
        inference_url = pipeline_metadata.get("inference_url")
        if not inference_url:
            issues.append(
                "pipeline_metadata['inference_url'] is missing. "
                "RL trainer requires this field to extract traces."
            )
        elif not isinstance(inference_url, str):
            issues.append(
                f"pipeline_metadata['inference_url'] must be a string, got: {type(inference_url).__name__}"
            )
        elif "?cid=" not in inference_url:
            issues.append(
                f"pipeline_metadata['inference_url'] should contain '?cid=' for trace correlation. "
                f"Got: {inference_url[:80]}..."
            )
    
    # Check trajectories and steps
    trajectories = response_data.get("trajectories", [])
    if not trajectories:
        issues.append("No trajectories found in response")
    
    for traj_idx, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            continue
        
        steps = trajectory.get("steps", [])
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            
            step_info = step.get("info", {})
            if not isinstance(step_info, dict):
                issues.append(
                    f"trajectory[{traj_idx}].steps[{step_idx}].info is not a dict"
                )
                continue
            
            # Check for nested meta.inference_url (backend expects this structure!)
            step_meta = step_info.get("meta", {})
            if not isinstance(step_meta, dict):
                issues.append(
                    f"trajectory[{traj_idx}].steps[{step_idx}].info.meta is missing or not a dict. "
                    f"RL trainer expects nested structure: info.meta.inference_url"
                )
                continue
            
            step_inference_url = step_meta.get("inference_url")
            if not step_inference_url:
                issues.append(
                    f"trajectory[{traj_idx}].steps[{step_idx}].info.meta['inference_url'] is missing. "
                    f"RL trainer needs this for trace extraction (nested structure required!)"
                )
            elif not isinstance(step_inference_url, str):
                issues.append(
                    f"trajectory[{traj_idx}].steps[{step_idx}].info.meta['inference_url'] must be a string, "
                    f"got: {type(step_inference_url).__name__}"
                )
    
    if issues and not warn_only:
        error_msg = "Task app response validation failed for RL training:\n" + "\n".join(
            f"  - {issue}" for issue in issues
        )
        raise ValueError(error_msg)
    
    return issues


def normalize_inference_url(url: str | None, *, default: str = "https://api.openai.com/v1/chat/completions") -> str:
    """Normalize an inference URL to include the /v1/chat/completions path.
    
    This utility ensures inference URLs have the correct path structure for OpenAI-compatible
    chat completions endpoints, while preserving query parameters (e.g., ?cid=trace_123) 
    that may be added for tracing.
    
    Args:
        url: The inference URL to normalize (may be None or incomplete)
        default: Default URL to use if url is None/empty
        
    Returns:
        Normalized URL with proper path and preserved query parameters
        
    Examples:
        >>> normalize_inference_url("https://api.groq.com")
        'https://api.groq.com/v1/chat/completions'
        
        >>> normalize_inference_url("https://modal.host?cid=trace_123")
        'https://modal.host/v1/chat/completions?cid=trace_123'
        
        >>> normalize_inference_url("https://api.openai.com/v1")
        'https://api.openai.com/v1/chat/completions'
        
        >>> normalize_inference_url("https://api.groq.com/openai/v1/chat/completions")
        'https://api.groq.com/openai/v1/chat/completions'
    """
    candidate = (url or default).strip()
    if not candidate:
        candidate = default
    
    parsed = urlparse(candidate)
    path = (parsed.path or "").rstrip("/")
    query = parsed.query or ""

    # Repair malformed URLs where the completions path ended up in the query string.
    # Example: https://host?cid=trace/v1/chat/completions  ->  https://host/v1/chat/completions?cid=trace
    if query and "/" in query:
        base_query, remainder = query.split("/", 1)
        remainder_path = remainder
        extra_query = ""
        for separator in ("&", "?"):
            idx = remainder_path.find(separator)
            if idx != -1:
                extra_query = remainder_path[idx + 1 :]
                remainder_path = remainder_path[:idx]
                break

        query_path = "/" + remainder_path.lstrip("/")
        merged_query_parts: list[str] = []
        if base_query:
            merged_query_parts.append(base_query)
        if extra_query:
            merged_query_parts.append(extra_query)
        merged_query = "&".join(part for part in merged_query_parts if part)

        if query_path and query_path != "/":
            combined_path = f"{path.rstrip('/')}{query_path}" if path else query_path
        else:
            combined_path = path

        parsed = parsed._replace(path=combined_path or "", query=merged_query)
        path = (parsed.path or "").rstrip("/")
        query = parsed.query or ""

    # Check if path already ends with a completions endpoint
    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        final_query = parsed.query or ""
        if final_query and "/" in final_query:
            parsed = parsed._replace(query=final_query.split("/", 1)[0])
        result = urlunparse(parsed)
        return str(result)
    
    # Determine what to append based on existing path
    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif path.endswith("/chat"):
        new_path = f"{path}/completions"
    else:
        new_path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"

    parsed = parsed._replace(path=new_path)
    final_query = parsed.query or ""
    if final_query and "/" in final_query:
        parsed = parsed._replace(query=final_query.split("/", 1)[0])

    result = urlunparse(parsed)
    return str(result)


def validate_task_app_url(url: str | None) -> str:
    """Validate and normalize a task app URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Normalized URL
        
    Raises:
        ValueError: If URL is invalid
    """
    if not url:
        raise ValueError("Task app URL is required")
    
    url = url.strip().rstrip("/")
    
    # Basic URL validation
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    
    if not url_pattern.match(url):
        raise ValueError(f"Invalid task app URL: {url}")
    
    return url


def _print_success(msg: str) -> None:
    """Print success message in green."""
    click.echo(click.style(f"✓ {msg}", fg="green"))


def _print_error(msg: str) -> None:
    """Print error message in red."""
    click.echo(click.style(f"✗ {msg}", fg="red"), err=True)


def _print_warning(msg: str) -> None:
    """Print warning message in yellow."""
    click.echo(click.style(f"⚠ {msg}", fg="yellow"))


def _print_info(msg: str) -> None:
    """Print info message."""
    click.echo(f"  {msg}")


async def validate_task_app_endpoint(
    url: str,
    api_key: str | None = None,
    min_instances: int = 10,
    verbose: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Validate a task app deployment.
    
    Returns:
        (success: bool, results: dict)
    """
    results: dict[str, Any] = {
        "url": url,
        "endpoints": {},
        "auth": {},
        "task_instances": {},
        "overall": False,
    }
    
    all_passed = True
    endpoints = TaskAppEndpoints()
    
    # Set up headers
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Validating Task App: {url}")
    click.echo(f"{'='*60}\n")
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # 1. Check root endpoint
        click.echo("1. Checking root endpoint...")
        try:
            resp = await client.get(f"{url}{endpoints.root}")
            if resp.status_code == 200:
                data = resp.json()
                _print_success(f"Root endpoint responds (status: {data.get('status')})")
                results["endpoints"]["root"] = {"passed": True, "data": data}
                if verbose:
                    _print_info(f"Service: {data.get('service', 'N/A')}")
            else:
                _print_error(f"Root endpoint returned {resp.status_code}")
                results["endpoints"]["root"] = {"passed": False, "status": resp.status_code}
                all_passed = False
        except Exception as e:
            _print_error(f"Root endpoint failed: {e}")
            results["endpoints"]["root"] = {"passed": False, "error": str(e)}
            all_passed = False
        
        # 2. Check health endpoint
        click.echo("\n2. Checking health endpoint...")
        try:
            resp = await client.get(f"{url}{endpoints.health}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                _print_success(f"Health endpoint responds (healthy: {data.get('healthy')})")
                results["endpoints"]["health"] = {"passed": True, "data": data}
                
                # Check auth configuration
                auth_info = data.get("auth", {})
                if auth_info.get("required"):
                    _print_info(f"Auth required: {auth_info.get('required')}")
                    _print_info(f"Expected key prefix: {auth_info.get('expected_prefix', 'N/A')}")
                    
                    if api_key:
                        _print_success("API key provided and accepted")
                        results["auth"]["provided"] = True
                        results["auth"]["accepted"] = True
                    else:
                        _print_warning("No API key provided but may be required")
                        results["auth"]["provided"] = False
                        results["auth"]["required"] = True
            else:
                _print_error(f"Health endpoint returned {resp.status_code}")
                results["endpoints"]["health"] = {"passed": False, "status": resp.status_code}
                all_passed = False
                
                if resp.status_code == 403:
                    _print_error("Authentication failed - provide API key with --api-key")
                    results["auth"]["error"] = "Authentication failed"
                    
        except Exception as e:
            _print_error(f"Health endpoint failed: {e}")
            results["endpoints"]["health"] = {"passed": False, "error": str(e)}
            all_passed = False
        
        # 3. Check info endpoint
        click.echo("\n3. Checking info endpoint...")
        try:
            resp = await client.get(f"{url}{endpoints.info}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                _print_success("Info endpoint responds")
                results["endpoints"]["info"] = {"passed": True, "data": data}
                
                if verbose:
                    service = data.get("service", {})
                    task_info = service.get("task", {})
                    if isinstance(task_info, dict):
                        _print_info(f"Task: {task_info.get('name', 'N/A')}")
                    _print_info(f"Version: {service.get('version', 'N/A')}")
                    
                    dataset = data.get("dataset", {})
                    if isinstance(dataset, dict):
                        _print_info(f"Dataset: {dataset.get('id', 'N/A')}")
            else:
                _print_error(f"Info endpoint returned {resp.status_code}")
                results["endpoints"]["info"] = {"passed": False, "status": resp.status_code}
                all_passed = False
        except Exception as e:
            _print_error(f"Info endpoint failed: {e}")
            results["endpoints"]["info"] = {"passed": False, "error": str(e)}
            all_passed = False
        
        # 4. Check task_info endpoint and instance count
        click.echo("\n4. Checking task_info endpoint and instance availability...")
        try:
            # Get taskset descriptor first
            resp = await client.get(f"{url}{endpoints.task_info}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                _print_success("Task info endpoint responds")
                results["endpoints"]["task_info"] = {"passed": True}
                
                taskset = data.get("taskset", {})
                if verbose and taskset:
                    if isinstance(taskset, dict):
                        _print_info(f"Taskset: {taskset.get('id', 'N/A')}")
                    else:
                        _print_info(f"Taskset: {taskset}")
                
                # Try to get specific task instances (seeds 0-19)
                # Fetch instances one by one to verify we can get at least min_instances
                instances = []
                for seed in range(min_instances + 5):  # Try a few extra
                    try:
                        resp_seed = await client.get(
                            f"{url}{endpoints.task_info}",
                            params={"seed": seed},
                            headers=headers,
                        )
                        if resp_seed.status_code == 200:
                            instance = resp_seed.json()
                            instances.append(instance)
                        else:
                            break  # Stop if we hit an invalid seed
                    except Exception:
                        break
                
                instance_count = len(instances)
                results["task_instances"]["count"] = instance_count
                results["task_instances"]["requested"] = min_instances
                
                if instance_count >= min_instances:
                    _print_success(f"Found {instance_count} task instances (≥ {min_instances} required)")
                    results["task_instances"]["passed"] = True
                    
                    if verbose and instances:
                        sample = instances[0]
                        task_info_sample = sample.get('task', {})
                        if isinstance(task_info_sample, dict):
                            _print_info(f"Sample task: {task_info_sample.get('name', 'N/A')}")
                        _print_info(f"Environment: {sample.get('environment', 'N/A')}")
                else:
                    _print_error(f"Only {instance_count} task instances available (need ≥ {min_instances})")
                    results["task_instances"]["passed"] = False
                    all_passed = False
            else:
                _print_error(f"Task info endpoint returned {resp.status_code}")
                results["endpoints"]["task_info"] = {"passed": False, "status": resp.status_code}
                all_passed = False
        except Exception as e:
            _print_error(f"Task info endpoint failed: {e}")
            results["endpoints"]["task_info"] = {"passed": False, "error": str(e)}
            results["task_instances"]["passed"] = False
            all_passed = False
        
        # 5. Check rollout endpoint structure (don't actually run a rollout)
        click.echo("\n5. Checking rollout endpoint availability...")
        try:
            # Just check if it's registered (OPTIONS or a lightweight probe)
            resp = await client.options(f"{url}{endpoints.rollout}", headers=headers)
            # Many servers return 200 for OPTIONS, some return 405
            if resp.status_code in (200, 204, 405):
                _print_success("Rollout endpoint is registered")
                results["endpoints"]["rollout"] = {"passed": True}
            else:
                _print_warning(f"Rollout endpoint returned unexpected status: {resp.status_code}")
                results["endpoints"]["rollout"] = {"passed": True, "note": "endpoint exists"}
        except Exception as e:
            # OPTIONS might not be supported, that's okay
            _print_info(f"Rollout endpoint check skipped (OPTIONS not supported): {e}")
            results["endpoints"]["rollout"] = {"passed": True, "note": "assumed present"}
    
    # Summary
    click.echo(f"\n{'='*60}")
    if all_passed:
        _print_success("All validations passed!")
        click.echo(f"{'='*60}\n")
    else:
        _print_error("Some validations failed. See errors above.")
        click.echo(f"{'='*60}\n")
    
    results["overall"] = all_passed
    return all_passed, results
