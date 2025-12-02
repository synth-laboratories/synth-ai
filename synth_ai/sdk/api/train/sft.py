"""First-class SDK API for SFT (Supervised Fine-Tuning).

This module provides high-level abstractions for running SFT jobs
both via CLI (`uvx synth-ai train`) and programmatically in Python scripts.

Example CLI usage:
    uvx synth-ai train --type sft --config my_config.toml --poll

Example SDK usage:
    from synth_ai.sdk.api.train.sft import SFTJob
    
    job = SFTJob.from_config("my_config.toml")
    job.submit()
    result = job.poll_until_complete()
    print(f"Fine-tuned model: {result['fine_tuned_model']}")
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from synth_ai.core.telemetry import log_info

from .builders import SFTBuildResult, build_sft_payload
from .pollers import SFTJobPoller
from .utils import http_post, post_multipart, validate_sft_jsonl


@dataclass
class SFTJobConfig:
    """Configuration for an SFT job."""
    
    config_path: Path
    backend_url: str
    api_key: str
    dataset_override: Optional[Path] = None
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")


class SFTJob:
    """High-level SDK class for running SFT jobs.
    
    This class provides a clean API for:
    1. Submitting SFT jobs
    2. Polling job status
    3. Retrieving results
    
    Example:
        >>> from synth_ai.sdk.api.train.sft import SFTJob
        >>> 
        >>> # Create job from config
        >>> job = SFTJob.from_config(
        ...     config_path="my_config.toml",
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"]
        ... )
        >>> 
        >>> # Submit job
        >>> job_id = job.submit()
        >>> print(f"Job submitted: {job_id}")
        >>> 
        >>> # Poll until complete
        >>> result = job.poll_until_complete(timeout=3600.0)
        >>> print(f"Fine-tuned model: {result.get('fine_tuned_model')}")
    """
    
    def __init__(
        self,
        config: SFTJobConfig,
        job_id: Optional[str] = None,
    ) -> None:
        """Initialize an SFT job.
        
        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
        """
        self.config = config
        self._job_id = job_id
        self._build_result: Optional[SFTBuildResult] = None
        self._train_file_id: Optional[str] = None
        self._val_file_id: Optional[str] = None
    
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset_override: Optional[str | Path] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> SFTJob:
        """Create a job from a TOML config file.
        
        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            dataset_override: Override dataset path from config
            allow_experimental: Allow experimental models
            overrides: Config overrides
            
        Returns:
            SFTJob instance
            
        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist
        """
        from synth_ai.core.env import get_backend_from_env
        
        config_path_obj = Path(config_path)
        
        # Resolve backend URL
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base
        
        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )
        
        config = SFTJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            dataset_override=Path(dataset_override) if dataset_override else None,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )
        
        return cls(config)
    
    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> SFTJob:
        """Resume an existing job by ID.
        
        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            
        Returns:
            SFTJob instance for the existing job
        """
        from synth_ai.core.env import get_backend_from_env
        
        # Resolve backend URL
        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base
        
        # Resolve API key
        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )
        
        # Create minimal config (we don't need the config file for resuming)
        config = SFTJobConfig(
            config_path=Path("/dev/null"),  # Dummy path
            backend_url=backend_url,
            api_key=api_key,
        )
        
        return cls(config, job_id=job_id)
    
    def _build_payload(self) -> SFTBuildResult:
        """Build the job payload from config."""
        if self._build_result is None:
            if not self.config.config_path.exists() or self.config.config_path.name == "/dev/null":
                raise RuntimeError(
                    "Cannot build payload: config_path is required for new jobs. "
                    "Use from_job_id() to resume an existing job."
                )
            
            self._build_result = build_sft_payload(
                config_path=self.config.config_path,
                dataset_override=self.config.dataset_override,
                allow_experimental=self.config.allow_experimental,
            )
        return self._build_result
    
    def submit(self) -> str:
        """Submit the job to the backend.

        Returns:
            Job ID

        Raises:
            RuntimeError: If job submission fails
        """
        ctx: Dict[str, Any] = {"config_path": str(self.config.config_path)}
        log_info("SFTJob.submit invoked", ctx=ctx)
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()
        
        # Validate datasets
        validate_sft_jsonl(build.train_file)
        if build.validation_file and build.validation_file.suffix == ".jsonl":
            validate_sft_jsonl(build.validation_file)
        
        # Upload training file
        upload_url = f"{self.config.backend_url.rstrip('/')}/files"
        resp = post_multipart(
            upload_url,
            api_key=self.config.api_key,
            file_field="file",
            file_path=build.train_file,
        )
        
        js = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        
        if resp.status_code is not None and resp.status_code >= 400 or "id" not in js:
            raise RuntimeError(
                f"Training file upload failed with status {resp.status_code}: {js or resp.text[:400]}"
            )
        
        self._train_file_id = js["id"]
        
        # Upload validation file if present
        if build.validation_file:
            vresp = post_multipart(
                upload_url,
                api_key=self.config.api_key,
                file_field="file",
                file_path=build.validation_file,
            )
            vjs = (
                vresp.json()
                if vresp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            if vresp.status_code is not None and vresp.status_code < 400 and "id" in vjs:
                self._val_file_id = vjs["id"]
        
        # Build payload with file IDs
        payload = dict(build.payload)
        payload["training_file_id"] = self._train_file_id
        if self._val_file_id:
            payload.setdefault("metadata", {}).setdefault("effective_config", {}).setdefault(
                "data", {}
            )["validation_files"] = [self._val_file_id]
        
        # Submit job
        create_url = f"{self.config.backend_url.rstrip('/')}/learning/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        resp = http_post(create_url, headers=headers, json_body=payload)
        
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Job submission failed with status {resp.status_code}: {resp.text[:500]}"
            )
        
        try:
            js = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}") from e
        
        job_id = js.get("job_id") or js.get("id")
        if not job_id:
            raise RuntimeError("Response missing job ID")

        self._job_id = job_id
        ctx["job_id"] = job_id
        log_info("SFTJob.submit completed", ctx=ctx)
        return job_id

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.
        
        Returns:
            Job status dictionary
            
        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        from synth_ai.sdk.learning.client import LearningClient
        
        async def _fetch() -> Dict[str, Any]:
            client = LearningClient(
                self.config.backend_url,
                self.config.api_key,
                timeout=30.0,
            )
            result = await client.get_job(self._job_id)  # type: ignore[arg-type]  # We check None above
            return dict(result) if isinstance(result, dict) else {}
        
        return asyncio.run(_fetch())
    
    def poll_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 5.0,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll job until it reaches a terminal state.
        
        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between poll attempts
            on_status: Optional callback called on each status update
            
        Returns:
            Final job status dictionary
            
        Raises:
            RuntimeError: If job hasn't been submitted yet
            TimeoutError: If timeout is exceeded
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        poller = SFTJobPoller(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            interval=interval,
            timeout=timeout,
        )
        
        outcome = poller.poll_job(self._job_id)  # type: ignore[arg-type]  # We check None above
        
        payload = dict(outcome.payload) if isinstance(outcome.payload, dict) else {}
        
        if on_status:
            on_status(payload)
        
        return payload
    
    def get_fine_tuned_model(self) -> Optional[str]:
        """Get the fine-tuned model ID from completed job.
        
        Returns:
            Fine-tuned model ID or None if not available
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        
        status = self.get_status()
        return status.get("fine_tuned_model")


__all__ = [
    "SFTJob",
    "SFTJobConfig",
]

