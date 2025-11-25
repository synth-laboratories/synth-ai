"""HTTP client for artifacts API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from synth_ai.core._utils.http import AsyncHttpClient

logger = logging.getLogger(__name__)


class ArtifactsClient:
    """Client for artifacts API endpoints.
    
    Provides methods to interact with the Synth AI artifacts API, including
    listing artifacts, retrieving model and prompt details, exporting models
    to HuggingFace, and accessing prompt snapshots.
    """
    
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        """Initialize the artifacts client.
        
        Args:
            base_url: Base URL of the backend API (e.g., "https://api.usesynth.ai")
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
    
    async def list_artifacts(
        self,
        *,
        artifact_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List all artifacts (models and prompts)."""
        params: Dict[str, Any] = {"limit": limit}
        if artifact_type:
            params["type"] = artifact_type
        if status:
            params["status"] = status
        
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get("/api/artifacts", params=params)
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model artifact."""
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/artifacts/models/{model_id}")
    
    async def get_prompt(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information about a prompt optimization job."""
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/artifacts/prompts/{job_id}")
    
    async def export_to_huggingface(
        self,
        *,
        wasabi_key: str,
        repo_id: str,
        repo_type: str = "model",
        artifact_kind: str = "lora",
        base_model: Optional[str] = None,
        visibility: str = "private",
        tags: Optional[List[str]] = None,
        bucket: Optional[str] = None,
        folder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a model artifact to HuggingFace Hub."""
        body: Dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "artifact_kind": artifact_kind,
            "key": wasabi_key,
            "visibility": visibility,
        }
        if base_model:
            body["base_model"] = base_model
        if tags:
            body["tags"] = tags
        if bucket:
            body["bucket"] = bucket
        if folder_name:
            body["folder_name"] = folder_name
        
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.post_json("/api/learning/exports/hf", json=body)
    
    async def get_prompt_snapshot(
        self,
        job_id: str,
        snapshot_id: str,
    ) -> Dict[str, Any]:
        """Get a specific prompt snapshot."""
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(f"/api/prompt-learning/online/jobs/{job_id}/snapshots/{snapshot_id}")
    
    async def list_prompt_snapshots(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """List all artifacts (snapshots) for a prompt job."""
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            artifacts = await http.get(f"/api/prompt-learning/online/jobs/{job_id}/artifacts")
            if isinstance(artifacts, list):
                return artifacts
            if isinstance(artifacts, dict) and isinstance(artifacts.get("artifacts"), list):
                return artifacts["artifacts"]
            return []
    
    async def get_models_on_wasabi(self) -> Dict[str, Any]:
        """Get models available on Wasabi."""
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get("/api/learning/models/on-wasabi")

