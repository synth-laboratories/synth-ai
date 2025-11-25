"""SDK jobs client for programmatic access to Synth AI jobs API.

This module provides a high-level async client for managing:
- Files (upload, list, retrieve, delete)
- SFT training jobs (create, list, retrieve, cancel, events, checkpoints)
- RL training jobs (create, list, retrieve, cancel, events, metrics)
- Models (list, retrieve, delete, list_jobs)

Usage:
    from synth_ai.sdk.jobs import JobsClient

    async with JobsClient(base_url, api_key) as client:
        # Upload a training file
        file = await client.files.upload(
            filename="train.jsonl",
            content=data,
            purpose="fine-tune"
        )

        # Create an SFT job
        job = await client.sft.create(
            training_file=file["id"],
            model="gpt-4o-mini"
        )

        # Check job status
        status = await client.sft.retrieve(job["id"])
"""

from __future__ import annotations

# Re-export from the actual implementation
from synth_ai.jobs.client import (
    FilesApi,
    JobsClient,
    ModelsApi,
    RlJobsApi,
    SftJobsApi,
)

__all__ = [
    "JobsClient",
    "FilesApi",
    "SftJobsApi",
    "RlJobsApi",
    "ModelsApi",
]

