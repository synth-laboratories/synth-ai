"""Read owner-prepared template identity without a preparation client.

See README.md. The receipt schema remains v1; preparation and any benchmark
registration belong to the release owner's authorized path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["TemplateReceipt"]


class TemplateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["synth.craftax-template.v1"] = "synth.craftax-template.v1"
    base_url: str
    evaluator_revision_id: str
    dataset_revision_id: str
    template_pool_revision_id: str
    baseline_sha256: str
    field_sha256: str
    task_sha256: str
    provider: str
    region: str
    execution_pool_created: Literal[False] = False
    compute_started: Literal[False] = False
