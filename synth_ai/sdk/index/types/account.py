"""Read-only owner setup. See SDK index_contract.md."""

from pydantic import Field

from synth_ai.types.auth import AuthContext as AuthContext
from synth_ai.types.base import ResponseIdentifier, ResponseModel


class IndexAccount(ResponseModel):
    tenant_id: ResponseIdentifier
    search_enabled: bool
    contribution_write_enabled: bool


class Collection(ResponseModel):
    collection_id: ResponseIdentifier
    name: str = Field(min_length=1, max_length=256)
    searchable: bool
    contents_readable: bool
