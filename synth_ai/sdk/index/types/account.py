"""Read-only owner setup. See SDK index_contract.md."""

from pydantic import Field

from synth_ai.types.auth import AuthContext as AuthContext
from synth_ai.types.base import Identifier, ResponseModel


class IndexAccount(ResponseModel):
    tenant_id: Identifier
    search_enabled: bool
    contribution_write_enabled: bool


class Collection(ResponseModel):
    collection_id: Identifier
    name: str = Field(min_length=1, max_length=256)
    searchable: bool
    contents_readable: bool
