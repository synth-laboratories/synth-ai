"""Frozen SDK-first operation contract. See backend SDK index_contract.md.

Provisional until backend pure-contract export adopts and verifies these facts.
"""

import re
from dataclasses import dataclass
from typing import Literal

CONTRACT = "index.v1"
PREFIX = "/api/v1"


@dataclass(frozen=True, slots=True)
class Operation:
    id: str
    method: Literal["GET", "POST", "PATCH"]
    path: str
    keyed: bool
    mutation: bool = False
    contract: Literal["index.v1", "swarms.v1"] = "index.v1"
    required_scope: str | None = None
    may_charge: bool = False

    @property
    def idempotent(self) -> bool:
        return not self.mutation or self.keyed

    def path_for(self, **identifiers: str) -> str:
        expected = set(re.findall(r"{([a-z_]+)}", self.path))
        if set(identifiers) != expected or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            for value in identifiers.values()
        ):
            raise ValueError("operation path requires exactly its bounded identifiers")
        return self.path.format(**identifiers)

    def matches_resource_path(self, resource_path: str) -> bool:
        parts = re.split(r"({[a-z_]+})", PREFIX + self.path)
        pattern = "".join(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}" if part.startswith("{") else re.escape(part)
            for part in parts
        )
        return re.fullmatch(pattern, resource_path) is not None


AUTH_CONTEXT = Operation("auth.context", "GET", "/auth/context", True, required_scope=None)
CAPABILITIES = Operation("capabilities.get", "GET", "/capabilities", False)
RESOLVE = Operation("mutations.resolve", "GET", "/mutations/resolve", True, required_scope=None)
PUBLIC_SEARCH = Operation(
    "index.public.search", "POST", "/index/public/search", False, True, required_scope=None
)
SEARCH = Operation(
    "index.search",
    "POST",
    "/index/search",
    True,
    True,
    required_scope="index:search",
    may_charge=True,
)
PUBLIC_CONTENTS = Operation(
    "index.public.contents.get", "POST", "/index/public/contents", False, required_scope=None
)
CONTENTS = Operation(
    "index.contents.get", "POST", "/index/contents", True, required_scope="index:read"
)
CREATE_SEARCH = Operation(
    "index.searches.create",
    "POST",
    "/index/searches",
    True,
    True,
    required_scope="index:search",
    may_charge=True,
)
GET_SEARCH = Operation(
    "index.searches.get", "GET", "/index/searches/{search_id}", True, required_scope="index:read"
)
LIST_SEARCHES = Operation(
    "index.searches.list", "GET", "/index/searches", True, required_scope="index:read"
)
SEARCH_HISTORY = Operation(
    "index.searches.history",
    "POST",
    "/index/searches/history",
    True,
    required_scope="index:read",
)
DELETE_SEARCH_HISTORY = Operation(
    "index.searches.history.delete",
    "POST",
    "/index/searches/{search_id}/history/delete",
    True,
    True,
    required_scope="index:search",
)
SEARCH_EVENTS = Operation(
    "index.searches.events",
    "GET",
    "/index/searches/{search_id}/events",
    True,
    required_scope="index:read",
)
SEARCH_RESULT = Operation(
    "index.searches.result",
    "GET",
    "/index/searches/{search_id}/result",
    True,
    required_scope="index:read",
)
CANCEL_SEARCH = Operation(
    "index.searches.cancel",
    "POST",
    "/index/searches/{search_id}/cancel",
    True,
    True,
    required_scope="index:search",
)
CONTROL = Operation(
    "index.searches.control",
    "GET",
    "/index/searches/{search_id}/controls/{operation_id}",
    True,
    required_scope="index:read",
)
SEARCH_USAGE = Operation(
    "index.searches.usage",
    "GET",
    "/index/searches/{search_id}/usage",
    True,
    required_scope="index:read",
)
SEARCH_USAGE_ITEMS = Operation(
    "index.searches.usage_operations",
    "GET",
    "/index/searches/{search_id}/usage/operations",
    True,
    required_scope="index:read",
)
USAGE = Operation(
    "index.usage.get", "GET", "/index/usage/{operation_id}", True, required_scope="index:read"
)
USAGE_ITEMS = Operation(
    "index.usage.operations",
    "GET",
    "/index/usage/{operation_id}/operations",
    True,
    required_scope="index:read",
)
ACCOUNT = Operation("index.account.get", "GET", "/index/account", True, required_scope="index:read")
COLLECTIONS = Operation(
    "index.collections.list", "GET", "/index/collections", True, required_scope="index:read"
)
COLLECTION = Operation(
    "index.collections.get",
    "GET",
    "/index/collections/{collection_id}",
    True,
    required_scope="index:read",
)
CONTRIBUTIONS = Operation(
    "index.contributions.list", "GET", "/index/contributions", False, required_scope=None
)
CONTRIBUTION = Operation(
    "index.contributions.get",
    "GET",
    "/index/contributions/{contribution_id}",
    False,
    required_scope=None,
)
CONTRIBUTION_REVISION = Operation(
    "index.contributions.revisions.get",
    "GET",
    "/index/contributions/{contribution_id}/revisions/{revision_id}",
    False,
    required_scope=None,
)
OWNED_CONTRIBUTION = Operation(
    "index.contributions.owner",
    "GET",
    "/index/contributions/{contribution_id}/owner",
    True,
    required_scope="index:intake",
)
CREATE_CONTRIBUTION = Operation(
    "index.contributions.create",
    "POST",
    "/index/contributions",
    True,
    True,
    required_scope="index:intake",
)
UPDATE_CONTRIBUTION = Operation(
    "index.contributions.update",
    "PATCH",
    "/index/contributions/{contribution_id}",
    True,
    True,
    required_scope="index:intake",
)
ATTEST = Operation(
    "index.contributions.attest",
    "POST",
    "/index/contributions/{contribution_id}/attest",
    True,
    True,
    required_scope="index:intake",
)
SUBMIT = Operation(
    "index.contributions.submit",
    "POST",
    "/index/contributions/{contribution_id}/submit",
    True,
    True,
    required_scope="index:intake",
)
CORRECT = Operation(
    "index.contributions.correct",
    "POST",
    "/index/contributions/{contribution_id}/corrections",
    True,
    True,
    required_scope="index:intake",
)
PREPARE_UPLOAD = Operation(
    "index.contributions.uploads.prepare",
    "POST",
    "/index/contributions/{contribution_id}/uploads",
    True,
    True,
    required_scope="index:intake",
)
GET_UPLOAD = Operation(
    "index.contributions.uploads.get",
    "GET",
    "/index/uploads/{upload_id}",
    True,
    required_scope="index:intake",
)
FINALIZE_UPLOAD = Operation(
    "index.uploads.finalize",
    "POST",
    "/index/uploads/{upload_id}/finalize",
    True,
    True,
    required_scope="index:intake",
)

ACCOUNT_ENTITLEMENTS = Operation(
    "account.entitlements", "GET", "/account/entitlements", True, required_scope=None
)

ALL_OPERATIONS: tuple[Operation, ...] = (
    CAPABILITIES,
    AUTH_CONTEXT,
    ACCOUNT_ENTITLEMENTS,
    RESOLVE,
    PUBLIC_SEARCH,
    SEARCH,
    PUBLIC_CONTENTS,
    CONTENTS,
    CREATE_SEARCH,
    GET_SEARCH,
    LIST_SEARCHES,
    SEARCH_HISTORY,
    DELETE_SEARCH_HISTORY,
    SEARCH_EVENTS,
    SEARCH_RESULT,
    CANCEL_SEARCH,
    CONTROL,
    SEARCH_USAGE,
    SEARCH_USAGE_ITEMS,
    USAGE,
    USAGE_ITEMS,
    ACCOUNT,
    COLLECTIONS,
    COLLECTION,
    CONTRIBUTIONS,
    CONTRIBUTION,
    CONTRIBUTION_REVISION,
    OWNED_CONTRIBUTION,
    CREATE_CONTRIBUTION,
    UPDATE_CONTRIBUTION,
    ATTEST,
    SUBMIT,
    CORRECT,
    PREPARE_UPLOAD,
    GET_UPLOAD,
    FINALIZE_UPLOAD,
)
