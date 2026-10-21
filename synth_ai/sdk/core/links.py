"""Current authorized provenance; an exact reference grants no access."""

from __future__ import annotations

from synth_ai.contracts.core import LINKS
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.options import ReadOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.refs import Link, Ref, Relation


def link_parameters(project_id, subject, object, relation, limit, cursor):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("Links limit must be in 1..=200")
    values = {"project_id": identifier(project_id), **page_parameters(limit, cursor)}
    if (subject is None) == (object is None):
        raise ValueError("Select exactly one links endpoint: subject or object")
    name, endpoint = ("subject", subject) if subject is not None else ("object", object)
    if not isinstance(endpoint, Ref):
        raise ValueError("A links endpoint must be an exact Ref")
    values[name] = ":".join(
        (endpoint.authority, endpoint.kind, endpoint.record_id, endpoint.revision)
    )
    values[name + "_digest_sha256"] = endpoint.digest_sha256
    if relation is not None:
        values["relation"] = Relation(relation).value
    return values


def link_page(page, project_id, subject, object, relation, limit):
    return verified(
        page,
        len(page.items) <= limit
        and all(
            item.project_id == project_id
            and (subject is None or item.subject == subject)
            and (object is None or item.object == object)
            and (relation is None or item.relation == relation)
            for item in page.items
        ),
        "Link page changed its selected project, endpoint, relation or bound",
    )


class Links:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        project_id: str,
        *,
        subject: Ref | None = None,
        object: Ref | None = None,
        relation: Relation | None = None,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        query = link_parameters(project_id, subject, object, relation, limit, cursor)
        return self._transport.request(
            LINKS,
            Page[Link],
            query=query,
            options=request,
            validate=lambda result: link_page(result, project_id, subject, object, relation, limit),
        )


class AsyncLinks:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(
        self,
        project_id: str,
        *,
        subject: Ref | None = None,
        object: Ref | None = None,
        relation: Relation | None = None,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        query = link_parameters(project_id, subject, object, relation, limit, cursor)
        return await self._transport.request(
            LINKS,
            Page[Link],
            query=query,
            options=request,
            validate=lambda result: link_page(result, project_id, subject, object, relation, limit),
        )
