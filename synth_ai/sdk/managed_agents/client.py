"""Anthropic-shaped client facade for Synth Managed Agents."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any, List

from synth_ai.sdk.managed_agents_anthropic import ManagedAgentsAnthropicClient


def _clean_body(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if value is not None}


def _merge_body(extra_body: dict[str, Any] | None, body: dict[str, Any]) -> dict[str, Any]:
    merged = _clean_body(body)
    if extra_body:
        merged.update(extra_body)
    return merged


def _read_upload(file: Any) -> tuple[str, bytes, str]:
    if isinstance(file, (str, Path)):
        path = Path(file)
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return path.name, content, content_type
    if isinstance(file, bytes):
        return "upload.bin", file, "application/octet-stream"
    read = getattr(file, "read", None)
    if callable(read):
        raw = read()
        content = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        name = str(getattr(file, "name", None) or "upload.bin")
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Path(name).name, content, content_type
    raise TypeError("file must be a path, bytes, or file-like object")


class _Resource:
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        self._transport = transport

    @property
    def with_raw_response(self) -> _Resource:
        return self

    @property
    def with_streaming_response(self) -> _Resource:
        return self


class _AgentsResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.versions = _AgentVersionsResource(transport)

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.create_agent(_merge_body(body.pop("extra_body", None), body))

    def retrieve(self, agent_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.get_agent(agent_id)

    def update(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.update_agent(
            agent_id, _merge_body(body.pop("extra_body", None), body)
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.list_agents(**_clean_body(params))

    def archive(self, agent_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.archive_agent(agent_id)


class _AgentVersionsResource(_Resource):
    def list(self, agent_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/agents/{agent_id}/versions",
            params=_clean_body(params) or None,
        )


class _EnvironmentsResource(_Resource):
    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.create_environment(_merge_body(body.pop("extra_body", None), body))

    def retrieve(self, environment_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.get_environment(environment_id)

    def update(self, environment_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.update_environment(
            environment_id,
            _merge_body(body.pop("extra_body", None), body),
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.list_environments(**_clean_body(params))

    def delete(self, environment_id: str, **_: Any) -> dict[str, Any]:
        return self.archive(environment_id)

    def archive(self, environment_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.archive_environment(environment_id)


class _SessionsResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.events = _SessionEventsResource(transport)
        self.resources = _SessionResourcesResource(transport)
        self.threads = _SessionThreadsResource(transport)

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.create_session(_merge_body(body.pop("extra_body", None), body))

    def retrieve(self, session_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.get_session(session_id)

    def update(self, session_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.update_session(
            session_id,
            _merge_body(body.pop("extra_body", None), body),
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.list_sessions(**_clean_body(params))

    def delete(self, session_id: str, **_: Any) -> dict[str, Any]:
        return self.archive(session_id)

    def archive(self, session_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.archive_session(session_id)

    def files(self, session_id: str, **params: Any) -> dict[str, Any]:
        cleaned = _clean_body(params)
        try:
            return self._transport.list_session_files(session_id, **cleaned)
        except Exception:
            fallback = dict(cleaned)
            fallback.setdefault("scope", "session")
            fallback.setdefault("scope_id", session_id)
            return self._transport.list_files(**fallback)

    def attach_resource(self, session_id: str, **body: Any) -> dict[str, Any]:
        return self.resources.create(session_id, **body)


class _SessionEventsResource(_Resource):
    def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.list_session_events(session_id, **_clean_body(params))

    def send(
        self, session_id: str, *, events: List[dict[str, Any]] | None = None, **body: Any
    ) -> dict[str, Any]:
        payload = _merge_body(body.pop("extra_body", None), body)
        if events is not None:
            payload["events"] = events
        return self._transport.post_session_events(session_id, payload)

    def create(
        self, session_id: str, *, events: List[dict[str, Any]] | None = None, **body: Any
    ) -> dict[str, Any]:
        return self.send(session_id, events=events, **body)

    def stream(self, session_id: str, **params: Any) -> Iterator[dict[str, Any]]:
        return self._transport.stream_session_events(session_id, **_clean_body(params))


class _SessionResourcesResource(_Resource):
    def create(self, session_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.add_session_resource(
            session_id,
            _merge_body(body.pop("extra_body", None), body),
        )

    def retrieve(self, session_id: str, resource_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.get_session_resource(session_id, resource_id)

    def update(self, session_id: str, resource_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.update_session_resource(
            session_id,
            resource_id,
            _merge_body(body.pop("extra_body", None), body),
        )

    def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.list_session_resources(session_id, **_clean_body(params))

    def delete(self, session_id: str, resource_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.delete_session_resource(session_id, resource_id)


class _SessionThreadsResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.events = _SessionThreadEventsResource(transport)

    def retrieve(self, session_id: str, thread_id: str, **_: Any) -> dict[str, Any]:
        page = self.list(session_id)
        for item in page.get("data", []):
            if str(item.get("id") or "") == str(thread_id):
                return dict(item)
        return self._transport.request("GET", f"/sessions/{session_id}/threads/{thread_id}")

    def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.list_session_threads(session_id, **_clean_body(params))

    def archive(self, session_id: str, thread_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/sessions/{session_id}/threads/{thread_id}/archive",
            json_body={},
        )


class _SessionThreadEventsResource(_Resource):
    def list(self, session_id: str, thread_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.list_session_thread_events(
            session_id,
            thread_id,
            **_clean_body(params),
        )

    def stream(self, session_id: str, thread_id: str, **params: Any) -> Iterator[dict[str, Any]]:
        return self._transport.stream(
            f"/sessions/{session_id}/threads/{thread_id}/events/stream",
            params=_clean_body(params) or None,
        )


class _FilesResource(_Resource):
    def create(self, **body: Any) -> dict[str, Any]:
        payload = _merge_body(body.pop("extra_body", None), body)
        if "file" in payload:
            file = payload.pop("file")
            return self.upload(file=file, **payload)
        return self._transport.create_file(payload)

    def upload(
        self,
        *,
        file: Any,
        name: str | None = None,
        content_type: str | None = None,
        scope: str = "session",
        metadata: dict[str, Any] | None = None,
        **body: Any,
    ) -> dict[str, Any]:
        inferred_name, content, inferred_type = _read_upload(file)
        payload = _merge_body(body.pop("extra_body", None), body)
        payload.update(
            {
                "name": name or inferred_name,
                "content_base64": base64.b64encode(content).decode("ascii"),
                "content_type": content_type or inferred_type,
                "scope": scope,
            }
        )
        if metadata is not None:
            payload["metadata"] = metadata
        return self._transport.create_file(payload)

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.list_files(**_clean_body(params))

    def retrieve_metadata(self, file_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.get_file(file_id)

    def download(self, file_id: str, **_: Any) -> bytes:
        return self._transport.download_file_content(file_id)

    def delete(self, file_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.delete_file(file_id)


class _SkillsResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.versions = _SkillVersionsResource(transport)

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", "/skills", json_body=_merge_body(body.pop("extra_body", None), body)
        )

    def retrieve(self, skill_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/skills/{skill_id}")

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.request("GET", "/skills", params=_clean_body(params) or None)

    def delete(self, skill_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("DELETE", f"/skills/{skill_id}")

    def archive(self, skill_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("POST", f"/skills/{skill_id}/archive", json_body={})


class _SkillVersionsResource(_Resource):
    def create(self, skill_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/skills/{skill_id}/versions",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def retrieve(self, skill_id: str, version_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/skills/{skill_id}/versions/{version_id}")

    def list(self, skill_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/skills/{skill_id}/versions",
            params=_clean_body(params) or None,
        )

    def delete(self, skill_id: str, version_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("DELETE", f"/skills/{skill_id}/versions/{version_id}")


class _VaultsResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.credentials = _VaultCredentialsResource(transport)

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", "/vaults", json_body=_merge_body(body.pop("extra_body", None), body)
        )

    def retrieve(self, vault_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/vaults/{vault_id}")

    def update(self, vault_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/vaults/{vault_id}",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.request("GET", "/vaults", params=_clean_body(params) or None)

    def delete(self, vault_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("DELETE", f"/vaults/{vault_id}")

    def archive(self, vault_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("POST", f"/vaults/{vault_id}/archive", json_body={})


class _VaultCredentialsResource(_Resource):
    def create(self, vault_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/vaults/{vault_id}/credentials",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def retrieve(self, vault_id: str, credential_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/vaults/{vault_id}/credentials/{credential_id}")

    def update(self, vault_id: str, credential_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/vaults/{vault_id}/credentials/{credential_id}",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def list(self, vault_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/vaults/{vault_id}/credentials",
            params=_clean_body(params) or None,
        )

    def delete(self, vault_id: str, credential_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("DELETE", f"/vaults/{vault_id}/credentials/{credential_id}")

    def archive(self, vault_id: str, credential_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/vaults/{vault_id}/credentials/{credential_id}/archive",
            json_body={},
        )

    def mcp_oauth_validate(self, vault_id: str, credential_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/vaults/{vault_id}/credentials/{credential_id}/mcp_oauth_validate",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )


class _MemoryStoresResource(_Resource):
    def __init__(self, transport: ManagedAgentsAnthropicClient) -> None:
        super().__init__(transport)
        self.memories = _MemoriesResource(transport)
        self.memory_versions = _MemoryVersionsResource(transport)

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", "/memory_stores", json_body=_merge_body(body.pop("extra_body", None), body)
        )

    def retrieve(self, memory_store_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/memory_stores/{memory_store_id}")

    def update(self, memory_store_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/memory_stores/{memory_store_id}",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.request("GET", "/memory_stores", params=_clean_body(params) or None)

    def delete(self, memory_store_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("DELETE", f"/memory_stores/{memory_store_id}")

    def archive(self, memory_store_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", f"/memory_stores/{memory_store_id}/archive", json_body={}
        )


class _MemoriesResource(_Resource):
    def create(self, memory_store_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/memory_stores/{memory_store_id}/items",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def retrieve(self, memory_store_id: str, memory_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/memory_stores/{memory_store_id}/items/{memory_id}")

    def update(self, memory_store_id: str, memory_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "PATCH",
            f"/memory_stores/{memory_store_id}/items/{memory_id}",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def list(self, memory_store_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/memory_stores/{memory_store_id}/items",
            params=_clean_body(params) or None,
        )

    def delete(self, memory_store_id: str, memory_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request(
            "DELETE", f"/memory_stores/{memory_store_id}/items/{memory_id}"
        )


class _MemoryVersionsResource(_Resource):
    def retrieve(self, memory_store_id: str, version_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/memory_stores/{memory_store_id}/memory_versions/{version_id}",
        )

    def list(self, memory_store_id: str, **params: Any) -> dict[str, Any]:
        return self._transport.request(
            "GET",
            f"/memory_stores/{memory_store_id}/memory_versions",
            params=_clean_body(params) or None,
        )


class _WebhooksResource(_Resource):
    def __init__(
        self,
        transport: ManagedAgentsAnthropicClient,
        *,
        webhook_key: str | bytes | None = None,
    ) -> None:
        super().__init__(transport)
        self._webhook_key = webhook_key

    def create(self, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST", "/webhooks", json_body=_merge_body(body.pop("extra_body", None), body)
        )

    def retrieve(self, webhook_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("GET", f"/webhooks/{webhook_id}")

    def update(self, webhook_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/webhooks/{webhook_id}",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def list(self, **params: Any) -> dict[str, Any]:
        return self._transport.request("GET", "/webhooks", params=_clean_body(params) or None)

    def archive(self, webhook_id: str, **_: Any) -> dict[str, Any]:
        return self._transport.request("POST", f"/webhooks/{webhook_id}/archive", json_body={})

    def test(self, webhook_id: str, **body: Any) -> dict[str, Any]:
        return self._transport.request(
            "POST",
            f"/webhooks/{webhook_id}/test",
            json_body=_merge_body(body.pop("extra_body", None), body),
        )

    def unwrap(
        self, payload: str, *, headers: dict[str, str], key: str | bytes | None = None
    ) -> dict[str, Any]:
        try:
            from standardwebhooks import Webhook
        except ImportError as exc:
            raise RuntimeError("Install standardwebhooks to use beta.webhooks.unwrap") from exc
        resolved_key = key if key is not None else self._webhook_key
        if resolved_key is None:
            raise ValueError("webhook key is required")
        Webhook(resolved_key).verify(payload, dict(headers))
        decoded = json.loads(payload)
        return dict(decoded) if isinstance(decoded, dict) else {"payload": decoded}


class _BetaResource(_Resource):
    def __init__(
        self,
        transport: ManagedAgentsAnthropicClient,
        *,
        webhook_key: str | bytes | None = None,
    ) -> None:
        super().__init__(transport)
        self.agents = _AgentsResource(transport)
        self.environments = _EnvironmentsResource(transport)
        self.sessions = _SessionsResource(transport)
        self.files = _FilesResource(transport)
        self.skills = _SkillsResource(transport)
        self.vaults = _VaultsResource(transport)
        self.memory_stores = _MemoryStoresResource(transport)
        self.webhooks = _WebhooksResource(transport, webhook_key=webhook_key)


class SynthManagedAgents:
    """Anthropic-shaped sync client for Synth Managed Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        anthropic_beta: str | None = None,
        webhook_key: str | bytes | None = None,
        transport: ManagedAgentsAnthropicClient | None = None,
    ) -> None:
        if transport is None:
            raise ValueError(
                "SynthManagedAgents has no implicit transport after retirement of the "
                "backend managed-agents proxy. Use from_horizons_private() with an "
                "explicit Horizons Private base URL and credential."
            )
        self._transport = transport
        self.webhook_key = webhook_key
        self.beta = _BetaResource(transport, webhook_key=webhook_key)

    @classmethod
    def from_horizons_private(
        cls,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        webhook_key: str | bytes | None = None,
    ) -> SynthManagedAgents:
        return cls(
            transport=ManagedAgentsAnthropicClient.from_horizons_private(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                anthropic_version=anthropic_version,
            ),
            webhook_key=webhook_key,
        )

    @classmethod
    def from_transport(cls, transport: ManagedAgentsAnthropicClient) -> SynthManagedAgents:
        return cls(transport=transport)

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._transport.request(method, path, **kwargs)

    def health(self) -> dict[str, Any]:
        return self._transport.health()

    def run_until_done(self, **kwargs: Any) -> Any:
        return self._transport.run_until_done(**kwargs)

    def download_session_files(self, **kwargs: Any) -> List[dict[str, str]]:
        return self._transport.download_session_files(**kwargs)


class _AsyncResource:
    def __init__(self, sync_resource: Any) -> None:
        self._sync = sync_resource

    @property
    def with_raw_response(self) -> _AsyncResource:
        return self

    @property
    def with_streaming_response(self) -> _AsyncResource:
        return self

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self._sync, name), *args, **kwargs)


class _AsyncAgentsResource(_AsyncResource):
    def __init__(self, sync_resource: _AgentsResource) -> None:
        super().__init__(sync_resource)
        self.versions = _AsyncAgentVersionsResource(sync_resource.versions)

    async def create(self, **body: Any) -> dict[str, Any]:
        return await self._call("create", **body)

    async def retrieve(self, agent_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", agent_id, **kwargs)

    async def update(self, agent_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("update", agent_id, **body)

    async def list(self, **params: Any) -> dict[str, Any]:
        return await self._call("list", **params)

    async def archive(self, agent_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("archive", agent_id, **kwargs)


class _AsyncAgentVersionsResource(_AsyncResource):
    async def list(self, agent_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("list", agent_id, **params)


class _AsyncEnvironmentsResource(_AsyncResource):
    async def create(self, **body: Any) -> dict[str, Any]:
        return await self._call("create", **body)

    async def retrieve(self, environment_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", environment_id, **kwargs)

    async def update(self, environment_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("update", environment_id, **body)

    async def list(self, **params: Any) -> dict[str, Any]:
        return await self._call("list", **params)

    async def delete(self, environment_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("delete", environment_id, **kwargs)

    async def archive(self, environment_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("archive", environment_id, **kwargs)


class _AsyncSessionsResource(_AsyncResource):
    def __init__(self, sync_resource: _SessionsResource) -> None:
        super().__init__(sync_resource)
        self.events = _AsyncSessionEventsResource(sync_resource.events)
        self.resources = _AsyncSessionResourcesResource(sync_resource.resources)
        self.threads = _AsyncSessionThreadsResource(sync_resource.threads)

    async def create(self, **body: Any) -> dict[str, Any]:
        return await self._call("create", **body)

    async def retrieve(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", session_id, **kwargs)

    async def update(self, session_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("update", session_id, **body)

    async def list(self, **params: Any) -> dict[str, Any]:
        return await self._call("list", **params)

    async def delete(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("delete", session_id, **kwargs)

    async def archive(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("archive", session_id, **kwargs)

    async def files(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("files", session_id, **params)

    async def attach_resource(self, session_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("attach_resource", session_id, **body)


class _AsyncSessionEventsResource(_AsyncResource):
    async def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("list", session_id, **params)

    async def send(
        self, session_id: str, *, events: List[dict[str, Any]] | None = None, **body: Any
    ) -> dict[str, Any]:
        return await self._call("send", session_id, events=events, **body)

    async def create(
        self, session_id: str, *, events: List[dict[str, Any]] | None = None, **body: Any
    ) -> dict[str, Any]:
        return await self._call("create", session_id, events=events, **body)

    async def stream(self, session_id: str, **params: Any) -> List[dict[str, Any]]:
        return await asyncio.to_thread(lambda: list(self._sync.stream(session_id, **params)))


class _AsyncSessionResourcesResource(_AsyncResource):
    async def create(self, session_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("create", session_id, **body)

    async def retrieve(self, session_id: str, resource_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", session_id, resource_id, **kwargs)

    async def update(self, session_id: str, resource_id: str, **body: Any) -> dict[str, Any]:
        return await self._call("update", session_id, resource_id, **body)

    async def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("list", session_id, **params)

    async def delete(self, session_id: str, resource_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("delete", session_id, resource_id, **kwargs)


class _AsyncSessionThreadsResource(_AsyncResource):
    def __init__(self, sync_resource: _SessionThreadsResource) -> None:
        super().__init__(sync_resource)
        self.events = _AsyncSessionThreadEventsResource(sync_resource.events)

    async def retrieve(self, session_id: str, thread_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", session_id, thread_id, **kwargs)

    async def list(self, session_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("list", session_id, **params)

    async def archive(self, session_id: str, thread_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("archive", session_id, thread_id, **kwargs)


class _AsyncSessionThreadEventsResource(_AsyncResource):
    async def list(self, session_id: str, thread_id: str, **params: Any) -> dict[str, Any]:
        return await self._call("list", session_id, thread_id, **params)

    async def stream(self, session_id: str, thread_id: str, **params: Any) -> List[dict[str, Any]]:
        return await asyncio.to_thread(
            lambda: list(self._sync.stream(session_id, thread_id, **params))
        )


class _AsyncFilesResource(_AsyncResource):
    async def create(self, **body: Any) -> dict[str, Any]:
        return await self._call("create", **body)

    async def upload(self, **body: Any) -> dict[str, Any]:
        return await self._call("upload", **body)

    async def list(self, **params: Any) -> dict[str, Any]:
        return await self._call("list", **params)

    async def retrieve_metadata(self, file_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve_metadata", file_id, **kwargs)

    async def download(self, file_id: str, **kwargs: Any) -> bytes:
        return await self._call("download", file_id, **kwargs)

    async def delete(self, file_id: str, **kwargs: Any) -> dict[str, Any]:
        return await self._call("delete", file_id, **kwargs)


class _AsyncSimpleResource(_AsyncResource):
    async def create(self, *args: Any, **body: Any) -> dict[str, Any]:
        return await self._call("create", *args, **body)

    async def retrieve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._call("retrieve", *args, **kwargs)

    async def update(self, *args: Any, **body: Any) -> dict[str, Any]:
        return await self._call("update", *args, **body)

    async def list(self, *args: Any, **params: Any) -> dict[str, Any]:
        return await self._call("list", *args, **params)

    async def delete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._call("delete", *args, **kwargs)

    async def archive(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._call("archive", *args, **kwargs)

    async def test(self, *args: Any, **body: Any) -> dict[str, Any]:
        return await self._call("test", *args, **body)

    async def mcp_oauth_validate(self, *args: Any, **body: Any) -> dict[str, Any]:
        return await self._call("mcp_oauth_validate", *args, **body)

    async def unwrap(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await self._call("unwrap", *args, **kwargs)


class _AsyncSkillsResource(_AsyncSimpleResource):
    def __init__(self, sync_resource: _SkillsResource) -> None:
        super().__init__(sync_resource)
        self.versions = _AsyncSimpleResource(sync_resource.versions)


class _AsyncVaultsResource(_AsyncSimpleResource):
    def __init__(self, sync_resource: _VaultsResource) -> None:
        super().__init__(sync_resource)
        self.credentials = _AsyncSimpleResource(sync_resource.credentials)


class _AsyncMemoryStoresResource(_AsyncSimpleResource):
    def __init__(self, sync_resource: _MemoryStoresResource) -> None:
        super().__init__(sync_resource)
        self.memories = _AsyncSimpleResource(sync_resource.memories)
        self.memory_versions = _AsyncSimpleResource(sync_resource.memory_versions)


class _AsyncBetaResource(_AsyncResource):
    def __init__(self, sync_resource: _BetaResource) -> None:
        super().__init__(sync_resource)
        self.agents = _AsyncAgentsResource(sync_resource.agents)
        self.environments = _AsyncEnvironmentsResource(sync_resource.environments)
        self.sessions = _AsyncSessionsResource(sync_resource.sessions)
        self.files = _AsyncFilesResource(sync_resource.files)
        self.skills = _AsyncSkillsResource(sync_resource.skills)
        self.vaults = _AsyncVaultsResource(sync_resource.vaults)
        self.memory_stores = _AsyncMemoryStoresResource(sync_resource.memory_stores)
        self.webhooks = _AsyncSimpleResource(sync_resource.webhooks)


class AsyncSynthManagedAgents:
    """Anthropic-shaped async client for Synth Managed Agents."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        anthropic_beta: str | None = None,
        webhook_key: str | bytes | None = None,
        sync_client: SynthManagedAgents | None = None,
    ) -> None:
        if sync_client is None:
            raise ValueError(
                "AsyncSynthManagedAgents has no implicit transport after retirement of "
                "the backend managed-agents proxy. Use from_horizons_private() with an "
                "explicit Horizons Private base URL and credential."
            )
        self._sync = sync_client
        self.beta = _AsyncBetaResource(sync_client.beta)

    @classmethod
    def from_horizons_private(
        cls,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        anthropic_version: str | None = None,
        webhook_key: str | bytes | None = None,
    ) -> AsyncSynthManagedAgents:
        return cls(
            sync_client=SynthManagedAgents.from_horizons_private(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                anthropic_version=anthropic_version,
                webhook_key=webhook_key,
            )
        )

    @classmethod
    def from_transport(cls, transport: ManagedAgentsAnthropicClient) -> AsyncSynthManagedAgents:
        return cls(sync_client=SynthManagedAgents.from_transport(transport))

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.request, method, path, **kwargs)

    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.health)

    async def run_until_done(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.run_until_done, **kwargs)

    async def download_session_files(self, **kwargs: Any) -> List[dict[str, str]]:
        return await asyncio.to_thread(self._sync.download_session_files, **kwargs)


__all__ = [
    "AsyncSynthManagedAgents",
    "SynthManagedAgents",
]
