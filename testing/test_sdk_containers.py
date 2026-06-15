from __future__ import annotations

from synth_ai import SynthClient
from synth_ai.sdk.containers import ContainersClient, ContainerSpec, ContainerType


def test_containers_client_crud_against_local_backend(
    backend_url: str,
    api_key: str,
) -> None:
    client = ContainersClient(api_key=api_key, backend_base=backend_url)
    created = client.create(
        ContainerSpec(
            name="harbor-eval-container",
            task_type=ContainerType.harbor_code,
            definition={"repo": "https://github.com/synth-labs/example"},
        )
    )

    assert created.id.startswith("container-")
    assert created.status == "ready"
    assert created.internal_url

    fetched = client.get(created.id)
    listed = client.list()
    waited = client.wait_ready(created.id, timeout=1.0, poll_interval=0.01)

    assert fetched.id == created.id
    assert [item.id for item in listed] == [created.id]
    assert waited.status == "ready"

    client.delete(created.id)
    assert client.list() == []


def test_synth_client_exposes_rewritten_container_surface(
    backend_url: str,
    api_key: str,
) -> None:
    client = SynthClient(api_key=api_key, base_url=backend_url)
    created = client.containers.create(
        ContainerSpec(
            name="composed-container",
            task_type=ContainerType.harbor_browser,
            definition={"kind": "browser"},
        )
    )

    assert created.name == "composed-container"
