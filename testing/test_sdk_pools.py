from __future__ import annotations

from synth_ai.sdk.pools import ContainerPoolsClient


def test_pools_client_end_to_end_against_local_backend(
    backend_url: str,
    api_key: str,
) -> None:
    client = ContainerPoolsClient(api_key=api_key, backend_base=backend_url)

    pool = client.create({"name": "harbor-pool", "target": "harbor", "runtime": "tblite"})
    pool_id = pool["id"]
    assert pool["status"] == "ready"

    listed = client.list(limit=10)
    fetched = client.get(pool_id)
    updated = client.update(pool_id, {"status": "warming", "owner": "testing"})
    urls = client.get_urls(pool_id)
    metrics = client.metrics.get(pool_id)

    assert [item["id"] for item in listed["items"]] == [pool_id]
    assert fetched["id"] == pool_id
    assert updated["status"] == "warming"
    assert urls["public_url"].endswith(".harbor.synth.ai")
    assert metrics["success_rate"] == 1.0

    task = client.tasks.create(
        pool_id,
        {"name": "gepa-eval-task", "container_ref": "container-1", "mode": "gepa"},
    )
    task_id = task["id"]
    task_list = client.tasks.list(pool_id)
    replaced_task = client.tasks.update(
        pool_id,
        task_id,
        {"name": "gepa-eval-task", "container_ref": "container-1", "mode": "gepa"},
    )
    patched_task = client.tasks.patch(pool_id, task_id, {"status": "active"})

    assert [item["id"] for item in task_list["items"]] == [task_id]
    assert replaced_task["mode"] == "gepa"
    assert patched_task["status"] == "active"

    pool_health = client.get_pool_container_health(pool_id)
    pool_info = client.get_pool_container_info(pool_id)
    pool_metadata = client.get_pool_container_metadata(pool_id)
    pool_rollout_exec = client.execute_pool_container_rollout(pool_id, {"input": "hello"})
    pool_eval = client.prompt_learning_evaluate_pool(pool_id, {"algorithm": "gepa"})

    assert pool_health["status"] == "healthy"
    assert pool_info["kind"] == "harbor_code"
    assert pool_metadata["metadata"]["owner"] == "testing"
    assert pool_rollout_exec["accepted"] is True
    assert pool_eval["request"]["algorithm"] == "gepa"

    task_health = client.get_task_container_health(pool_id, task_id)
    task_info = client.get_task_container_info(pool_id, task_id)
    task_metadata = client.get_task_container_metadata(pool_id, task_id)
    task_rollout_exec = client.execute_task_container_rollout(
        pool_id,
        task_id,
        {"input": "hello"},
    )
    task_eval = client.prompt_learning_evaluate_task(
        pool_id,
        task_id,
        {"algorithm": "gepa"},
    )

    assert task_health["status"] == "healthy"
    assert task_info["kind"] == "harbor_code"
    assert task_metadata["metadata"]["owner"] == "testing"
    assert task_rollout_exec["accepted"] is True
    assert task_eval["request"]["algorithm"] == "gepa"

    rollout_request = {"task_id": task_id, "mode": "eval", "messages": []}
    pool_rollout = client.rollouts.create(pool_id, rollout_request)
    rollout_id = pool_rollout["id"]

    pool_rollout_list = client.rollouts.list(pool_id)
    pool_rollout_get = client.rollouts.get(pool_id, rollout_id)
    pool_rollout_artifacts = client.rollouts.artifacts(pool_id, rollout_id)
    pool_rollout_usage = client.rollouts.usage(pool_id, rollout_id)
    pool_rollout_summary = client.rollouts.summary(pool_id, rollout_id)
    pool_rollout_events = list(client.rollouts.events(pool_id, rollout_id))
    cancelled_pool_rollout = client.rollouts.cancel(pool_id, rollout_id)

    assert [item["id"] for item in pool_rollout_list["items"]] == [rollout_id]
    assert pool_rollout_get["request"]["task_id"] == task_id
    assert pool_rollout_artifacts["items"][0]["kind"] == "report"
    assert pool_rollout_usage["tokens"] == 42
    assert pool_rollout_summary["score"] == 1.0
    assert [event["event"] for event in pool_rollout_events] == [
        "rollout.started",
        "rollout.completed",
    ]
    assert cancelled_pool_rollout["state"] == "cancelled"

    global_rollout = client.agent_rollouts.create({"pool_id": pool_id, "mode": "eval"})
    global_rollout_id = global_rollout["id"]

    global_rollout_list = client.agent_rollouts.list()
    global_rollout_get = client.agent_rollouts.get(global_rollout_id)
    global_rollout_artifacts = client.agent_rollouts.artifacts(global_rollout_id)
    global_rollout_usage = client.agent_rollouts.usage(global_rollout_id)
    global_rollout_summary = client.agent_rollouts.summary(global_rollout_id)
    global_rollout_events = list(client.agent_rollouts.events(global_rollout_id))
    cancelled_global_rollout = client.agent_rollouts.cancel(global_rollout_id)

    assert [item["id"] for item in global_rollout_list["items"]] == [global_rollout_id]
    assert global_rollout_get["request"]["pool_id"] == pool_id
    assert global_rollout_artifacts["items"][0]["artifact_id"].endswith(global_rollout_id)
    assert global_rollout_usage["tokens"] == 24
    assert global_rollout_summary["score"] == 1.0
    assert [event["event"] for event in global_rollout_events] == [
        "rollout.started",
        "rollout.completed",
    ]
    assert cancelled_global_rollout["state"] == "cancelled"

    queue_status = client.get_queue_status()
    capabilities = client.get_capabilities()
    deleted_task = client.tasks.delete(pool_id, task_id)
    deleted_pool = client.delete(pool_id)

    assert queue_status["running"] == 1
    assert "synthtunnel" in capabilities["tunnels"]
    assert deleted_task["status"] == "deleted"
    assert deleted_pool["status"] == "deleted"
