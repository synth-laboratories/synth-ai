from synth_ai.sdk.research.contracts.local_execution_profile import (
    LOCAL_EVAL_CONTRACT_SCHEMA_VERSION,
    LocalEvalContract,
    local_execution_payload,
)


def test_local_execution_payload_binds_the_manager_slot() -> None:
    contract = LocalEvalContract(
        schema_version=LOCAL_EVAL_CONTRACT_SCHEMA_VERSION,
        runtime_id="local_docker",
        worker_pool_id="slot5",
        launch_target="local-dockerized",
        requires_hosted_capacity=False,
        product_source_mirrors={},
        task_env={},
    )

    assert local_execution_payload(contract) == {
        "slot_id": "slot5",
        "runtime_id": "local_docker",
        "dispatch_pool": "slot5",
        "host_kind": "docker",
        "requires_hosted_capacity": False,
    }
