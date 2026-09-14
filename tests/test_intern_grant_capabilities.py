from synth_ai.sdk.research.contracts.intern_grants import InternCapabilityOperation


def test_pool_deployment_capabilities_are_publicly_mirrorable() -> None:
    assert {
        InternCapabilityOperation.POOL_DEPLOYMENT_CREATE.value,
        InternCapabilityOperation.POOL_DEPLOYMENT_UPDATE.value,
        InternCapabilityOperation.POOL_DEPLOYMENT_DELETE.value,
    } == {
        "pool.deployment_create",
        "pool.deployment_update",
        "pool.deployment_delete",
    }
