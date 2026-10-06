"""SYN-3937 generated Forge consumer retains exact owning actor IDs."""
import pytest
from pydantic import ValidationError
from synth_ai.sdk.research.contracts.forge.contracts import ExactReference, Scope


def test_generated_reference_preserves_native_execution_bytes():
    identifier = 'intern-exec:' + 'a' * 64
    reference = ExactReference(authority='mloky',kind='mloky_execution',record_id=identifier,
                               revision='1',digest_sha256='b'*64)
    assert reference.record_id == identifier
    with pytest.raises(ValidationError):
        reference.model_validate({**reference.model_dump(),'authority':'human'})
    with pytest.raises(ValidationError):
        Scope(organization_id=identifier,project_id='project')
