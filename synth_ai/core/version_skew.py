"""Safe version-skew diagnostics shared by negotiation and response decoding."""

from synth_ai.errors import ContractError


def schema_error(
    *,
    contract: str,
    sdk_schema: str,
    backend_schema: str | None,
    backend_release: str | None,
    minimum_sdk_version: str | None,
    stage: str,
) -> ContractError:
    from synth_ai import __version__

    target = f">={minimum_sdk_version}" if minimum_sdk_version else "the latest version"
    return ContractError(
        f"synth-ai {__version__} ({contract} schema {sdk_schema[:12]}) cannot read this backend "
        f"(release {backend_release or 'unknown'}, schema {(backend_schema or 'unknown')[:12]}). "
        f"Upgrade synth-ai to {target}: pip install -U synth-ai",
        code="schema_mismatch",
        stage=stage,
        cause=f"contract={contract} sdk_schema_sha256={sdk_schema} "
        f"backend_schema_sha256={backend_schema or 'unknown'}",
    )
