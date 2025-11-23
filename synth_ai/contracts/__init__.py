"""
Synth AI Contracts

OpenAPI contracts for implementing Task Apps in any language.
These contracts define the HTTP interface between Synth optimizers
(MIPRO, GEPA) and your Task App service.

Usage:
    # Get the contract file path
    from synth_ai.contracts import TASK_APP_CONTRACT_PATH

    # Get the contract as a string
    from synth_ai.contracts import get_task_app_contract
    yaml_content = get_task_app_contract()

    # Or access directly via CLI
    # synth contracts show task-app
"""

from pathlib import Path

CONTRACTS_DIR = Path(__file__).parent
TASK_APP_CONTRACT_PATH = CONTRACTS_DIR / "task_app.yaml"


def get_task_app_contract() -> str:
    """Return the Task App contract as a YAML string.

    This OpenAPI spec defines the HTTP interface that Task Apps must implement
    to work with Synth's MIPRO and GEPA prompt optimizers.

    Returns:
        The full OpenAPI 3.1 specification as a YAML string.

    Example:
        >>> contract = get_task_app_contract()
        >>> print(contract[:50])
        openapi: 3.1.0
        info:
          title: Synth Task App
    """
    return TASK_APP_CONTRACT_PATH.read_text()


def get_contract_path(contract_name: str = "task-app") -> Path:
    """Get the filesystem path to a contract file.

    Args:
        contract_name: Name of the contract. Currently only "task-app" is supported.

    Returns:
        Path to the contract YAML file.

    Raises:
        ValueError: If the contract name is not recognized.
    """
    if contract_name in ("task-app", "task_app"):
        return TASK_APP_CONTRACT_PATH
    raise ValueError(f"Unknown contract: {contract_name}. Available: task-app")


__all__ = [
    "CONTRACTS_DIR",
    "TASK_APP_CONTRACT_PATH",
    "get_task_app_contract",
    "get_contract_path",
]
