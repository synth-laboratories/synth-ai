from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def contract():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "contracts/synth-v1.openapi.yaml").read_text()
    )
