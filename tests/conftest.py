from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from tests.pbt.config import SdkPbtConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("sdk-pbt", "Hypothesis stateful SDK PBT options")
    group.addoption(
        "--base-url",
        action="store",
        default=os.environ.get("SYNTH_PBT_BASE_URL"),
        help="Optional live local-slot core-api base URL (default: in-process MockTransport).",
    )
    group.addoption(
        "--api-key",
        action="store",
        default=os.environ.get("SYNTH_PBT_API_KEY"),
        help="Optional Bearer API key for --base-url live local-slot mode.",
    )
    group.addoption(
        "--trace-out",
        action="store",
        default=os.environ.get("SYNTH_PBT_TRACE_OUT"),
        help="Optional directory path for JSONL state-transition traces.",
    )
    group.addoption(
        "--pbt-examples",
        action="store",
        type=int,
        default=int(os.environ.get("SYNTH_PBT_MAX_EXAMPLES", "50")),
        help="Hypothesis max_examples per stateful PBT machine (default: 50).",
    )
    group.addoption(
        "--pbt-steps",
        action="store",
        type=int,
        default=int(os.environ.get("SYNTH_PBT_STEP_COUNT", "20")),
        help="Hypothesis stateful_step_count per stateful PBT machine (default: 20).",
    )
    group.addoption(
        "--everystate-spec-dir",
        action="store",
        default=os.environ.get("SYNTH_PBT_SPEC_DIR"),
        help="Optional override directory containing RunControl/HostedOptions/BillingWallet .tla specs.",
    )


@pytest.fixture(scope="session")
def contract():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "contracts/synth-v1.openapi.yaml").read_text()
    )


@pytest.fixture(scope="session")
def sdk_pbt_config(request: pytest.FixtureRequest) -> SdkPbtConfig:
    trace_raw = request.config.getoption("--trace-out")
    spec_raw = request.config.getoption("--everystate-spec-dir")
    return SdkPbtConfig(
        base_url=request.config.getoption("--base-url"),
        api_key=request.config.getoption("--api-key"),
        trace_out=Path(str(trace_raw)) if trace_raw else None,
        max_examples=int(request.config.getoption("--pbt-examples")),
        stateful_step_count=int(request.config.getoption("--pbt-steps")),
        spec_dir=Path(str(spec_raw)) if spec_raw else None,
    )
