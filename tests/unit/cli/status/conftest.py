from collections.abc import Iterator

import pytest


@pytest.fixture(scope="module")
def status_modules() -> Iterator[dict[str, object]]:
    import synth_ai.cli.status as status_module

    yield {
        "config": status_module,
        "client": status_module,
        "errors": status_module,
    }
