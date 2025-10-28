
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.skip(reason="Crafter tracing tests skipped per request")
async def test_crafter_text_prompt_tracing():
    assert True


@pytest.mark.asyncio
@pytest.mark.skip(reason="Crafter tracing tests skipped per request")
async def test_crafter_image_prompt_tracing():
    assert True
