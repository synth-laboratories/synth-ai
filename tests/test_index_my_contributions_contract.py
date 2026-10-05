"""MyContributions must decode the backend's current row shape (backend 75091f04a)."""

import pytest
from pydantic import ValidationError

from synth_ai.sdk.index.lifecycle import MyContributions

ROW = {
    "reference": {
        "contribution_id": "1e555888-f2de-4978-b23f-c2c326daaa33",
        "revision_id": "d8fe1839-1af6-4a56-aff9-ff7c0bf2edb4",
    },
    "contribution_id": "1e555888-f2de-4978-b23f-c2c326daaa33",
    "title": "Scientific revision provenance integrity audit",
    "status": "submitted",
    "audience": "public",
    "current_revision_id": None,
    "generation": 0,
    "updated_at": "2026-10-05T16:50:00Z",
}


def test_backend_publication_fields_decode():
    page = MyContributions.model_validate(
        {
            "items": [
                {
                    **ROW,
                    "publication_mode": "private",
                    "public_revision_id": None,
                    "access": "owned",
                    "abstract": "Historical eight-check audit.",
                }
            ]
        }
    )
    assert page.items[0].publication_mode == "private"
    assert page.items[0].access == "owned"


def test_older_rows_without_publication_fields_still_decode():
    item = MyContributions.model_validate({"items": [ROW]}).items[0]
    assert item.publication_mode is None and item.access == "owned"


def test_unknown_publication_mode_still_refuses():
    with pytest.raises(ValidationError):
        MyContributions.model_validate({"items": [{**ROW, "publication_mode": "leaked"}]})


def test_next_cursor_decodes():
    page = MyContributions.model_validate({"items": [ROW], "next_cursor": "abc123"})
    assert page.next_cursor == "abc123"
    assert MyContributions.model_validate({"items": [], "next_cursor": None}).next_cursor is None
