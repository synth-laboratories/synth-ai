"""Upload MIME validation follows Core's bounded binary-capable contract."""

import pytest

from synth_ai.errors import UploadFormatUnsupportedError, check_upload_format


@pytest.mark.parametrize("media_type", ["application/octet-stream", "application/pdf", "image/png"])
def test_binary_media_types_are_admitted(media_type):
    assert check_upload_format(media_type) == media_type


@pytest.mark.parametrize(
    "media_type",
    [None, "", "text", "/plain", "text/", "text/pla in", "text/é", "text/" + "a" * 251],
)
def test_invalid_media_types_are_refused(media_type):
    with pytest.raises(UploadFormatUnsupportedError):
        check_upload_format(media_type)
