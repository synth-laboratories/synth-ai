"""Local transfer and recovery bounds; admission limits come from the contract."""

from ._generated.file_limits import FILE_MAX_BYTES, MEDIA_TYPE_MAX_BYTES

UPLOAD_MAX_BYTES = FILE_MAX_BYTES
UPLOAD_MEDIA_TYPE_MAX_BYTES = MEDIA_TYPE_MAX_BYTES
UPLOAD_BINARY_MEDIA_TYPE = "application/octet-stream"
SSE_FRAME_MAX_BYTES = 1 << 20
ERROR_RESPONSE_MAX_BYTES = 64 << 10
CHECKPOINT_MAX_EVENTS = 4096
CHECKPOINT_MAX_ID_LENGTH = 1024
