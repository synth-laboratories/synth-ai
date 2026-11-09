import httpx
import pytest

from synth_ai._limits import ERROR_RESPONSE_MAX_BYTES
from synth_ai._transport import Transport
from synth_ai.errors import ResponseContractError


class EndlessError(httpx.AsyncByteStream):
    def __init__(self):
        self.chunks = 0
        self.closed = False

    async def __aiter__(self):
        while True:
            self.chunks += 1
            yield b"x" * 8192

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("operation", ["events", "download"])
@pytest.mark.parametrize("encoding", ["identity", "gzip"])
async def test_stream_error_is_bounded_and_connection_closed(tmp_path, operation, encoding):
    source = EndlessError()

    def refuse(request):
        return httpx.Response(
            503,
            stream=source,
            headers={"content-encoding": encoding, "x-request-id": "error-7"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        transport = Transport("http://localhost", "token", client=client)
        with pytest.raises(ResponseContractError) as caught:
            if operation == "events":
                _ = [event async for event in transport.stream_events("run")]
            else:
                await transport.download_output(
                    "run",
                    "output",
                    tmp_path / "output",
                    revision_id="revision",
                    digest_sha256="a" * 64,
                    max_bytes=1024,
                )
        assert caught.value.request_id == "error-7"
    assert source.closed
    assert source.chunks == (ERROR_RESPONSE_MAX_BYTES // 8192 + 1 if encoding == "identity" else 0)
    assert not (tmp_path / "output").exists()
