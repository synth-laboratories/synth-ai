import pytest

from synth_ai._limits import SSE_FRAME_MAX_BYTES
from synth_ai._stream import Frame, bounded_lines, frames


@pytest.mark.parametrize("delimiter", [b"\n", b"\r\n", b"\r"])
async def test_delimiters_and_unicode_split_at_every_byte(delimiter):
    wire = b"\xef\xbb\xbfid: 7" + delimiter + "data: café 🌱".encode() + delimiter * 2

    async def chunks():
        for value in wire:
            yield bytes([value])

    assert [frame async for frame in frames(bounded_lines(chunks()))] == [
        Frame("message", "café 🌱", "7")
    ]


async def test_unterminated_line_is_refused_before_consuming_unbounded_source():
    consumed = 0

    async def chunks():
        nonlocal consumed
        for _ in range(1024):
            consumed += 1
            yield b"x" * 16384

    with pytest.raises(ValueError, match="line exceeds"):
        _ = [frame async for frame in frames(bounded_lines(chunks()))]
    assert consumed == SSE_FRAME_MAX_BYTES // 16384 + 1


async def test_many_short_lines_cannot_bypass_aggregate_frame_bound():
    async def chunks():
        for _ in range(SSE_FRAME_MAX_BYTES):
            yield b": comment\n"

    with pytest.raises(ValueError, match="frame exceeds"):
        _ = [frame async for frame in frames(bounded_lines(chunks()))]


@pytest.mark.parametrize("suffix", [b"data: partial", b"data: partial\n"])
async def test_truncated_frame_never_dispatches_or_advances_identity(suffix):
    async def chunks():
        yield b"id: 8\n" + suffix

    assert [frame async for frame in frames(bounded_lines(chunks()))] == []


async def test_invalid_utf8_is_protocol_failure():
    async def chunks():
        yield b"data: \xff\n\n"

    with pytest.raises(UnicodeDecodeError):
        _ = [frame async for frame in frames(bounded_lines(chunks()))]
