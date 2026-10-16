"""Offline SSE transport stand-in shared by laws and controls.

See https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
Only in-process httpx.MockTransport is used; the socket guard remains active.
"""

import httpx
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport


async def collect_events(mode, body, content_type="text/event-stream"):
    headers = {} if content_type is None else {"content-type": content_type}

    def response(request):
        return httpx.Response(200, content=body, headers=headers)

    if mode == "sync":
        transport = HttpTransport("http://offline.invalid", {})
        transport.client.close()
        transport.client = httpx.Client(
            base_url="http://offline.invalid", transport=httpx.MockTransport(response)
        )
        try:
            return list(transport.stream_sse("/events", operation_id="offline_events"))
        finally:
            transport.close()
    transport = AsyncHttpTransport("http://offline.invalid", {})
    await transport.client.aclose()
    transport.client = httpx.AsyncClient(
        base_url="http://offline.invalid", transport=httpx.MockTransport(response)
    )
    try:
        return [
            event async for event in transport.stream_sse("/events", operation_id="offline_events")
        ]
    finally:
        await transport.close()
