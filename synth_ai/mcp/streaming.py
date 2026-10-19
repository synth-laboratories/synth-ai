"""Select the stateless streaming transport only for explicit live tools."""

import json

from starlette.responses import JSONResponse


class StreamingToolSelector:
    def __init__(self, ordinary, streaming, tools):
        self.ordinary, self.streaming, self.tools = ordinary, streaming, tools

    async def __call__(self, scope, receive, send):
        if scope.get("method") != "POST":
            await self.ordinary(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            if len(body) + len(message.get("body", b"")) > 128 * 1024:
                await JSONResponse({"error": "request_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            value = json.loads(body)
            selected = (
                isinstance(value, dict)
                and value.get("method") == "tools/call"
                and isinstance(value.get("params"), dict)
                and value["params"].get("name") in self.tools
            )
        except (ValueError, TypeError):
            selected = False
        if selected and b"last-event-id" in dict(scope.get("headers", ())):
            await JSONResponse({"error": "strict_events_cannot_resume"}, status_code=409)(
                scope, receive, send
            )
            return
        received = False

        async def replay():
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await (self.streaming if selected else self.ordinary)(scope, replay, send)
