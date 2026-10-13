"""Owned cancellable synchronous exchange. See backend SDK index_contract.md."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from synth_ai.contracts.operations import Operation
from synth_ai.errors import (
    ContractError,
    OutcomeUncertainError,
    RequestTimeoutError,
    SynthError,
    TransportError,
)
from synth_ai.options import ClientOptions, ReadOptions
from synth_ai.types.base import ClientTiming, ResponseModel

from .blocking import DeadlineTransport
from .cleanup import handle_cleanup_failure
from .lifetime import Closeable
from .request import build_request, encode_body, mutation_identity
from .response import decode_response, retry_delay

T = TypeVar("T", bound=ResponseModel)


class HttpTransport:
    def __init__(self, base_url: str, api_key: str | None, options: ClientOptions) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._options = options
        self.transfer_hosts = options.upload_hosts or (urlsplit(base_url).hostname or "",)
        self._owned = True
        self.client = httpx.Client(
            trust_env=False, transport=DeadlineTransport(options.maximum_connections)
        )
        self._closed = False
        self._streams: set[Closeable] = set()
        self._responses: set[httpx.Response] = set()

    def _adopt_response(self, response: httpx.Response) -> httpx.Response:
        if self._closed or len(self._responses) >= self._options.maximum_connections:
            response.close()
            raise ContractError("SDK client closed or active response limit reached")
        self._responses.add(response)
        return response

    def _release_response(self, response: httpx.Response) -> None:
        try:
            response.close()
        finally:
            self._responses.discard(response)

    @property
    def request_timeout_seconds(self) -> float:
        return self._options.timeout_seconds

    @property
    def maximum_response_bytes(self) -> int:
        return self._options.maximum_response_bytes

    def _register_stream(self, stream: Closeable) -> None:
        if self._closed:
            raise RuntimeError("SDK client is closed")
        if len(self._streams) >= self._options.maximum_connections:
            raise RuntimeError("SDK active stream limit reached")
        self._streams.add(stream)

    def _unregister_stream(self, stream: Closeable) -> None:
        self._streams.discard(stream)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for stream in tuple(self._streams):
            try:
                stream.close()
            except Exception as error:
                errors.append(error)
        for response in tuple(self._responses):
            try:
                self._release_response(response)
            except Exception as error:
                errors.append(error)
        if self._owned:
            try:
                self.client.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("SDK response cleanup failed", errors)

    def _open_stream(
        self,
        operation: Operation,
        *,
        path: str,
        query: Mapping[str, str | int],
        remaining_seconds: float,
        digest: str,
    ) -> httpx.Response:
        """Open one SDK-owned read response, without retries or redirects."""
        if self._closed:
            raise RuntimeError("SDK client is closed")
        if operation.mutation or operation.method != "GET":
            raise ValueError("byte streams require an observational GET operation")
        request = build_request(
            self.client,
            self._base_url,
            self._api_key,
            operation,
            path,
            b"",
            query,
            ReadOptions(timeout_seconds=remaining_seconds, read_retry_count=0),
            None,
            remaining_seconds,
            self._options.connect_timeout_seconds,
        )
        request.headers["Accept"] = "application/octet-stream"
        request.headers["If-Match"] = '"' + digest + '"'
        response = self.client.send(request, stream=True, auth=None, follow_redirects=False)
        return self._adopt_response(response)

    def request(
        self,
        operation: Operation,
        model: type[T],
        *,
        body: BaseModel | None = None,
        path: str | None = None,
        query: Mapping[str, str | int] | None = None,
        options: ReadOptions | None = None,
        validate: Callable[[T], T] | None = None,
    ) -> T:
        if self._closed:
            raise RuntimeError("SDK client is closed")
        payload = encode_body(body)
        identity = mutation_identity(operation, path, payload, options)
        started = time.monotonic()
        timeout = (
            options.timeout_seconds
            if options is not None and options.timeout_seconds is not None
            else self._options.timeout_seconds
        )
        deadline = started + timeout
        retry_count = (
            options.read_retry_count
            if options is not None and options.read_retry_count is not None
            else self._options.read_retry_count
        )
        attempts = 1 if operation.mutation else retry_count + 1
        for attempt in range(1, attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Request deadline expired", identity=identity)
            request = build_request(
                self.client,
                self._base_url,
                self._api_key,
                operation,
                path,
                payload,
                query,
                options,
                identity,
                remaining,
                self._options.connect_timeout_seconds,
            )
            response: httpx.Response | None = None
            try:
                response = self.client.send(request, stream=True, auth=None, follow_redirects=False)
                self._adopt_response(response)
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ContractError(
                        "Compressed responses are outside the bounded wire contract"
                    )
                content = bytearray()
                first_byte: float | None = None
                for chunk in response.iter_bytes():
                    if self._closed:
                        raise ContractError("SDK client closed during response")
                    now = time.monotonic()
                    if first_byte is None:
                        first_byte = now
                    if len(content) + len(chunk) > self._options.maximum_response_bytes:
                        raise ContractError("Response exceeded the configured byte bound")
                    content.extend(chunk)
                    if now >= deadline:
                        raise RequestTimeoutError("Response deadline expired")
                finished = time.monotonic()
                if finished >= deadline:
                    raise RequestTimeoutError("Response deadline expired")
                if (
                    not operation.mutation
                    and response.headers.get("X-Synth-Contract") == operation.contract
                    and response.status_code in {429, 502, 503, 504}
                    and attempt < attempts
                ):
                    delay = retry_delay(response, self._options.retry_delay_seconds)
                    self._release_response(response)
                    if delay >= deadline - time.monotonic():
                        raise RequestTimeoutError("Retry would exceed the request deadline")
                    time.sleep(delay)
                    continue
                timing = ClientTiming(
                    total_ms=(finished - started) * 1000,
                    first_byte_ms=((first_byte or finished) - started) * 1000,
                    attempts=attempt,
                )
                decoded = decode_response(
                    response,
                    bytes(content),
                    model,
                    timing,
                    identity,
                    operation.mutation,
                    operation.contract,
                )
                return validate(decoded) if validate is not None else decoded
            except (httpx.TransportError, RequestTimeoutError, ContractError) as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation delivery is uncertain; retain its identity", identity=identity
                    ) from cause
                if isinstance(cause, httpx.TransportError) and attempt < attempts:
                    delay = self._options.retry_delay_seconds
                    if delay >= deadline - time.monotonic():
                        raise RequestTimeoutError(
                            "Retry would exceed the request deadline"
                        ) from cause
                    time.sleep(delay)
                    continue
                if isinstance(cause, httpx.TimeoutException):
                    raise RequestTimeoutError("HTTP request timed out") from cause
                if isinstance(cause, httpx.TransportError):
                    raise TransportError("HTTP exchange failed") from cause
                raise
            except SynthError:
                raise
            except Exception as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation exchange failed unexpectedly; reconcile its identity",
                        identity=identity,
                    ) from cause
                raise TransportError("HTTP exchange failed unexpectedly") from cause
            finally:
                if response is not None:
                    primary = sys.exception()
                    try:
                        self._release_response(response)
                    except Exception as cause:
                        handle_cleanup_failure(
                            cause, primary, mutation=operation.mutation, identity=identity
                        )
        raise RuntimeError("Bounded request loop exhausted without an outcome")
