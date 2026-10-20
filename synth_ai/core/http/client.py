"""Owned cancellable synchronous exchange. See backend SDK index_contract.md."""

from __future__ import annotations

import random
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from synth_ai.contracts.operations import CAPABILITIES, Operation
from synth_ai.core.capabilities import require_operation, validate_release
from synth_ai.errors import (
    AuthenticationError,
    ContractError,
    OutcomeUncertainError,
    RequestTimeoutError,
    SynthError,
    TransportError,
)
from synth_ai.options import ClientOptions, ReadOptions
from synth_ai.types.base import ClientTiming
from synth_ai.types.capabilities import Capabilities, NegotiatedCapability

from .backoff import delay_seconds, retry_after_seconds, retry_failure
from .blocking import DeadlineTransport
from .cleanup import handle_cleanup_failure
from .credentials import CredentialProvider
from .identity import annotate_request_error, request_options
from .lifetime import Closeable
from .request import build_request, encode_body, mutation_identity
from .response import (
    annotate_response_error,
    classify_exchange,
    decode_response,
    validate_negotiated_response,
)

T = TypeVar("T", bound=BaseModel)


class HttpTransport:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        options: ClientOptions,
        *,
        credential_provider: CredentialProvider | None = None,
        _sleep: Callable[[float], None] | None = None,
        _random: random.Random | None = None,
    ) -> None:
        if options.local_development_retention and urlsplit(base_url).hostname not in {
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Local development retention requires a loopback API address")
        self.local_development_retention = options.local_development_retention
        self.strict_contract = options.strict_contract
        self._base_url = base_url
        self._api_key = api_key
        self._credential_provider = credential_provider
        self._options = options
        self._sleep = _sleep or time.sleep
        self._random = _random or random.Random()
        self.transfer_hosts = options.upload_hosts or (urlsplit(base_url).hostname or "",)
        self._owned = options.http_client is None
        # The inner pool handles environment proxies/CA so every network route
        # stays on the cancellable loop; outer proxy mounts would bypass it.
        self.client = options.http_client or httpx.Client(
            trust_env=False, auth=None, transport=DeadlineTransport(options.maximum_connections)
        )
        self._closed = False
        self._streams: set[Closeable] = set()
        self._responses: set[httpx.Response] = set()
        self._close_lock = threading.Lock()
        self._close_finished = threading.Event()
        self._close_error: BaseException | None = None

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
    def candidate_manifest_sha256(self) -> str | None:
        return self._options.candidate_manifest_sha256

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def request_timeout_seconds(self) -> float:
        return self._options.timeout_seconds

    @property
    def maximum_response_bytes(self) -> int:
        return self._options.maximum_response_bytes

    def _register_stream(self, stream: Closeable) -> None:
        if self._closed:
            raise ContractError("SDK client is closed", code="client_closed")
        if len(self._streams) >= self._options.maximum_connections:
            raise ContractError("SDK active stream limit reached", code="stream_limit")
        self._streams.add(stream)

    def _unregister_stream(self, stream: Closeable) -> None:
        self._streams.discard(stream)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                closing = False
            else:
                self._closed = True
                closing = True
        if not closing:
            if not self._close_finished.wait(self._options.timeout_seconds + 4):
                raise RuntimeError("SDK client cleanup did not complete")
            if self._close_error is not None:
                raise self._close_error
            return
        try:
            self._close_resources()
        except BaseException as error:
            self._close_error = error
            raise
        finally:
            self._close_finished.set()

    def _close_resources(self) -> None:
        errors: list[Exception] = []
        if self._credential_provider is not None:
            try:
                self._credential_provider.close()
            except Exception as error:
                errors.append(error)
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

    def _negotiate(self, operation: Operation, deadline: float) -> NegotiatedCapability | None:
        if operation.keyed and self._api_key is None and self._credential_provider is None:
            raise AuthenticationError("This operation requires an API key")
        if operation.contract == "tunnels.v2":
            from synth_ai.contracts.tunnels import require_control_operation

            require_control_operation(operation, self._options)
            return None
        if operation.contract == "account.v1":
            from synth_ai.contracts.account import require_account_read

            require_account_read(operation, self._options)
            return None
        if operation.id == CAPABILITIES.id:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Capability negotiation exceeded the request deadline")
        receipt = self.request(
            CAPABILITIES,
            Capabilities,
            options=ReadOptions(timeout_seconds=remaining, read_retry_count=0),
        )
        return require_operation(
            receipt,
            operation,
            surface="http",
            expected_manifest_sha256=self._options.release_manifest_sha256,
            candidate_manifest_sha256=self._options.candidate_manifest_sha256,
            strict_contract=self.strict_contract,
        )

    def _open_stream(
        self,
        operation: Operation,
        *,
        path: str,
        query: Mapping[str, str | int | tuple[str, ...]],
        remaining_seconds: float,
        digest: str | None,
        accept: str = "application/octet-stream",
    ) -> httpx.Response:
        selected = request_options(
            operation,
            ReadOptions(
                timeout_seconds=remaining_seconds,
                read_retry_count=0,
            ),
        )
        try:
            return self._open_stream_request(
                operation,
                path=path,
                query=query,
                remaining_seconds=remaining_seconds,
                digest=digest,
                options=selected,
                accept=accept,
            )
        except SynthError as error:
            annotate_request_error(error, operation, selected)
            raise
        except (httpx.TransportError, TimeoutError) as cause:
            kind = (
                RequestTimeoutError
                if isinstance(cause, (httpx.TimeoutException, TimeoutError))
                else TransportError
            )
            error = kind("Byte-stream exchange failed")
            annotate_request_error(error, operation, selected)
            raise error from cause

    def _open_stream_request(
        self,
        operation: Operation,
        *,
        path: str,
        query: Mapping[str, str | int | tuple[str, ...]],
        remaining_seconds: float,
        digest: str | None,
        options: ReadOptions,
        accept: str = "application/octet-stream",
    ) -> httpx.Response:
        """Open one SDK-owned read response, without retries or redirects."""
        if self._closed:
            raise ContractError("SDK client is closed", code="client_closed")
        if operation.mutation or operation.method != "GET":
            raise ValueError("byte streams require an observational GET operation")
        deadline = time.monotonic() + remaining_seconds
        key = (
            self._credential_provider.resolve(deadline)
            if operation.keyed and self._credential_provider is not None
            else self._api_key
        )
        negotiated = self._negotiate(operation, deadline)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise RequestTimeoutError("Capability negotiation exceeded the stream deadline")
        request = build_request(
            self.client,
            self._base_url,
            key,
            operation,
            path,
            b"",
            query,
            options,
            None,
            remaining_seconds,
            self._options.connect_timeout_seconds,
            negotiated=negotiated,
        )
        request.headers["Accept"] = accept
        request.headers.pop("If-Match", None)
        if digest is not None:
            request.headers["If-Match"] = '"' + digest + '"'
        response = self.client.send(request, stream=True, auth=None, follow_redirects=False)
        response = self._adopt_response(response)
        try:
            validate_negotiated_response(response, negotiated, None, False)
        except BaseException as primary:
            try:
                self._release_response(response)
            except Exception as cause:
                handle_cleanup_failure(cause, primary, mutation=False, identity=None)
            raise
        return response

    def mutation_events(self, operation: Operation, *, path: str, body: BaseModel, options=None):
        from .stream import MutationEventStream

        return MutationEventStream(self, operation, path=path, body=body, options=options)

    def request(
        self,
        operation: Operation,
        model: type[T],
        *,
        body: BaseModel | None = None,
        path: str | None = None,
        query: Mapping[str, str | int | tuple[str, ...]] | None = None,
        options: ReadOptions | None = None,
        validate: Callable[[T], T] | None = None,
    ) -> T:
        selected = request_options(operation, options)
        try:
            return self._request(
                operation,
                model,
                body=body,
                path=path,
                query=query,
                options=selected,
                validate=validate,
            )
        except SynthError as error:
            annotate_request_error(error, operation, selected)
            raise

    def _request(
        self,
        operation: Operation,
        model: type[T],
        *,
        body: BaseModel | None = None,
        path: str | None = None,
        query: Mapping[str, str | int | tuple[str, ...]] | None = None,
        options: ReadOptions | None = None,
        validate: Callable[[T], T] | None = None,
    ) -> T:
        if self._closed:
            raise ContractError("SDK client is closed", code="client_closed")
        if options is not None and options.response_metadata is not None:
            options.response_metadata.rate_limits = ()
        payload = encode_body(body)
        identity = mutation_identity(operation, path, payload, options)
        started = time.monotonic()
        timeout = (
            options.timeout_seconds
            if options is not None and options.timeout_seconds is not None
            else self._options.request_timeout(operation.id)
        )
        deadline = started + timeout
        key = (
            self._credential_provider.resolve(deadline)
            if operation.keyed and self._credential_provider is not None
            else self._api_key
        )
        negotiated = self._negotiate(operation, deadline)
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
                key,
                operation,
                path,
                payload,
                query,
                options,
                identity,
                remaining,
                self._options.connect_timeout_seconds,
                negotiated=negotiated,
                attempt=attempt,
            )
            response: httpx.Response | None = None
            try:
                response = self.client.send(request, stream=True, auth=None, follow_redirects=False)
                self._adopt_response(response)
                if options is not None and options.response_metadata is not None:
                    options.response_metadata.capture(response.headers)
                exchange = classify_exchange(
                    response,
                    None,
                    contract=operation.contract,
                    mutation=operation.mutation,
                    identity=identity,
                    attempt=attempt,
                    attempts=attempts,
                    operation=operation.id,
                )
                content = bytearray()
                first_byte: float | None = None
                if exchange.kind == "decode":
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise ContractError(
                            "Compressed responses are outside the bounded wire contract"
                        )
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
                    exchange = classify_exchange(
                        response,
                        bytes(content),
                        contract=operation.contract,
                        mutation=operation.mutation,
                        identity=identity,
                        attempt=attempt,
                        attempts=attempts,
                        operation=operation.id,
                    )
                finished = time.monotonic()
                if finished >= deadline:
                    raise RequestTimeoutError("Response deadline expired")
                if exchange.error is not None:
                    raise annotate_response_error(response, exchange.error)
                if exchange.kind == "retry":
                    hint = retry_after_seconds(response.headers.get("Retry-After"))
                    delay = delay_seconds(
                        attempt,
                        self._options.retry_base_delay_seconds,
                        self._options.retry_max_delay_seconds,
                        hint,
                        rng=self._random,
                    )
                    failure = retry_failure(
                        status=response.status_code,
                        delay=delay,
                        remaining=deadline - time.monotonic(),
                        retry_after=hint,
                    )
                    if failure is not None:
                        annotate_response_error(response, failure)
                    self._release_response(response)
                    response = None
                    if failure is not None:
                        raise failure
                    self._sleep(delay)
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
                    negotiated=negotiated,
                    operation=operation.id,
                )
                if operation.id == CAPABILITIES.id and isinstance(decoded, Capabilities):
                    validate_release(
                        decoded,
                        self._options.release_manifest_sha256,
                        self._options.candidate_manifest_sha256,
                    )
                return validate(decoded) if validate is not None else decoded
            except ContractError as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation response is invalid; reconcile its identity",
                        identity=identity,
                        reason="response_invalid",
                        status=response.status_code if response is not None else None,
                    ) from cause
                raise
            except (httpx.TransportError, RequestTimeoutError) as cause:
                if operation.mutation:
                    raise OutcomeUncertainError(
                        "Mutation delivery is uncertain; retain its identity",
                        identity=identity,
                        reason="response_lost",
                    ) from cause
                if isinstance(cause, httpx.TransportError) and attempt < attempts:
                    delay = delay_seconds(
                        attempt,
                        self._options.retry_base_delay_seconds,
                        self._options.retry_max_delay_seconds,
                        None,
                        rng=self._random,
                    )
                    failure = retry_failure(
                        status=None,
                        delay=delay,
                        remaining=deadline - time.monotonic(),
                        retry_after=None,
                    )
                    if response is not None:
                        self._release_response(response)
                        response = None
                    if failure is not None:
                        raise failure from cause
                    self._sleep(delay)
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
