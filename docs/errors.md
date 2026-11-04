# Errors

Catch `SwarmsAPIError` for HTTP refusals. Its `problem` preserves the RFC 9457
fields `type`, `title`, `status`, `detail`, `instance` and the Synth extensions
`code`, `service`, `stage`, `upstream_status`, `cause`, `outcome`.
`request_id` carries the response request identifier.

Status-specific subclasses cover invalid requests, authentication, payment,
forbidden access, missing resources, conflicts, rate limits and upstream/server
failures. Unknown codes remain intact. Malformed responses and transport failures
have distinct classes; a mutation transport failure has outcome `unknown`.

`RunFailedError` means a closed run failed. `WaitTimeoutError` means polling
expired; it does not cancel the run. The CLI prints one JSON refusal to stderr.
