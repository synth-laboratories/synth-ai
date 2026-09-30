"""Use Clerk OAuth for Index without managing agent login credentials.

See testing/specifications/index-mcp-clerk.md. Codex/Claude own browser login,
refresh and credential storage; callers explicitly supply their current token.
"""

from urllib.parse import urlsplit

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

from .client import AsyncIndexAPI, IndexAPI


def oauth_headers(access_token: str) -> dict[str, str]:
    if (
        not isinstance(access_token, str)
        or not access_token
        or any(character.isspace() for character in access_token)
    ):
        raise ValueError("A nonempty OAuth access token without whitespace is required")
    return {"Authorization": f"Bearer {access_token}", "X-Synth-OAuth-Provider": "clerk"}


def _validate_url(base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("Backend URL must not contain credentials, query, or fragment")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("OAuth requires HTTPS or local loopback")


def index_with_oauth(base_url: str, access_token: str) -> IndexAPI:
    """Create an Index client; close its transport when finished."""
    _validate_url(base_url)
    return IndexAPI(HttpTransport(base_url=base_url, headers=oauth_headers(access_token)))


def async_index_with_oauth(base_url: str, access_token: str) -> AsyncIndexAPI:
    _validate_url(base_url)
    return AsyncIndexAPI(AsyncHttpTransport(base_url=base_url, headers=oauth_headers(access_token)))
