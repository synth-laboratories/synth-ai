"""TUI URL profiles for backend/frontend mode switching."""

from dataclasses import dataclass

from synth_ai.core.urls import BACKEND_URL_BASE, FRONTEND_URL_BASE, local_backend_url


@dataclass(frozen=True)
class TuiUrlProfile:
    backend_url: str
    frontend_url: str


TUI_URL_PROFILES: dict[str, TuiUrlProfile] = {
    "prod": TuiUrlProfile(
        backend_url=BACKEND_URL_BASE,
        frontend_url=FRONTEND_URL_BASE,
    ),
    "dev": TuiUrlProfile(
        backend_url="https://synth-backend-dev-docker.onrender.com",
        frontend_url="http://localhost:3000",
    ),
    "local": TuiUrlProfile(
        backend_url=local_backend_url(),
        frontend_url="http://localhost:3000",
    ),
}


def resolve_tui_profile(name: str) -> TuiUrlProfile:
    """Return the profile for a mode name, defaulting to prod."""
    normalized = name.strip().lower()
    if normalized in ("dev", "development"):
        normalized = "dev"
    elif normalized in ("local", "localhost"):
        normalized = "local"
    else:
        normalized = "prod"
    return TUI_URL_PROFILES[normalized]
