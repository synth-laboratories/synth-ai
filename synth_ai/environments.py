"""Named API targets. See docs/protocols.md for origin and credential precedence."""

import os
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from .errors import InvalidRequestError, local_problem

__all__ = ["Environment", "EnvironmentProfile", "resolve_base_url"]

# Staging origin published by synth-backend/deploy/railway/RUNBOOK.md.
STAGING_API_BASE_URL = "https://core-api-staging-staging-acef.up.railway.app"
PRODUCTION_API_BASE_URL = "https://api.usesynth.ai"


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PROD = "prod"


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    environment: Environment
    api_base_url: str

    @classmethod
    def load(cls, environment: Environment) -> "EnvironmentProfile":
        selected = Environment(environment)
        match selected:
            case Environment.LOCAL:
                variable = "SYNTH_LOCAL_API_BASE_URL"
                origin = os.getenv(variable)
                if origin is None:
                    raise InvalidRequestError(
                        local_problem(
                            "environment_profile_unconfigured",
                            "environment_profile",
                            status=422,
                            detail=variable + " must name the manager-provided local slot origin",
                        )
                    )
            case Environment.STAGING:
                variable = "SYNTH_STAGING_API_BASE_URL"
                origin = os.getenv(variable, STAGING_API_BASE_URL)
            case Environment.PROD:
                variable = "SYNTH_PROD_API_BASE_URL"
                origin = os.getenv(variable, PRODUCTION_API_BASE_URL)
        parsed = urlsplit(origin)
        valid_scheme = parsed.scheme == "https" or (
            selected == Environment.LOCAL and parsed.scheme == "http"
        )
        if (
            not valid_scheme
            or parsed.hostname is None
            or parsed.hostname == ""
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise InvalidRequestError(
                local_problem(
                    "environment_origin_invalid",
                    "environment_profile",
                    status=422,
                    detail=variable + " must contain an API origin",
                )
            )
        return cls(selected, origin.rstrip("/"))


def resolve_base_url(base_url: str | None, environment: Environment | None) -> str | None:
    """Explicit profiles ignore the ambient default origin and keep host-keyed credentials."""
    if environment is None:
        return base_url
    if base_url is not None:
        raise InvalidRequestError(
            local_problem(
                "environment_profile_conflict",
                "environment_profile",
                status=422,
                detail="Select environment or base_url; configure the named profile's origin to override it",
            )
        )
    return EnvironmentProfile.load(environment).api_base_url
