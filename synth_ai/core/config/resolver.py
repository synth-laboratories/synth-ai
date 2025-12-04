from __future__ import annotations

from typing import Optional, overload

import click

from synth_ai.core.telemetry import log_info


def _get_required_value(*args, **kwargs):
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.lib.errors import get_required_value
    return get_required_value(*args, **kwargs)


class ConfigResolver:
    """Resolve configuration values with consistent precedence and messaging."""

    @overload
    @staticmethod
    def resolve(
        name: str,
        *,
        cli_value: Optional[str] = None,
        env_value: Optional[str] = None,
        config_value: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        docs_url: Optional[str] = None,
    ) -> Optional[str]: ...

    @overload
    @staticmethod
    def resolve(
        name: str,
        *,
        cli_value: Optional[str] = None,
        env_value: Optional[str] = None,
        config_value: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = True,
        docs_url: Optional[str] = None,
    ) -> str: ...

    @staticmethod
    def resolve(
        name: str,
        *,
        cli_value: Optional[str] = None,
        env_value: Optional[str] = None,
        config_value: Optional[str] = None,
        default: Optional[str] = None,
        required: bool = False,
        docs_url: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve value with CLI > ENV > CONFIG > DEFAULT precedence."""
        ctx = {"name": name, "required": required, "has_cli": cli_value is not None, "has_env": env_value is not None}
        log_info("ConfigResolver.resolve invoked", ctx=ctx)

        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            stripped = value.strip()
            return stripped if stripped else None

        cli_clean = _clean(cli_value)
        env_clean = _clean(env_value)
        config_clean = _clean(config_value)
        default_clean = _clean(default)

        if cli_clean and config_clean and cli_clean != config_clean:
            click.secho(
                f"⚠️  {name}: CLI flag overrides config file "
                f"(using {cli_clean}, ignoring {config_clean})",
                err=True,
                fg="yellow",
            )

        resolved = cli_clean or env_clean or config_clean or default_clean
        if required:
            return _get_required_value(
                name,
                cli_value=cli_clean,
                env_value=env_clean,
                config_value=config_clean,
                default=default_clean,
                docs_url=docs_url,
            )
        return resolved
