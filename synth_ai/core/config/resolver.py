from typing import Optional, overload

import click

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for config resolution.") from exc


def _get_required_value(*args, **kwargs):
    """Lazy import to avoid circular dependency."""
    from synth_ai.core.config.errors import get_required_value

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
        payload = synth_ai_py.resolve_config_value(
            cli_value,
            env_value,
            config_value,
            default,
        )
        if isinstance(payload, dict) and payload.get("cli_overrides_config"):
            click.secho(
                f"⚠️  {name}: CLI flag overrides config file "
                f"(using {payload.get('cli_value')}, ignoring {payload.get('config_value')})",
                err=True,
                fg="yellow",
            )

        resolved = None
        if isinstance(payload, dict):
            resolved = payload.get("value")

        if required and resolved is None:
            return _get_required_value(
                name,
                cli_value=cli_value,
                env_value=env_value,
                config_value=config_value,
                default=default,
                docs_url=docs_url,
            )

        return resolved
