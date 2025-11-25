from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import click

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _toml = None  # type: ignore[assignment]

from synth_ai.core.tracing_v3 import SessionTracer  # type: ignore[import-untyped]
from synth_ai.sdk.task.config import FilterConfig

from .errors import (
    FilterCliError,
    FilterConfigNotFoundError,
    FilterConfigParseError,
    InvalidFilterConfigError,
    MissingFilterTableError,
    NoSessionsMatchedError,
    NoTracesFoundError,
    TomlUnavailableError,
)
from .validation import validate_filter_options

__all__ = ["command", "get_command", "filter_command"]


def _parse_datetime_for_trace(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        value = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                dt = datetime.fromtimestamp(float(value), tz=UTC)
            except Exception:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            return None
    return None


def _score_ok(value: Any, min_val: Any, max_val: Any) -> bool:
    try:
        if value is None:
            return min_val is None
        value = float(value)
    except Exception:
        return False
    if min_val is not None and value < float(min_val):
        return False
    return not (max_val is not None and value > float(max_val))


def _load_filter_config(config_path: Path) -> tuple[FilterConfig, dict[str, Any]]:
    if _toml is None:
        raise TomlUnavailableError(hint="Install tomli or use Python 3.11+")

    if not config_path.exists():
        raise FilterConfigNotFoundError(path=str(config_path))

    try:
        config_data = _toml.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - validation tests cover common cases
        raise FilterConfigParseError(path=str(config_path), detail=str(exc)) from exc

    filter_cfg_dict = config_data.get("filter") if isinstance(config_data, dict) else None
    if not isinstance(filter_cfg_dict, dict):
        raise MissingFilterTableError()

    try:
        normalized = validate_filter_options(filter_cfg_dict)
        normalized_dict = dict(normalized)
        filter_cfg = FilterConfig.from_dict(normalized_dict)
    except (ValueError, TypeError) as validation_error:
        raise InvalidFilterConfigError(detail=str(validation_error)) from validation_error

    click.echo(
        f"✓ Config validated: db={filter_cfg.db}, output={filter_cfg.output}"
    )
    if filter_cfg.min_official_score is not None:
        click.echo(
            f"  → Filtering for official score >= {filter_cfg.min_official_score}"
        )
    if filter_cfg.limit:
        click.echo(f"  → Limiting to {filter_cfg.limit} examples")

    return filter_cfg, normalized_dict


def _extract_content(content: Any) -> Any:
    if isinstance(content, dict) and "content" in content:
        return content["content"]
    return content


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        payload = content.get("payload") if isinstance(content.get("payload"), dict) else None
        if payload and "content" in payload:
            return _extract_text(payload["content"])
        for key in ("text", "content", "content_text"):
            if key in content:
                value = content[key]
                if isinstance(value, str):
                    return value
        try:
            return json.dumps(content)
        except Exception:  # pragma: no cover - defensive
            return str(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts) if parts else str(content)
    return str(content)


def _select_messages(message_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, msg_row in enumerate(message_rows):
        msg_type = msg_row.get("message_type")
        content_raw = msg_row.get("content")
        if msg_type not in {"user", "policy_user_prompt"}:
            continue

        # Look backwards for system prompt
        system_msg = None
        for prev in range(index - 1, -1, -1):
            prev_type = message_rows[prev].get("message_type")
            if prev_type == "policy_system_prompt":
                system_msg = message_rows[prev]
                break

        assistant_msg = None
        tool_call_msg = None
        for follow in range(index + 1, len(message_rows)):
            next_type = message_rows[follow].get("message_type")
            if next_type == "assistant":
                assistant_msg = message_rows[follow]
                break
            elif next_type == "policy_tool_call":
                tool_call_msg = message_rows[follow]
                break

        try:
            user_content = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
        except Exception:
            user_content = content_raw

        user_content = _extract_content(user_content)
        user_text = _extract_text(user_content)
        if not user_text:
            continue

        messages = []
        
        # Add system prompt if found
        if system_msg is not None:
            try:
                system_content_raw = system_msg.get("content")
                system_content = json.loads(system_content_raw) if isinstance(system_content_raw, str) else system_content_raw
                system_content = _extract_content(system_content)
                system_text = _extract_text(system_content)
                if system_text:
                    messages.append({"role": "system", "content": system_text})
            except Exception:
                pass

        # Add user message
        user_payload = user_content if isinstance(user_content, list) else user_text
        messages.append({"role": "user", "content": user_payload})

        # Add assistant/tool call response
        assistant_content = None
        if tool_call_msg is not None:
            raw = tool_call_msg.get("content")
            try:
                assistant_content = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                assistant_content = raw
            assistant_content = _extract_content(assistant_content)
        elif assistant_msg is not None:
            raw = assistant_msg.get("content")
            try:
                assistant_content = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                assistant_content = raw
            assistant_content = _extract_content(assistant_content)

        assistant_payload = (
            assistant_content
            if isinstance(assistant_content, list)
            else (_extract_text(assistant_content) if assistant_content is not None else "[no response recorded]")
        )
        messages.append({"role": "assistant", "content": assistant_payload})

        records.append({"messages": messages})
    return records


@click.command(
    "filter",
    help="Export filtered tracing sessions to SFT-ready JSONL based on a TOML config.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    required=True,
    help="Path to TOML config describing the input trace DB, score thresholds, and output JSONL.",
)
def filter_command(config_path: str) -> None:
    try:
        filter_cfg, raw_cfg = _load_filter_config(Path(config_path))
    except FilterCliError as exc:
        raise click.ClickException(_format_filter_error(exc)) from exc

    db_url = filter_cfg.get_db_url()
    output_path = filter_cfg.get_output_path()

    splits = set(filter_cfg.splits)
    task_ids = set(filter_cfg.task_ids)
    models = set(filter_cfg.models)
    min_official = filter_cfg.min_official_score
    max_official = filter_cfg.max_official_score
    min_judge_scores = filter_cfg.min_judge_scores
    max_judge_scores = filter_cfg.max_judge_scores
    min_created = _parse_datetime_for_trace(raw_cfg.get("min_created_at"))
    max_created = _parse_datetime_for_trace(raw_cfg.get("max_created_at"))
    limit = filter_cfg.limit

    async def _run() -> None:
        tracer = SessionTracer(db_url=db_url, auto_save=False)
        await tracer.initialize()

        if tracer.db is None:
            raise FilterCliError("Database not initialized")

        df = await tracer.db.query_traces(
            "SELECT session_id, created_at, metadata FROM session_traces ORDER BY created_at"
        )
        if getattr(df, "empty", True):
            raise NoTracesFoundError(db_url=db_url)

        sessions = df.to_dict("records")
        accepted: list[dict[str, Any]] = []

        for row in sessions:
            metadata_raw = row.get("metadata")
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except Exception:
                    metadata = {}
            elif isinstance(metadata_raw, dict):
                metadata = dict(metadata_raw)
            else:
                metadata = {}

            created_at_raw = row.get("created_at")
            created_at_dt = _parse_datetime_for_trace(created_at_raw)
            session_id = row.get("session_id")

            if splits and metadata.get("task_split") not in splits:
                continue
            if task_ids and metadata.get("task_id") not in task_ids:
                continue
            if models and metadata.get("model") not in models:
                continue

            if min_created and (created_at_dt is None or created_at_dt < min_created):
                continue
            if max_created and (created_at_dt is None or created_at_dt > max_created):
                continue

            total_reward = None
            achievements_count = None
            if min_official is not None or max_official is not None:
                if tracer.db is None:
                    raise FilterCliError("Database not initialized")
                reward_rows = await tracer.db.query_traces(
                    "SELECT total_reward, achievements_count FROM outcome_rewards WHERE session_id = :session_id",
                    {"session_id": session_id},
                )
                reward_records = (
                    reward_rows.to_dict("records")
                    if hasattr(reward_rows, "to_dict")
                    else []
                )
                if reward_records:
                    total_reward = reward_records[0].get("total_reward")
                    achievements_count = reward_records[0].get("achievements_count")
                    if not _score_ok(total_reward, min_official, max_official):
                        continue
                elif min_official is not None:
                    continue

            judge_scores = metadata.get("judge_scores") or {}
            include = True
            for judge_name, threshold in (min_judge_scores or {}).items():
                if not _score_ok(judge_scores.get(judge_name), threshold, None):
                    include = False
                    break
            if not include:
                continue
            for judge_name, threshold in (max_judge_scores or {}).items():
                if not _score_ok(judge_scores.get(judge_name), None, threshold):
                    include = False
                    break
            if not include:
                continue

            messages_query = (
                "\n            SELECT message_type, content, timestamp \n            FROM messages \n            WHERE session_id = :session_id\n            ORDER BY timestamp ASC, id ASC\n        "
            )
            if tracer.db is None:
                raise FilterCliError("Database not initialized")
            msg_df = await tracer.db.query_traces(messages_query, {"session_id": session_id})
            message_rows = (
                msg_df.to_dict("records") if hasattr(msg_df, "to_dict") else []
            )

            if not message_rows:
                prompt = metadata.get("prompt") or ""
                completion = metadata.get("completion") or ""
                if prompt and completion:
                    accepted.append(
                        {
                            "messages": [
                                {"role": "user", "content": str(prompt)},
                                {"role": "assistant", "content": str(completion)},
                            ],
                            "metadata": {
                                "session_id": session_id,
                                "env_name": metadata.get("env_name"),
                                "policy_name": metadata.get("policy_name"),
                                "seed": metadata.get("seed"),
                                "total_reward": total_reward,
                                "achievements_count": achievements_count,
                                "model": metadata.get("model"),
                                "created_at": created_at_dt.isoformat()
                                if created_at_dt
                                else created_at_raw,
                            },
                        }
                    )
                continue

            for record in _select_messages(message_rows):
                record["metadata"] = {
                    "session_id": session_id,
                    "env_name": metadata.get("env_name"),
                    "policy_name": metadata.get("policy_name"),
                    "seed": metadata.get("seed"),
                    "total_reward": total_reward,
                    "achievements_count": achievements_count,
                    "model": metadata.get("model"),
                    "created_at": created_at_dt.isoformat() if created_at_dt else created_at_raw,
                }
                accepted.append(record)

        if not accepted:
            raise NoSessionsMatchedError()

        if limit is not None and limit > 0:
            accepted[:] = accepted[:limit]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for item in accepted:
                handle.write(json.dumps(item, ensure_ascii=False))
                handle.write("\n")

        click.echo(f"Wrote {len(accepted)} examples -> {output_path}")
        if tracer.db is not None:
            await tracer.db.close()

    try:
        asyncio.run(_run())
    except FilterCliError as exc:
        raise click.ClickException(_format_filter_error(exc)) from exc


def _format_filter_error(err: FilterCliError) -> str:
    if isinstance(err, TomlUnavailableError):
        hint = err.hint or "Install tomli or use Python 3.11+."
        return f"TOML parser not available. {hint}"
    if isinstance(err, FilterConfigNotFoundError):
        return f"Filter config not found: {err.path}"
    if isinstance(err, FilterConfigParseError):
        return f"Failed to parse TOML '{err.path}': {err.detail}"
    if isinstance(err, MissingFilterTableError):
        return "Config must contain a [filter] table."
    if isinstance(err, InvalidFilterConfigError):
        return f"Invalid filter config: {err.detail}"
    if isinstance(err, NoTracesFoundError):
        return f"No traces found in database ({err.db_url})."
    if isinstance(err, NoSessionsMatchedError):
        hint = err.hint or "Adjust the filter thresholds or choose a different dataset."
        return f"No sessions matched the provided filters. {hint}"
    return str(err)


command = filter_command


def get_command() -> click.Command:
    return command
