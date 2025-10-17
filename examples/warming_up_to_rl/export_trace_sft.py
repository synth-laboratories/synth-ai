#!/usr/bin/env python3
"""Export behavioural-cloning datasets from tracing_v3 SQLite traces with filters."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Row = sqlite3.Row


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


AchievementMap = dict[tuple[str, int], dict[str, list[str]]]


def fetch_achievement_data(
    conn: sqlite3.Connection,
) -> tuple[
    AchievementMap,
    Counter,
    Counter,
    Counter,
    dict[str, set[str]],
    dict[str, set[str]],
]:
    achievements_map: AchievementMap = defaultdict(lambda: {"unique": [], "all": []})
    session_unique_sets: dict[str, set[str]] = defaultdict(set)
    session_final_achievements: dict[str, set[str]] = defaultdict(set)
    achievement_name_counts: Counter = Counter()

    rows = conn.execute(
        """
        SELECT er.session_id, er.reward_value, er.annotation, ev.metadata
        FROM event_rewards er
        JOIN events ev ON er.event_id = ev.id
        WHERE er.reward_type = 'unique_achievement_delta' AND er.reward_value > 0
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        annotation = _parse_json(row["annotation"]) or {}
        metadata = _parse_json(row["metadata"]) or {}
        turn = metadata.get("turn")
        if turn is None:
            continue
        new_unique = annotation.get("new_unique") or []
        if not isinstance(new_unique, list):
            continue
        if new_unique:
            achievements_map[(session_id, int(turn))]["unique"].extend(new_unique)
            session_unique_sets[session_id].update(new_unique)

    rows = conn.execute(
        """
        SELECT er.session_id, er.reward_value, er.annotation, ev.metadata
        FROM event_rewards er
        JOIN events ev ON er.event_id = ev.id
        WHERE er.reward_type = 'achievement_delta' AND er.reward_value > 0
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        annotation = _parse_json(row["annotation"]) or {}
        metadata = _parse_json(row["metadata"]) or {}
        turn = metadata.get("turn")
        if turn is None:
            continue
        turned_true = annotation.get("turned_true") or []
        if not isinstance(turned_true, list):
            continue
        if turned_true:
            achievements_map[(session_id, int(turn))]["all"].extend(turned_true)

    rows = conn.execute(
        """
        SELECT session_id, reward_metadata
        FROM outcome_rewards
        WHERE reward_metadata IS NOT NULL
        """
    ).fetchall()
    for row in rows:
        session_id = row["session_id"]
        metadata = _parse_json(row["reward_metadata"])
        if not isinstance(metadata, dict):
            continue
        final_achievements = metadata.get("achievements") or []
        if isinstance(final_achievements, list):
            cleaned = [a for a in final_achievements if isinstance(a, str)]
            session_unique_sets[session_id].update(cleaned)
            session_final_achievements[session_id].update(cleaned)

    unique_counts_per_session: Counter = Counter()
    for session_id, achievement_set in session_unique_sets.items():
        unique_counts_per_session[session_id] = len(achievement_set)
        achievement_name_counts.update(achievement_set)

    achievement_size_counts: Counter = Counter()
    for _session_id, count in unique_counts_per_session.items():
        achievement_size_counts[count] += 1

    return (
        achievements_map,
        unique_counts_per_session,
        achievement_name_counts,
        achievement_size_counts,
        session_unique_sets,
        session_final_achievements,
    )


def fetch_session_models(conn: sqlite3.Connection) -> dict[str, tuple[str, str, int]]:
    rows = conn.execute(
        """
        SELECT session_id, model_name, provider, COUNT(*) AS calls
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY session_id, model_name, provider
        """
    ).fetchall()

    session_models: dict[str, tuple[str, str, int]] = {}
    for row in rows:
        session_id = row["session_id"]
        calls = int(row["calls"] or 0)
        current = session_models.get(session_id)
        if current is None or calls > current[2]:
            session_models[session_id] = (row["model_name"], row["provider"], calls)
    return session_models


def fetch_outcome_rewards(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT session_id, total_reward, reward_metadata
        FROM outcome_rewards
        """
    ).fetchall()

    outcome_data: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = _parse_json(row["reward_metadata"])
        achievements = set()
        if isinstance(metadata, dict):
            ach = metadata.get("achievements") or []
            if isinstance(ach, list):
                achievements = {a for a in ach if isinstance(a, str)}
        outcome_data[row["session_id"]] = {
            "total_reward": float(row["total_reward"] or 0.0),
            "achievements": achievements,
        }
    return outcome_data


def fetch_event_reward_totals(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, float]]]:
    rows = conn.execute(
        """
        SELECT session_id, reward_type, COUNT(*) AS events, COALESCE(SUM(reward_value), 0) AS total_value
        FROM event_rewards
        GROUP BY session_id, reward_type
        """
    ).fetchall()

    event_totals: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        event_totals[row["session_id"]][row["reward_type"]] = {
            "events": int(row["events"] or 0),
            "total": float(row["total_value"] or 0.0),
        }
    return event_totals


def parse_event_filters(specs: list[str] | None) -> list[tuple[str, float]]:
    filters: list[tuple[str, float]] = []
    if not specs:
        return filters
    for spec in specs:
        reward_type, _, min_val_str = spec.partition(":")
        reward_type = reward_type.strip()
        if not reward_type:
            continue
        min_val = 0.0
        if min_val_str:
            try:
                min_val = float(min_val_str)
            except ValueError as e:
                print(f"Invalid event reward specification '{spec}'", file=sys.stderr)
                raise SystemExit(1) from e
        filters.append((reward_type, min_val))
    return filters


def _collect_content(
    parts: Iterable[dict[str, Any]] | None,
) -> tuple[Any, bool]:
    """Normalise multimodal content parts into OpenAI-style segments."""

    if not parts:
        return "", False

    segments: list[dict[str, Any]] = []
    has_image = False

    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text = part.get("text")
            if isinstance(text, str):
                segments.append({"type": "text", "text": text})
        elif ptype == "image":
            uri = part.get("uri")
            mime_type = part.get("mime_type") or "image/png"
            data_url = None
            if isinstance(uri, str) and uri.startswith("data:"):
                data_url = uri
            else:
                source = part.get("data") or part.get("source")
                if isinstance(source, dict):
                    base64_data = source.get("data")
                    media_type = source.get("media_type") or mime_type
                    if isinstance(base64_data, str) and base64_data:
                        data_url = f"data:{media_type};base64,{base64_data}"
            if data_url:
                has_image = True
                segments.append({"type": "image_url", "image_url": {"url": data_url}})
        elif ptype == "image_url":
            image_url = part.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    has_image = True
                    segments.append({"type": "image_url", "image_url": {"url": url}})

    if not segments:
        return "", False
    if not has_image and len(segments) == 1 and segments[0]["type"] == "text":
        return segments[0]["text"], False
    return segments, has_image


def _normalise_output_content(content: Any) -> tuple[Any, bool]:
    if isinstance(content, list):
        return _collect_content(content)
    if isinstance(content, str):
        return content, False
    if content is None:
        return "", False
    return str(content), False


def _normalise_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    if not tool_calls:
        return normalised
    for idx, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        entry = dict(call)

        func_payload: dict[str, Any] | None = (
            entry.get("function") if isinstance(entry.get("function"), dict) else None
        )
        name = entry.get("name") or (func_payload.get("name") if func_payload else None) or "tool"

        args = None
        if func_payload and "arguments" in func_payload:
            args = func_payload.get("arguments")
        else:
            args = entry.get("arguments")
            if args is None:
                raw = entry.pop("arguments_json", None)
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except Exception:
                        args = raw

        if isinstance(args, dict | list):
            args_str = json.dumps(args, ensure_ascii=False)
        elif isinstance(args, str):
            args_str = args
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)

        call_id = entry.get("id") or entry.get("call_id") or f"call_{idx}"

        normalised.append(
            {
                "id": str(call_id),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": args_str,
                },
            }
        )

    return normalised


def build_sft_dataset(
    conn: sqlite3.Connection,
    achievements_map: AchievementMap,
    sessions_filter: set[str],
    *,
    allowed_models: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, session_id, metadata, model_name, provider, call_records
        FROM events
        WHERE event_type = 'cais' AND call_records IS NOT NULL
        ORDER BY session_id, id
        """
    ).fetchall()

    dataset: list[dict[str, Any]] = []
    cumulative_unique: dict[str, int] = defaultdict(int)
    session_turn_counters: dict[str, int] = defaultdict(int)

    for row in rows:
        session_id = row["session_id"]
        if session_id not in sessions_filter:
            continue
        if allowed_models and row["model_name"] not in allowed_models:
            continue

        metadata = _parse_json(row["metadata"]) or {}
        turn = metadata.get("turn")
        if turn is None:
            step_id = metadata.get("step_id")
            if isinstance(step_id, str) and step_id.startswith("turn_"):
                try:
                    turn = int(step_id.split("_", 1)[1])
                except (ValueError, IndexError):
                    turn = None
        if turn is None:
            turn = session_turn_counters[session_id]
            session_turn_counters[session_id] = turn + 1
        else:
            try:
                turn = int(turn)
            except (TypeError, ValueError):
                continue
            session_turn_counters[session_id] = max(session_turn_counters[session_id], turn + 1)

        call_records = _parse_json(row["call_records"]) or []
        if not isinstance(call_records, list) or not call_records:
            continue

        for record in call_records:
            messages: list[dict[str, Any]] = []
            input_has_image = False
            for message in record.get("input_messages", []):
                role = message.get("role", "unknown")
                content, has_image = _collect_content(message.get("parts"))
                if (content == "" or content is None) and not has_image:
                    continue
                if has_image and role == "user":
                    input_has_image = True
                messages.append({"role": role, "content": content})

            assistant_content_value: Any = ""
            assistant_has_image = False
            assistant_tool_calls: list[dict[str, Any]] = []

            output_text = record.get("output_text")
            parsed_response: dict[str, Any] | None = None
            if isinstance(output_text, str) and output_text:
                try:
                    parsed_response = json.loads(output_text)
                except json.JSONDecodeError:
                    parsed_response = None

            if parsed_response:
                choices = parsed_response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    assistant_content_value, assistant_has_image = _normalise_output_content(
                        message.get("content")
                    )
                    assistant_tool_calls = _normalise_tool_calls(message.get("tool_calls"))

            if not assistant_tool_calls:
                assistant_tool_calls = _normalise_tool_calls(record.get("output_tool_calls"))

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content_value,
            }
            if assistant_tool_calls:
                assistant_message["tool_calls"] = assistant_tool_calls

            content_empty = assistant_message.get("content") in ("", None)
            if content_empty and not assistant_message.get("tool_calls"):
                continue

            messages.append(assistant_message)

            if len(messages) < 2:
                continue

            achievements = achievements_map.get((session_id, turn), {"unique": [], "all": []})
            cumulative_unique[session_id] += len(achievements.get("unique", []))

            metadata = {
                "session_id": session_id,
                "turn": turn,
                "model": row["model_name"],
                "provider": row["provider"] or "unknown",
                "achievements": {
                    "new_unique": achievements.get("unique", []),
                    "turned_true": achievements.get("all", []),
                    "cumulative_unique": cumulative_unique[session_id],
                },
                "user_has_image": input_has_image,
                "assistant_has_image": assistant_has_image,
                "has_image": input_has_image or assistant_has_image,
            }

            dataset.append({"messages": messages, "metadata": metadata})
            if limit is not None and len(dataset) >= limit:
                return dataset

    return dataset


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


def _validate_dataset(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for idx, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"row {idx}: missing messages list")
            if len(errors) >= 20:
                break
            continue
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"row {idx}: message {msg_idx} is not an object")
                break
            if "role" not in msg or "content" not in msg:
                errors.append(f"row {idx}: message {msg_idx} missing role/content")
                break
            if not isinstance(msg["role"], str):
                errors.append(f"row {idx}: message {msg_idx} role not string")
                break
            if not isinstance(msg["content"], str):
                errors.append(f"row {idx}: message {msg_idx} content not string")
                break
        if len(errors) >= 20:
            break
    if errors:
        summary = "\n - ".join(errors)
        raise SystemExit(f"Validation error while exporting dataset:\n - {summary}")


def _find_trace_database() -> Path | None:
    """Automatically discover the trace database in common locations."""

    # Check for demo directory from state
    try:
        state_path = Path.home() / ".synth-ai" / "demo.json"
        if state_path.exists():
            import json

            with state_path.open() as f:
                data = json.load(f)
                demo_dir = data.get("DEMO_DIR")
                if demo_dir:
                    candidate = Path(demo_dir) / "traces" / "v3" / "synth_ai.db"
                    if candidate.exists():
                        return candidate
    except Exception:
        pass

    # Search upward from current directory
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "traces" / "v3" / "synth_ai.db"
        if candidate.exists():
            return candidate

    # Check standard locations
    standard_locations = [
        Path("traces/v3/synth_ai.db"),
        Path("../traces/v3/synth_ai.db"),
        Path.home() / "synth-ai" / "traces" / "v3" / "synth_ai.db",
    ]

    for location in standard_locations:
        try:
            if location.exists():
                return location.resolve()
        except Exception:
            continue

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Path to tracing_v3 SQLite DB")
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Destination JSONL path for the exported dataset",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to sessions whose dominant model matches (repeatable)",
    )
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="Restrict to sessions whose dominant provider matches (repeatable)",
    )
    parser.add_argument(
        "--min-unique", type=int, default=None, help="Minimum unique achievements per session"
    )
    parser.add_argument(
        "--max-unique", type=int, default=None, help="Maximum unique achievements per session"
    )
    parser.add_argument(
        "--exclude-achievement",
        action="append",
        dest="exclude_achievements",
        help="Achievements to ignore when evaluating --min-unique/--max-unique (repeatable)",
    )
    parser.add_argument(
        "--require-achievement",
        action="append",
        dest="required_achievements",
        help="Require these outcome achievements (repeatable)",
    )
    parser.add_argument(
        "--min-outcome-reward",
        type=float,
        default=None,
        help="Minimum total outcome reward per session",
    )
    parser.add_argument(
        "--max-outcome-reward",
        type=float,
        default=None,
        help="Maximum total outcome reward per session",
    )
    parser.add_argument(
        "--event-reward",
        action="append",
        dest="event_reward_filters",
        help="Require reward_type[:min_total] in event_rewards (repeatable)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of examples to emit"
    )
    args = parser.parse_args()

    # Auto-discover database if not specified
    db_path = args.db
    if db_path is None:
        db_path = _find_trace_database()
        if db_path:
            print(f"Found trace database: {db_path}")
        else:
            print("\nTrace database configuration:")
            db_input = input("Trace database path [traces/v3/synth_ai.db]: ").strip()
            db_path = Path(db_input) if db_input else Path("traces/v3/synth_ai.db")

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path = args.output
    if not output_path:
        output_path = Path("ft_data/crafter_traces.jsonl")
        print(f"Output will be written to: {output_path.resolve()}")

    min_unique = args.min_unique
    if min_unique is None:
        min_unique = 0  # Default to including all traces
        print(f"Minimum unique achievements filter: {min_unique} (all traces)")

    # Override args with prompted values
    args.db = db_path
    args.output = output_path
    args.min_unique = min_unique

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        raise SystemExit(1)

    conn = connect(args.db)
    try:
        (
            achievements_map,
            unique_counts_per_session,
            _name_counts,
            _size_counts,
            session_unique_sets,
            session_final_achievements,
        ) = fetch_achievement_data(conn)
        session_models = fetch_session_models(conn)
        outcome_data = fetch_outcome_rewards(conn)
        event_totals = fetch_event_reward_totals(conn)
        event_filters = parse_event_filters(args.event_reward_filters)

        allowed_models = set(args.models) if args.models else None
        allowed_providers = set(args.providers) if args.providers else None
        required_achievements = set(args.required_achievements or [])
        excluded_achievements = set(args.exclude_achievements or [])

        eligible_sessions: set[str] = set()
        for session_id, (model_name, provider, _calls) in session_models.items():
            if allowed_models and model_name not in allowed_models:
                continue
            if allowed_providers and (provider or "unknown") not in allowed_providers:
                continue

            session_uniques = session_unique_sets.get(session_id, set())
            adjusted_uniques = {a for a in session_uniques if a not in excluded_achievements}
            unique_count = len(adjusted_uniques)
            if args.min_unique is not None and unique_count < args.min_unique:
                continue
            if args.max_unique is not None and unique_count > args.max_unique:
                continue

            outcome = outcome_data.get(session_id)
            total_reward = outcome["total_reward"] if outcome else 0.0
            final_achievements = (
                outcome["achievements"]
                if outcome
                else session_final_achievements.get(session_id, set())
            )

            if args.min_outcome_reward is not None and total_reward < args.min_outcome_reward:
                continue
            if args.max_outcome_reward is not None and total_reward > args.max_outcome_reward:
                continue
            if required_achievements and not required_achievements.issubset(final_achievements):
                continue

            session_event_totals = event_totals.get(session_id, {})
            meets_event_filters = True
            for reward_type, min_total in event_filters:
                total = session_event_totals.get(reward_type, {}).get("total", 0.0)
                if total < min_total:
                    meets_event_filters = False
                    break
            if not meets_event_filters:
                continue

            eligible_sessions.add(session_id)

        if not eligible_sessions:
            print("No sessions matched the provided filters.", file=sys.stderr)
            raise SystemExit(1)

        dataset = build_sft_dataset(
            conn,
            achievements_map,
            eligible_sessions,
            allowed_models=allowed_models,
            limit=args.limit,
        )

        if not dataset:
            print(
                "No rollout steps matched the filters (after session selection).", file=sys.stderr
            )
            raise SystemExit(1)

        _validate_dataset(dataset)
        write_jsonl(args.output, dataset)
        session_ids = {item.get("metadata", {}).get("session_id") for item in dataset}
        session_ids.discard(None)
        print(
            f"Wrote {len(dataset)} examples from {len(session_ids)} session(s) -> {args.output.resolve()}",
            file=sys.stderr,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
