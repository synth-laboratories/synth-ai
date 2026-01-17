#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class PatternMessage:
    role: str
    pattern: str
    order: int


@dataclass
class PatternSpec:
    pattern_id: str
    trace_glob: str
    messages: List[PatternMessage]


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in ("text", "input_text"):
                    text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content or "")


def _has_image_content(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "image_url" for item in content)


def _build_regex(pattern: str) -> re.Pattern:
    # Wildcard names are intentionally ignored; any {name} is treated as a wildcard.
    regex_parts: List[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\n":
            regex_parts.append("\n")
        elif char == "\t":
            regex_parts.append("\t")
        elif char == "\r":
            regex_parts.append("\r")
        elif char == "{":
            wildcard_match = re.match(r"\{(\w+)\}", pattern[i:])
            if wildcard_match:
                regex_parts.append(wildcard_match.group(0))
                i += len(wildcard_match.group(0)) - 1
            else:
                regex_parts.append(re.escape(char))
        elif char == "}":
            regex_parts.append(re.escape(char))
        else:
            regex_parts.append(re.escape(char))
        i += 1

    escaped = "".join(regex_parts)
    wildcard_matches = list(re.finditer(r"\{(\w+)\}", escaped))
    regex_pattern = escaped
    for idx, match in enumerate(wildcard_matches):
        wildcard_name = match.group(1)
        is_last = idx == len(wildcard_matches) - 1
        replacement = f"(?P<{wildcard_name}>.+)" if is_last else f"(?P<{wildcard_name}>.+?)"
        regex_pattern = regex_pattern.replace(match.group(0), replacement, 1)

    return re.compile(regex_pattern, re.DOTALL)


def _match_message(actual: Dict[str, Any], pattern_msg: PatternMessage) -> Tuple[bool, str]:
    actual_role = actual.get("role", "")
    if actual_role != pattern_msg.role:
        return False, f"role mismatch: {actual_role!r} != {pattern_msg.role!r}"

    pattern_text = pattern_msg.pattern
    is_image_wildcard = (
        pattern_text.startswith("{{")
        and pattern_text.endswith("}}")
        and pattern_text.count("{{") == 1
    )
    if is_image_wildcard:
        if _has_image_content(actual.get("content")):
            return True, ""
        return False, "expected image content but none found"

    content_text = _extract_text_from_content(actual.get("content", ""))
    regex = _build_regex(pattern_text)
    if regex.search(content_text):
        return True, ""
    return False, "content mismatch"


def _match_prompt(
    actual_messages: List[Dict[str, Any]], pattern_messages: List[PatternMessage]
) -> Tuple[bool, str]:
    actual_idx = 0
    for pattern_msg in pattern_messages:
        matched = False
        while actual_idx < len(actual_messages):
            ok, _ = _match_message(actual_messages[actual_idx], pattern_msg)
            if ok:
                matched = True
                actual_idx += 1
                break
            actual_idx += 1
        if not matched:
            return False, f"no match for pattern role={pattern_msg.role!r}"
    return True, ""


def _load_pattern_spec(path: Path) -> PatternSpec:
    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("pattern_matching", {})
    pattern_id = str(meta.get("id") or path.stem)
    trace_glob = str(meta.get("trace_glob") or "")
    messages_raw = data.get("prompt_learning", {}).get("initial_prompt", {}).get("messages", [])
    messages: List[PatternMessage] = []
    for msg in messages_raw:
        messages.append(
            PatternMessage(
                role=str(msg.get("role", "user")),
                pattern=str(msg.get("pattern", msg.get("content", ""))),
                order=int(msg.get("order", 0)),
            )
        )
    messages.sort(key=lambda m: m.order)
    return PatternSpec(pattern_id=pattern_id, trace_glob=trace_glob, messages=messages)


def _load_trace_messages(trace_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    conversation = metadata.get("conversation") or {}
    request_messages = conversation.get("request_messages") or []
    if not isinstance(request_messages, list):
        return []
    return request_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold-pattern validation against traces.")
    parser.add_argument("--traces-root", default="pattern_traces")
    parser.add_argument("--gold-dir", default="pattern_matching/gold_patterns")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-traces", type=int, default=10)
    args = parser.parse_args()

    traces_root = Path(args.traces_root).resolve()
    gold_dir = Path(args.gold_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    results: Dict[str, Any] = {
        "patterns": [],
        "notes": {"wildcard_names_ignored": True},
    }
    for pattern_file in sorted(gold_dir.glob("*.toml")):
        spec = _load_pattern_spec(pattern_file)
        if not spec.trace_glob:
            continue

        trace_files = sorted(traces_root.glob(spec.trace_glob))
        if args.max_traces > 0:
            trace_files = trace_files[: args.max_traces]

        matched = 0
        mismatches: List[Dict[str, Any]] = []
        for trace_file in trace_files:
            messages = _load_trace_messages(trace_file)
            ok, reason = _match_prompt(messages, spec.messages)
            if ok:
                matched += 1
            else:
                if len(mismatches) < 5:
                    mismatches.append({"trace": str(trace_file), "reason": reason})

        total = len(trace_files)
        match_rate = matched / total if total else 0.0
        results["patterns"].append(
            {
                "pattern_id": spec.pattern_id,
                "trace_count": total,
                "matched_count": matched,
                "match_rate": match_rate,
                "mismatches": mismatches,
            }
        )

        print(f"[{spec.pattern_id}] matched {matched}/{total} (match_rate={match_rate:.2%})")
        for m in mismatches:
            print(f"  - mismatch: {m['reason']} ({m['trace']})")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote results: {output_path}")


if __name__ == "__main__":
    main()
