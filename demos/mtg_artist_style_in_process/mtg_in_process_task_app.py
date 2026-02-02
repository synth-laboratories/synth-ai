"""MTG artist style matching task app (in-process, text-only).

This demo is intentionally lightweight: it scores prompts against the target
style description without external image generation dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from synth_ai.data.artifacts import Artifact
from synth_ai.sdk.localapi._impl import (
    LocalAPIConfig,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
    build_trace_payload,
)

logger = logging.getLogger(__name__)

APP_ID = "mtg_artist_style_in_process"
APP_NAME = "MTG Artist Style Matching (In-Process)"
APP_DESCRIPTION = "Text-only MTG artist style matching demo for in-process GEPA/eval."

_DEMO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = _DEMO_ROOT / "mtg_artist_style" / "card_descriptions.json"

DEFAULT_ARTIST_KEY = (os.environ.get("MTG_ARTIST_KEY") or "seb_mckinnon").strip()
DEFAULT_MAX_EXAMPLES = int(os.environ.get("MTG_MAX_EXAMPLES", "8"))
DEFAULT_CUSTOMER_NOTE = (
    os.environ.get("MTG_CUSTOMER_STYLE_NOTE")
    or "Emphasize moody lighting, textured brushwork, and gothic silhouettes."
).strip()

DEFAULT_PROMPT_SECTIONS = [
    {
        "role": "system",
        "pattern": (
            "You are a creative prompt writer. Generate a short art-direction prompt "
            "that matches the requested MTG artist style."
        ),
        "order": 0,
    },
    {
        "role": "user",
        "pattern": (
            "Card name: {card_name}\n"
            "Card description: {card_description}\n"
            "Target artist style: {target_artist_style}\n"
            "Customer style note: {customer_style_note}\n\n"
            "Return a single paragraph prompt. Do NOT mention the artist name directly."
        ),
        "order": 1,
    },
]

_STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "under",
    "their",
    "there",
    "about",
    "while",
    "where",
    "when",
    "which",
    "your",
    "you",
    "for",
    "are",
    "was",
    "were",
    "his",
    "her",
    "its",
    "our",
    "not",
    "only",
    "use",
    "uses",
    "used",
    "also",
    "into",
}


def _safe_format(pattern: str, mapping: dict[str, str]) -> str:
    try:
        return pattern.format_map(mapping)
    except Exception:
        return pattern


def _render_messages_from_sections(
    sections: Iterable[dict[str, Any]], values: dict[str, str]
) -> list[dict[str, str]]:
    ordered = sorted(sections, key=lambda item: item.get("order", 0))
    messages: list[dict[str, str]] = []
    for section in ordered:
        role = section.get("role", "user")
        pattern = section.get("pattern", "")
        if not isinstance(pattern, str):
            pattern = str(pattern)
        messages.append({"role": role, "content": _safe_format(pattern, values)})
    return messages


def _normalize_keywords(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]+", " ", text.lower())
    tokens = [token.strip() for token in cleaned.split() if token.strip()]
    return [token for token in tokens if len(token) > 3 and token not in _STOPWORDS]


def _score_prompt(
    prompt_text: str,
    target_style: str,
    artist_name: str,
    customer_note: str,
) -> tuple[float, dict[str, Any]]:
    prompt_lower = prompt_text.lower()
    keywords = _normalize_keywords(target_style)
    if customer_note:
        keywords.extend(_normalize_keywords(customer_note))
    unique_keywords = sorted(set(keywords))

    matched = [kw for kw in unique_keywords if kw in prompt_lower]
    total = max(len(unique_keywords), 1)
    base_score = len(matched) / total

    penalty = 0.0
    if artist_name and artist_name.lower() in prompt_lower:
        penalty = 0.25

    score = max(0.0, min(1.0, base_score - penalty))
    return score, {
        "matched_keywords": matched,
        "total_keywords": total,
        "penalty": penalty,
    }


@dataclass
class MtgSample:
    index: int
    card_name: str
    card_description: str
    artist_name: str
    target_artist_style: str
    artist_key: str


class MtgStyleDataset:
    def __init__(self, artist_key: str, max_examples: int) -> None:
        self._artist_key = artist_key
        self._max_examples = max_examples
        self._artists: dict[str, Any] | None = None
        self._cards: list[dict[str, Any]] | None = None

    def _load(self) -> None:
        if self._cards is not None:
            return
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"MTG dataset not found: {DATA_PATH}")
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        artists = payload.get("artists", {})
        cards = payload.get("cards", [])

        artist_key = self._artist_key
        if artist_key and artist_key.lower() != "all":
            cards = [card for card in cards if card.get("artist_key") == artist_key]

        cards = [card for card in cards if card.get("art_description")]
        if self._max_examples > 0:
            cards = cards[: self._max_examples]

        if not cards:
            raise ValueError("No MTG cards available for the selected artist filter.")

        self._artists = artists
        self._cards = cards

    def size(self) -> int:
        self._load()
        return len(self._cards or [])

    def sample(self, index: int) -> MtgSample:
        self._load()
        cards = self._cards or []
        card = cards[index % len(cards)]
        artist_key = card.get("artist_key", "")
        artist_meta = (self._artists or {}).get(artist_key, {})
        return MtgSample(
            index=index % len(cards),
            card_name=card.get("card_name", ""),
            card_description=card.get("art_description", ""),
            artist_name=artist_meta.get("name", card.get("artist", "")),
            target_artist_style=artist_meta.get("style_description", ""),
            artist_key=artist_key,
        )


_DATASET_CACHE: dict[tuple[str, int], MtgStyleDataset] = {}


def _get_dataset(artist_key: str, max_examples: int) -> MtgStyleDataset:
    key = (artist_key or "all", max_examples)
    dataset = _DATASET_CACHE.get(key)
    if dataset is None:
        dataset = MtgStyleDataset(artist_key, max_examples)
        _DATASET_CACHE[key] = dataset
    return dataset


def build_config() -> LocalAPIConfig:
    dataset = _get_dataset(DEFAULT_ARTIST_KEY, DEFAULT_MAX_EXAMPLES)

    def provide_taskset_description() -> dict[str, Any]:
        return {
            "id": APP_ID,
            "name": APP_NAME,
            "splits": ["train"],
            "size": dataset.size(),
        }

    def provide_task_instances(seeds: Iterable[int]) -> Iterable[TaskInfo]:
        items: list[TaskInfo] = []
        for seed in seeds:
            sample = dataset.sample(int(seed))
            items.append(
                TaskInfo(
                    task={
                        "id": APP_ID,
                        "name": APP_NAME,
                        "description": APP_DESCRIPTION,
                    },
                    dataset={"id": APP_ID, "name": "mtg_artist_style", "splits": ["train"]},
                    inference={"model": "prompt-only"},
                    limits={"max_turns": 1},
                    task_metadata={
                        "card_name": sample.card_name,
                        "artist": sample.artist_name,
                        "artist_key": sample.artist_key,
                    },
                )
            )
        return items

    async def rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        seed = int(request.env.seed or 0)
        env_cfg = dict(request.env.config or {})
        artist_key = str(env_cfg.get("artist_key") or DEFAULT_ARTIST_KEY)
        max_examples = int(env_cfg.get("max_examples") or DEFAULT_MAX_EXAMPLES)
        customer_note = str(env_cfg.get("customer_style_note") or DEFAULT_CUSTOMER_NOTE)

        local_dataset = _get_dataset(artist_key, max_examples)
        sample = local_dataset.sample(seed)

        prompt_sections = env_cfg.get("prompt_sections") or DEFAULT_PROMPT_SECTIONS
        values = {
            "card_name": sample.card_name,
            "card_description": sample.card_description,
            "target_artist_style": sample.target_artist_style,
            "artist_name": sample.artist_name,
            "customer_style_note": customer_note,
        }
        messages = _render_messages_from_sections(prompt_sections, values)
        prompt_text = "\n\n".join(
            [message.get("content", "") for message in messages if message.get("content")]
        )

        reward, details = _score_prompt(
            prompt_text,
            sample.target_artist_style,
            sample.artist_name,
            customer_note,
        )

        reward_info = RolloutMetrics(
            outcome_reward=reward,
            details=details,
        )

        trace_payload = build_trace_payload(
            messages,
            {"output": prompt_text},
            correlation_id=request.trace_correlation_id,
            session_id=f"mtg-{seed}",
            metadata={"artist_key": sample.artist_key},
        )

        artifact = Artifact(
            content={
                "prompt": prompt_text,
                "card_name": sample.card_name,
                "artist_key": sample.artist_key,
                "matched_keywords": details.get("matched_keywords", []),
            },
            content_type="application/json",
            metadata={"app_id": APP_ID, "artist_key": sample.artist_key},
            trace_correlation_id=request.trace_correlation_id,
        )

        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=reward_info,
            trace=trace_payload,
            artifact=[artifact],
        )

    return LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description=APP_DESCRIPTION,
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=rollout,
    )


if __name__ == "__main__":
    from synth_ai.sdk.localapi import run_local_api

    run_local_api(build_config(), host="127.0.0.1", port=8114)
