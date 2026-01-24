"""MTG artist style image generation task app (import-safe).

No side effects at import time.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import httpx
from synth_ai.sdk.localapi import create_local_api
from synth_ai.sdk.localapi._impl import (
    Criterion,
    LocalAPIConfig,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    Rubric,
    RubricBundle,
    TaskInfo,
    build_trace_payload,
    extract_trace_correlation_id,
)

logger = logging.getLogger(__name__)

demo_dir = Path(__file__).resolve().parent
DATA_PATH = demo_dir / "card_descriptions.json"

APP_ID = "mtg_artist_style"
APP_NAME = "MTG Artist Style Image Generation"

DEFAULT_ARTIST_KEY = (os.environ.get("MTG_ARTIST_KEY") or "seb_mckinnon").strip() or "seb_mckinnon"
DEFAULT_MAX_EXAMPLES = int(os.environ.get("MTG_MAX_EXAMPLES", "12"))

DEFAULT_USER_PROMPT = (
    "Generate a Magic: The Gathering card illustration that matches the target artist style.\n\n"
    "Card name: {card_name}\n"
    "Card description: {card_description}\n"
    "Target artist style: {target_artist_style}\n\n"
    "Do NOT mention the artist name in the prompt or any text. Return only the image."
)

MTG_RUBRICS = RubricBundle(
    outcome=Rubric(
        version="1.0",
        goal_text="Evaluate whether the generated image matches the target MTG artist style",
        criteria=[
            Criterion(
                id="artist_identification",
                description="Does the generated image match the target artist style described in the prompt?",
                weight=1.0,
                required=True,
            )
        ],
        aggregation="weighted_sum",
    )
)


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


class MtgArtistDataset:
    """Dataset of MTG card descriptions grouped by artist style."""

    def __init__(self, artist_key: str, max_examples: int) -> None:
        self._artist_key = artist_key
        self._max_examples = max_examples
        self._artists: dict[str, Any] | None = None
        self._cards: list[dict[str, Any]] | None = None

    def _load(self) -> None:
        if self._cards is not None:
            return
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

    def sample(self, index: int) -> dict[str, Any]:
        self._load()
        cards = self._cards or []
        card = cards[index % len(cards)]
        artist_key = card.get("artist_key", "")
        artist_meta = (self._artists or {}).get(artist_key, {})
        return {
            "index": index % len(cards),
            "card_name": card.get("card_name", ""),
            "card_description": card.get("art_description", ""),
            "artist_name": artist_meta.get("name", card.get("artist", "")),
            "target_artist_style": artist_meta.get("style_description", ""),
            "artist_key": artist_key,
            "image_path": card.get("image_path", ""),
        }


def create_mtg_task_app() -> Any:
    dataset = MtgArtistDataset(DEFAULT_ARTIST_KEY, DEFAULT_MAX_EXAMPLES)

    def provide_taskset_description() -> dict[str, Any]:
        return {"splits": ["train"], "sizes": {"train": dataset.size()}}

    def provide_task_instances(seeds: Iterable[int]) -> Iterable[TaskInfo]:
        for seed in seeds:
            sample = dataset.sample(int(seed))
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": sample["index"]},
                environment="mtg_artist_style",
                inference={},
                limits={"max_turns": 1},
                task_metadata={
                    "card_name": sample["card_name"],
                    "card_description": sample["card_description"],
                    "artist_name": sample["artist_name"],
                    "target_artist_style": sample["target_artist_style"],
                    "artist_key": sample["artist_key"],
                    "reference_image_path": sample["image_path"],
                },
            )

    async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        seed = int(request.env.seed)
        sample = dataset.sample(seed)

        policy_cfg = dict(request.policy.config or {})
        inference_url = str(policy_cfg.get("inference_url") or "")
        if not inference_url:
            raise ValueError("inference_url is required in policy config")

        env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
        if not env_api_key:
            raise RuntimeError(
                "ENVIRONMENT_API_KEY is required in environment for calling inference interceptor"
            )

        prompt_sections = (request.env.config or {}).get("prompt_sections")
        values = {
            "card_name": sample["card_name"],
            "card_description": sample["card_description"],
            "target_artist_style": sample["target_artist_style"],
            "artist_name": sample["artist_name"],
        }
        if prompt_sections:
            messages = _render_messages_from_sections(prompt_sections, values)
        else:
            messages = [
                {"role": "user", "content": _safe_format(DEFAULT_USER_PROMPT, values)}
            ]

        model = str(policy_cfg.get("model") or "gemini-2.5-flash-image")
        llm_response: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{inference_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": policy_cfg.get("temperature", 0.7),
                },
                headers={
                    "Authorization": f"Bearer {env_api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                logger.error(
                    "Interceptor error: status=%s body=%s",
                    resp.status_code,
                    resp.text,
                )
                raise
            llm_response = resp.json()

        image_data_url = None
        choices = llm_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if isinstance(image_url, str) and image_url.startswith("data:image/"):
                            image_data_url = image_url
                            break
            elif isinstance(content, str) and content.startswith("data:image/"):
                image_data_url = content

        if not image_data_url:
            raise ValueError("No image data in policy model response")

        # Save generated image for inspection
        output_dir = demo_dir / "generated_images"
        output_dir.mkdir(exist_ok=True)
        run_id_short = (
            request.trace_correlation_id[:8] if request.trace_correlation_id else "unknown"
        )
        _, data = image_data_url.split(",", 1)
        img_path = output_dir / f"seed_{seed}_run_{run_id_short}.png"
        img_path.write_bytes(base64.b64decode(data))
        logger.info("Saved generated image to %s", img_path)

        trace_correlation_id = extract_trace_correlation_id(
            policy_config=policy_cfg,
            inference_url=inference_url,
        )
        trace_payload = build_trace_payload(
            messages=messages,
            response=llm_response,
            correlation_id=trace_correlation_id,
            metadata={
                "run_id": request.trace_correlation_id,
                "seed": seed,
                "card_name": sample["card_name"],
            },
        )

        return RolloutResponse(
            run_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=0.0),
            trace=trace_payload,
            trace_correlation_id=trace_correlation_id,
            inference_url=inference_url,
        )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="MTG artist style prompt optimization local API",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            rubrics=MTG_RUBRICS,
            cors_origins=["*"],
            require_api_key=False,
            ensure_localapi_auth=False,
        )
    )


app = create_mtg_task_app()

