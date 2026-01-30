"""MTG artist style image generation task app (import-safe).

No side effects at import time.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import httpx
from synth_ai.sdk.localapi import create_local_api
from synth_ai.sdk.localapi.helpers import extract_api_key, normalize_chat_completion_url
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

_LOGGED_POLICY_KEYS = False

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


def _extract_image_data_url(llm_response: dict[str, Any]) -> str | None:
    choices = llm_response.get("choices") or []
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url" or "image_url" in part:
                        image_url = part.get("image_url", {}).get("url", "")
                        if isinstance(image_url, str) and image_url.startswith("data:image/"):
                            return image_url
        elif isinstance(content, str) and content.startswith("data:image/"):
            return content
        image_url = message.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
            if isinstance(url, str) and url.startswith("data:image/"):
                return url

    data_items = llm_response.get("data")
    if isinstance(data_items, list):
        for item in data_items:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("image_url")
            if isinstance(url, str) and url.startswith("data:image/"):
                return url
            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64:
                mime = item.get("mime_type") or item.get("media_type") or "image/png"
                return f"data:{mime};base64,{b64}"

    return None


def _scrub_response_for_log(value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    if depth >= max_depth:
        return "<truncated>"
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"b64_json", "data", "inline_data", "image", "images"}:
                scrubbed[key] = "<omitted>"
            else:
                scrubbed[key] = _scrub_response_for_log(
                    item, depth=depth + 1, max_depth=max_depth
                )
        return scrubbed
    if isinstance(value, list):
        return [
            _scrub_response_for_log(item, depth=depth + 1, max_depth=max_depth)
            for item in value[:8]
        ]
    if isinstance(value, str):
        if len(value) > 400:
            return value[:400] + "…"
        return value
    return value


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
        policy_cfg.setdefault("model", "gemini-2.5-flash-image")
        global _LOGGED_POLICY_KEYS
        if not _LOGGED_POLICY_KEYS:
            logger.info("Policy config keys: %s", list(policy_cfg.keys()))
            logger.info("Policy inference_url: %s", policy_cfg.get("inference_url"))
            logger.info(
                "Policy model/provider: model=%s provider=%s",
                policy_cfg.get("model"),
                policy_cfg.get("provider"),
            )
            _LOGGED_POLICY_KEYS = True
        provider = str(
            policy_cfg.get("provider")
            or os.environ.get("MTG_POLICY_PROVIDER", "google")
        ).lower()
        inference_url = str(policy_cfg.get("inference_url") or "")

        env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
        synth_api_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
        gemini_api_key = (
            (os.environ.get("GEMINI_API_KEY") or "").strip()
            or (os.environ.get("GOOGLE_API_KEY") or "").strip()
        )
        default_keys: dict[str, str] = {}
        if env_api_key:
            default_keys["ENVIRONMENT_API_KEY"] = env_api_key
        if synth_api_key:
            default_keys["SYNTH_API_KEY"] = synth_api_key
        api_key = extract_api_key(
            fastapi_request,
            policy_cfg,
            default_env_keys=default_keys or None,
        )
        if synth_api_key:
            api_key = synth_api_key
            logger.info("Using SYNTH_API_KEY for inference interceptor")
        elif env_api_key:
            api_key = env_api_key
            logger.info("Using ENVIRONMENT_API_KEY for inference interceptor")
        if provider != "google":
            raise ValueError("MTG image demo requires provider=google (Gemini)")
        if not inference_url:
            raise ValueError("inference_url is required in policy config")
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini")
        if not api_key:
            raise RuntimeError(
                "Missing API key for inference interceptor (ENVIRONMENT_API_KEY or request headers)."
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

        endpoint = normalize_chat_completion_url(str(inference_url))
        payload: dict[str, Any] = {
            "model": policy_cfg["model"],
            "messages": messages,
            "temperature": policy_cfg.get("temperature", 0.7),
        }
        if "max_completion_tokens" in policy_cfg:
            payload["max_completion_tokens"] = policy_cfg.get("max_completion_tokens")

        max_attempts = int(os.environ.get("MTG_POLICY_RETRY_ATTEMPTS", "3"))
        backoff_s = float(os.environ.get("MTG_POLICY_RETRY_BACKOFF_S", "1.5"))
        retryable = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None

        debug_raw = os.environ.get("MTG_DEBUG_RAW_POLICY_RESPONSE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if debug_raw:
            logger.error(
                "Policy request debug: inference_url=%s endpoint=%s model=%s provider=%s",
                inference_url,
                endpoint,
                policy_cfg.get("model"),
                policy_cfg.get("provider"),
            )
        async with httpx.AsyncClient(timeout=180.0) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    if debug_raw:
                        logger.error(
                            "Policy request attempt %s/%s to %s",
                            attempt,
                            max_attempts,
                            endpoint,
                        )
                    resp = await client.post(
                        endpoint,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "X-API-Key": api_key,
                            "x-goog-api-key": gemini_api_key,
                            "Content-Type": "application/json",
                        },
                    )
                    resp.raise_for_status()
                    llm_response = resp.json()
                    if debug_raw:
                        logger.error(
                            "Policy response status=%s keys=%s",
                            resp.status_code,
                            list(llm_response.keys()),
                        )
                    break
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response else None
                    body = exc.response.text if exc.response else "no-body"
                    if status in retryable and attempt < max_attempts:
                        logger.warning(
                            "Interceptor error (attempt %s/%s): status=%s body=%s",
                            attempt,
                            max_attempts,
                            status,
                            body[:500],
                        )
                        await asyncio.sleep(backoff_s * attempt)
                        continue
                    logger.error(
                        "Interceptor error: status=%s body=%s",
                        status if status is not None else "unknown",
                        body[:500],
                    )
                    if debug_raw:
                        logger.error("Policy request failed at status=%s", status)
                    last_exc = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.exception("Interceptor call failed")
                    if debug_raw:
                        logger.error("Policy request exception: %s", exc)
                    break

        if last_exc is not None:
            raise last_exc

        image_data_url = _extract_image_data_url(llm_response)

        if not image_data_url:
            choices = llm_response.get("choices", [])
            logger.error(
                "No image data in policy response. message_content=%s response=%s",
                (choices[0].get("message", {}).get("content") if choices else None),
                _scrub_response_for_log(llm_response),
            )
            if debug_raw:
                try:
                    logger.error("Raw policy response: %s", json.dumps(llm_response))
                except Exception:
                    logger.error("Raw policy response (repr): %r", llm_response)
            raise ValueError("No image data in policy model response")

        save_images = os.environ.get("MTG_SAVE_IMAGES", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if save_images:
            output_dir = demo_dir / "generated_images"
            output_dir.mkdir(exist_ok=True)
            run_id_short = (
                request.trace_correlation_id[:8] if request.trace_correlation_id else "unknown"
            )
            _, data = image_data_url.split(",", 1)
            img_path = output_dir / f"seed_{seed}_run_{run_id_short}.png"
            img_path.write_bytes(base64.b64decode(data))
            logger.info("Saved generated image to %s", img_path)
        else:
            logger.info("Skipping image write (MTG_SAVE_IMAGES=0)")

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
