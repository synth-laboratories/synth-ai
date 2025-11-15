"""Shared helpers for GEPA benchmark task apps (HotpotQA, IFBench, HoVer, PUPA)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import httpx
from fastapi import HTTPException


def _resolve_inference_url(base_url: str) -> str:
    """Normalise a base inference URL to the chat completions endpoint.
    
    Handles:
    - Standard OpenAI/Groq URLs: https://api.openai.com/v1 -> .../v1/chat/completions
    - Interceptor URLs (GEPA): https://...modal.host/v1/gepa-... -> .../v1/gepa-.../chat/completions
    - URLs with query parameters: .../v1/trial-id?cid=trace_... -> .../v1/trial-id/chat/completions?cid=trace_...
    - Already complete URLs: .../chat/completions -> unchanged
    """
    from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

    normalised = (base_url or "").rstrip("/")
    if not normalised:
        raise RuntimeError("policy.config.inference_url required")
    
    # Parse URL to separate path from query parameters
    parsed = urlparse(normalised)
    path = parsed.path.rstrip("/")
    query = parsed.query
    fragment = parsed.fragment
    
    # Debug: Always log the input URL
    print(f"[RESOLVE_URL] Input: {base_url} -> path={path}", flush=True)
    
    # Already complete
    if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
        print(f"[RESOLVE_URL] Already complete: {normalised}", flush=True)
        return normalised
    
    # Check if this looks like an interceptor URL with trial_id
    # Interceptor URLs have /v1/ followed by an identifier (e.g., /v1/gepa-..., /v1/pl-..., /v1/baseline-...)
    # These URLs already have /v1/{trial_id} in them, so we should append /chat/completions
    # We detect this by checking if the URL contains /v1/ followed by something (not just ending with /v1)
    if "/v1/" in path and not path.endswith("/v1"):
        # Check if it already ends with /chat/completions
        if path.endswith("/chat/completions"):
            print(f"[RESOLVE_URL] Already has /chat/completions: {normalised}", flush=True)
            return normalised
        # This is likely an interceptor URL with trial_id - append /chat/completions to path
        new_path = f"{path}/chat/completions"
        # Reconstruct URL with query parameters preserved
        result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
        print(f"[RESOLVE_URL] Interceptor URL with trial_id: {base_url} -> {result}", flush=True)
        return result
    
    # Standard case: append /v1/chat/completions
    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    else:
        new_path = f"{path}/v1/chat/completions"
    
    # Reconstruct URL with query parameters preserved
    result = urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, query, fragment))
    print(f"[RESOLVE_URL] Standard URL: {base_url} -> {result}", flush=True)
    return result


_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


def _substitute_placeholders(text: str, values: Mapping[str, Any]) -> str:
    """Replace `{placeholder}` tokens in `text` with entries from `values`."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        replacement = values.get(key)
        return str(replacement) if replacement is not None else match.group(0)

    return _PLACEHOLDER_PATTERN.sub(_replace, text)


def render_messages(
    policy_config: Mapping[str, Any],
    placeholders: Mapping[str, Any],
    default_messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Render chat messages either from policy prompt patterns or defaults."""

    prompt_config = policy_config.get("prompt") if isinstance(policy_config, Mapping) else None
    rendered: list[dict[str, str]] = []
    if prompt_config and isinstance(prompt_config, Mapping):
        messages = prompt_config.get("messages")
        if isinstance(messages, Sequence):
            for entry in messages:
                if not isinstance(entry, Mapping):
                    continue
                role = str(entry.get("role") or "user")
                pattern = entry.get("pattern") or entry.get("content") or ""
                content = _substitute_placeholders(str(pattern), placeholders)
                rendered.append({"role": role, "content": content})
    if not rendered:
        for entry in default_messages:
            role = str(entry.get("role") or "user")
            pattern = entry.get("pattern") or entry.get("content") or ""
            content = _substitute_placeholders(str(pattern), placeholders)
            rendered.append({"role": role, "content": content})
    return rendered


async def call_chat_completion(
    policy_config: Mapping[str, Any],
    placeholders: Mapping[str, Any],
    default_messages: Sequence[Mapping[str, str]],
    *,
    tool_spec: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Invoke an OpenAI-compatible chat/completions endpoint.

    Returns:
        response_text: The assistant message text (empty string if missing).
        raw_response: The JSON payload from the provider.
        messages: The messages sent to the model (after placeholder substitution).
    """

    if not isinstance(policy_config, Mapping):
        raise RuntimeError("policy.config must be a mapping for chat completion calls")

    messages = render_messages(policy_config, placeholders, default_messages)

    model = policy_config.get("model")
    if not model:
        raise RuntimeError("policy.config.model required for rollout")

    temperature = policy_config.get("temperature", 0.0)
    max_tokens = policy_config.get("max_tokens")
    max_completion_tokens = policy_config.get("max_completion_tokens", max_tokens or 512)

    inference_url = policy_config.get("inference_url") or ""
    final_url = _resolve_inference_url(str(inference_url))

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    # Add reasoning_effort if specified (for o1 models)
    if "reasoning_effort" in policy_config:
        payload["reasoning_effort"] = policy_config["reasoning_effort"]
    if tool_spec:
        payload["tools"] = list(tool_spec)
    if tool_choice:
        payload["tool_choice"] = tool_choice

    # Choose API key based on inference URL
    # Prefer provider-specific keys matching the URL, fall back to OPENAI/SYNTH
    api_key = None
    final_url_lower = final_url.lower()
    if "groq" in final_url_lower:
        api_key = os.getenv("GROQ_API_KEY")
    elif "openai.com" in final_url_lower or "api.openai.com" in final_url_lower:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        # Fallback: try OPENAI first, then SYNTH, then GROQ
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SYNTH_API_KEY") or os.getenv("GROQ_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        response = await client.post(final_url, json=payload, headers=headers)

    try:
        data = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=502,
            detail=f"Inference provider returned invalid JSON: {response.text[:800]}",
        ) from exc

    if response.status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail=f"Inference provider returned an error: {data}",
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid inference request: {data}",
        )

    response_text = ""
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if isinstance(choices, Sequence) and choices:
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            response_text = str(message.get("content") or "")

    return response_text, data, messages


def normalise_answer(text: str) -> str:
    """Normalise free-form text answers (HotpotQA style)."""

    lowered = text.lower()
    # Remove punctuation and articles.
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    cleaned = re.sub(r"\b(a|an|the)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)


def count_emojis(text: str) -> int:
    """Return rough count of emoji characters."""

    return len(_EMOJI_PATTERN.findall(text))


def tokenize(text: str) -> list[str]:
    """Simple whitespace/token splitter with punctuation stripping."""

    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [token for token in cleaned.split() if token]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def count_numbers(text: str) -> int:
    """Count occurrences of numeric tokens."""

    return len(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def unique_word_count(tokens: Iterable[str]) -> int:
    """Return number of unique tokens."""

    return len(set(tokens))


PRONOUNS = {
    "i",
    "me",
    "you",
    "he",
    "him",
    "she",
    "her",
    "it",
    "we",
    "us",
    "they",
    "them",
    "my",
    "mine",
    "your",
    "yours",
    "his",
    "hers",
    "its",
    "our",
    "ours",
    "their",
    "theirs",
}


def count_pronouns(tokens: Iterable[str]) -> int:
    """Count pronoun tokens from a predefined list."""

    return sum(1 for token in tokens if token in PRONOUNS)


__all__ = [
    "call_chat_completion",
    "count_emojis",
    "count_numbers",
    "count_pronouns",
    "normalise_answer",
    "render_messages",
    "sentence_split",
    "tokenize",
    "unique_word_count",
]
