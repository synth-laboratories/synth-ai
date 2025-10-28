from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SFTMessageContent = str | dict[str, Any] | list[Any] | None


class SFTDataError(ValueError):
    """Raised when a JSONL record cannot be coerced into an SFTExample."""


@dataclass(slots=True)
class SFTToolDefinition:
    name: str
    description: str | None
    parameters: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTToolCall:
    name: str
    arguments: Any
    call_id: str | None = None
    type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTMessage:
    role: str
    content: SFTMessageContent
    tool_calls: list[SFTToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None  # NEW: Explicit reasoning/thinking content
    raw_content: str | None = None  # NEW: Original unparsed content (before reasoning extraction)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTExample:
    messages: list[SFTMessage]
    tools: list[SFTToolDefinition] = field(default_factory=list)
    tool_choice: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _parse_tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_tool_definition(raw: Any, *, index: int) -> SFTToolDefinition:
    if not isinstance(raw, dict):
        raise SFTDataError(f"tool {index} is not an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SFTDataError(f"tool {index} missing name")
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise SFTDataError(f"tool {index} description must be a string if present")
    parameters = raw.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise SFTDataError(f"tool {index} parameters must be an object if present")
    return SFTToolDefinition(
        name=name, description=description, parameters=parameters, raw=dict(raw)
    )


def _coerce_tool_call(raw: Any, *, index: int) -> SFTToolCall:
    if not isinstance(raw, dict):
        raise SFTDataError(f"tool_call {index} is not an object")

    call_id = raw.get("id")
    call_type = raw.get("type")

    fn_payload: dict[str, Any] | None = None
    name: str | None = None
    arguments: Any = None

    fn_obj = raw.get("function")
    if isinstance(fn_obj, dict):
        fn_payload = fn_obj
        name_val = fn_payload.get("name")
        name = name_val if isinstance(name_val, str) else None
        arguments = fn_payload.get("arguments")
    if name is None:
        maybe_name = raw.get("name")
        if isinstance(maybe_name, str):
            name = maybe_name
            arguments = raw.get("arguments")

    if not isinstance(name, str) or not name.strip():
        raise SFTDataError(f"tool_call {index} missing function name")

    parsed_arguments = _parse_tool_arguments(arguments)

    normalized_id = None
    if call_id is not None:
        normalized_id = str(call_id)
    normalized_type = None
    if call_type is not None:
        normalized_type = str(call_type)

    return SFTToolCall(
        name=name,
        arguments=parsed_arguments,
        call_id=normalized_id,
        type=normalized_type,
        raw=dict(raw),
    )


def _coerce_message(raw: Any, *, index: int) -> SFTMessage:
    if not isinstance(raw, dict):
        raise SFTDataError(f"message {index} is not an object")
    role = raw.get("role")
    if not isinstance(role, str) or not role.strip():
        raise SFTDataError(f"message {index} has invalid role")

    content = raw.get("content")
    if content is not None and not isinstance(content, str | list | dict):
        raise SFTDataError(f"message {index} has unsupported content type {type(content).__name__}")

    raw_tool_calls = raw.get("tool_calls")
    tool_calls: list[SFTToolCall] = []
    if raw_tool_calls is not None:
        if not isinstance(raw_tool_calls, list | tuple):
            raise SFTDataError(f"message {index} tool_calls must be a list")
        for call_index, call in enumerate(raw_tool_calls):
            tool_calls.append(_coerce_tool_call(call, index=call_index))

    tool_call_id = raw.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        tool_call_id = str(tool_call_id)

    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise SFTDataError(f"message {index} name must be a string if present")
    
    # NEW: Extract reasoning and raw_content if present
    reasoning = raw.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise SFTDataError(f"message {index} reasoning must be a string if present")
    
    raw_content = raw.get("raw_content")
    if raw_content is not None and not isinstance(raw_content, str):
        raise SFTDataError(f"message {index} raw_content must be a string if present")

    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning", "raw_content"}
    }

    return SFTMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
        reasoning=reasoning,
        raw_content=raw_content,
        extra=extra,
    )


def coerce_example(raw: Any, *, min_messages: int = 1) -> SFTExample:
    if not isinstance(raw, dict):
        raise SFTDataError("record is not an object")

    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, Sequence):
        raise SFTDataError("missing messages[] list")
    if len(messages_raw) < min_messages:
        raise SFTDataError(f"missing messages[] with at least {min_messages} turns")

    messages = [_coerce_message(msg, index=i) for i, msg in enumerate(messages_raw)]

    tools: list[SFTToolDefinition] = []
    if "tools" in raw and raw["tools"] is not None:
        tools_raw = raw["tools"]
        if not isinstance(tools_raw, Sequence):
            raise SFTDataError("tools must be provided as a list when present")
        for tool_index, tool in enumerate(tools_raw):
            tools.append(_coerce_tool_definition(tool, index=tool_index))

    tool_choice = raw.get("tool_choice")

    metadata_field = raw.get("metadata")
    metadata: dict[str, Any] = {}
    if metadata_field is not None:
        if not isinstance(metadata_field, dict):
            raise SFTDataError("metadata must be an object if present")
        metadata = dict(metadata_field)

    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"messages", "tools", "tool_choice", "metadata"}
    }

    return SFTExample(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        metadata=metadata,
        extra=extra,
    )


def parse_jsonl_line(line: str, *, min_messages: int = 1) -> SFTExample:
    record = json.loads(line)
    return coerce_example(record, min_messages=min_messages)


def iter_sft_examples(
    source: Iterable[str], *, min_messages: int = 1, skip_empty: bool = True
) -> Iterator[SFTExample]:
    for line in source:
        if skip_empty and not line.strip():
            continue
        yield parse_jsonl_line(line, min_messages=min_messages)


def collect_sft_jsonl_errors(
    path: Path,
    *,
    min_messages: int = 1,
    max_lines: int | None = None,
    max_errors: int | None = None,
) -> list[str]:
    errors: list[str] = []
    lines_checked = 0

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            if max_lines is not None and lines_checked >= max_lines:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            lines_checked += 1
            try:
                parse_jsonl_line(stripped, min_messages=min_messages)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: invalid JSON ({exc.msg})")
            except SFTDataError as exc:
                errors.append(f"Line {lineno}: {exc}")
            if max_errors is not None and len(errors) >= max_errors:
                break
    if lines_checked == 0 and (max_errors is None or len(errors) < max_errors):
        errors.append("File contains no SFT examples")
    return errors


def validate_jsonl_or_raise(
    path: Path,
    *,
    min_messages: int = 1,
    max_lines: int | None = None,
    max_errors: int | None = None,
    error_factory: type[Exception] = ValueError,
) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

    issues = collect_sft_jsonl_errors(
        path,
        min_messages=min_messages,
        max_lines=max_lines,
        max_errors=max_errors,
    )
    if issues:
        truncated = max_errors is not None and len(issues) >= max_errors
        suffix = "" if not truncated else f" (showing first {max_errors} issues)"
        details = "\n - ".join(issues)
        raise error_factory(f"{path}: Dataset validation failed{suffix}:\n - {details}")


def load_jsonl(path: Path, *, min_messages: int = 1) -> list[SFTExample]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as fh:
        return list(iter_sft_examples(fh, min_messages=min_messages))


# Reasoning/Thinking Utilities
# ============================================================================


def extract_reasoning(content: str, *, tag: str = "think") -> tuple[str | None, str]:
    """Extract reasoning from content with <think> tags.
    
    Args:
        content: Raw content string
        tag: Tag name to extract (default: "think")
    
    Returns:
        Tuple of (reasoning, clean_content)
        - reasoning: Content inside tags, or None if no tags found
        - clean_content: Content with tags removed
    
    Examples:
        >>> extract_reasoning("<think>Let me analyze...</think>The answer is 42")
        ('Let me analyze...', 'The answer is 42')
        >>> extract_reasoning("Just plain text")
        (None, 'Just plain text')
    """
    import re
    
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    
    if not matches:
        return None, content
    
    # Combine all reasoning blocks
    reasoning = "\n\n".join(m.strip() for m in matches)
    
    # Remove all reasoning blocks from content
    clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
    
    return reasoning, clean_content


def strip_reasoning(content: str, *, tag: str = "think") -> str:
    """Remove reasoning tags from content.
    
    Args:
        content: Content with potential reasoning tags
        tag: Tag name to strip (default: "think")
    
    Returns:
        Content with reasoning tags removed
    """
    _, clean = extract_reasoning(content, tag=tag)
    return clean


def message_has_reasoning(message: SFTMessage) -> bool:
    """Check if a message has explicit reasoning.
    
    Args:
        message: SFTMessage to check
    
    Returns:
        True if message has reasoning field or <think> tags in content
    """
    # Check explicit reasoning field
    if message.reasoning:
        return True
    
    # Check for reasoning tags in content
    if isinstance(message.content, str):
        reasoning, _ = extract_reasoning(message.content)
        return reasoning is not None
    
    return False


def validate_message_content(
    message: SFTMessage, *, require_content: bool = True
) -> tuple[bool, str | None]:
    """Validate that message has valid content combinations.
    
    Rules:
    - Must have at least one of: reasoning + tool_calls, reasoning + content, 
      content, raw_content, or tool_calls
    - If raw_content present with reasoning + content, they should be consistent
    - Cannot have neither reasoning, content, raw_content, nor tool_calls
    
    Args:
        message: SFTMessage to validate
        require_content: If True, require some form of content (default: True)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    has_reasoning = bool(message.reasoning)
    has_content = message.content is not None and message.content != ""
    has_raw = bool(message.raw_content)
    has_tools = len(message.tool_calls) > 0
    
    # Check for completely empty message
    if require_content and not (has_reasoning or has_content or has_raw or has_tools):
        return False, "Message has no reasoning, content, raw_content, or tool_calls"
    
    # Valid combinations:
    # 1. reasoning + tool_calls (reasoning-based action)
    if has_reasoning and has_tools:
        return True, None
    
    # 2. reasoning + content (reasoning then output)
    if has_reasoning and has_content:
        # If raw_content present, validate consistency
        if has_raw and message.raw_content:
            # Raw should contain both reasoning and content
            reasoning_in_raw, content_in_raw = extract_reasoning(message.raw_content)
            if message.reasoning and reasoning_in_raw != message.reasoning.strip():
                logger.warning(
                    "raw_content reasoning doesn't match reasoning field"
                )
            # This is okay - just a warning, not an error
        return True, None
    
    # 3. content only (standard message)
    if has_content and not has_reasoning:
        return True, None
    
    # 4. raw_content only (unparsed content)
    if has_raw and not (has_reasoning and has_content):
        return True, None
    
    # 5. tool_calls only (action without reasoning/content - like OpenAI format)
    if has_tools and not has_content:
        return True, None
    
    # 6. reasoning only (pure thinking turn)
    if has_reasoning and not has_content and not has_tools:
        return True, None
    
    return True, None


# Vision/Multimodal Utilities
# ============================================================================


def has_image_content(content: SFTMessageContent) -> bool:
    """Check if message content contains image data (OpenAI multimodal format).
    
    Supports:
    - List of content parts: [{"type": "text", ...}, {"type": "image_url", ...}]
    - Single dict with type field: {"type": "image_url", "image_url": {...}}
    
    Args:
        content: Message content (can be str, list, dict, or None)
    
    Returns:
        True if content contains an image segment
    
    Examples:
        >>> has_image_content([{"type": "text", "text": "What's this?"}, 
        ...                    {"type": "image_url", "image_url": {"url": "..."}}])
        True
        >>> has_image_content("Just text")
        False
    """
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and part.get("type") in {"image", "image_url"}
            for part in content
        )
    elif isinstance(content, dict):
        return content.get("type") in {"image", "image_url"}
    return False


def message_has_image(message: SFTMessage) -> bool:
    """Check if an SFTMessage contains image content.
    
    Args:
        message: SFTMessage to check
    
    Returns:
        True if the message contains image content
    """
    return has_image_content(message.content)


def example_has_image(example: SFTExample) -> bool:
    """Check if an SFTExample contains any image content.
    
    Args:
        example: SFTExample to check
    
    Returns:
        True if any message in the example contains image content
    """
    return any(message_has_image(msg) for msg in example.messages)


def count_images_in_content(content: SFTMessageContent) -> int:
    """Count the number of images in message content.
    
    Args:
        content: Message content to analyze
    
    Returns:
        Number of image segments found
    """
    if isinstance(content, list):
        return sum(
            1 for part in content
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}
        )
    elif isinstance(content, dict) and content.get("type") in {"image", "image_url"}:
        return 1
    return 0


def extract_image_urls(content: SFTMessageContent) -> list[str]:
    """Extract all image URLs from message content.
    
    Filters out invalid entries:
    - Non-string URLs
    - Empty strings
    - Whitespace-only strings
    
    Args:
        content: Message content to extract from
    
    Returns:
        List of valid image URL strings (may be http(s):// URLs or data:image/... base64)
    """
    urls: list[str] = []
    
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                # Handle both formats:
                # {"type": "image_url", "image_url": {"url": "..."}}
                # {"type": "image", "image": "..."}
                if "image_url" in part and isinstance(part["image_url"], dict):
                    url = part["image_url"].get("url")
                    if isinstance(url, str) and url.strip():  # Filter empty/whitespace
                        urls.append(url)
                elif "image" in part and isinstance(part["image"], str):
                    if part["image"].strip():  # Filter empty/whitespace
                        urls.append(part["image"])
    elif isinstance(content, dict) and content.get("type") in {"image", "image_url"}:
        image_url_data = content.get("image_url")
        if isinstance(image_url_data, dict):
            url = image_url_data.get("url")
            if isinstance(url, str) and url.strip():  # Filter empty/whitespace
                urls.append(url)
        else:
            image_value = content.get("image")
            if isinstance(image_value, str) and image_value.strip():  # Filter empty/whitespace
                urls.append(image_value)
    
    return urls


def validate_vision_example(
    example: SFTExample, *, require_images: bool = True
) -> tuple[bool, str | None]:
    """Validate a vision SFT example.
    
    Checks:
    - If require_images is True, at least one message must contain an image
    - All image URLs must be non-empty, non-whitespace strings
    - Image entries must have valid URL data
    - Messages must follow valid structure
    
    Args:
        example: SFTExample to validate
        require_images: If True, fail if no images are present
    
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Count actual valid URLs and detect any invalid entries
    total_valid_urls = 0
    
    # Validate image URLs in each message
    for i, msg in enumerate(example.messages):
        # Check if this message has image_url type entries
        if not isinstance(msg.content, list | dict):
            continue
        
        # Count image_url type entries vs valid URLs
        content_list = msg.content if isinstance(msg.content, list) else [msg.content]
        image_type_count = sum(
            1 for item in content_list
            if isinstance(item, dict) and item.get("type") in {"image", "image_url"}
        )
        
        if image_type_count > 0:
            # Extract valid URLs (after filtering)
            urls = extract_image_urls(msg.content)
            
            # If we have image_url type entries but fewer valid URLs, some are invalid
            if len(urls) < image_type_count:
                return False, f"Message {i}: Has {image_type_count} image_url entries but only {len(urls)} valid URLs (some are empty, null, or missing)"
            
            # Validate each URL (double-check, though extract_image_urls should have filtered)
            for url in urls:
                # extract_image_urls already filters for isinstance(url, str) and url.strip()
                # but let's be defensive
                if not isinstance(url, str):
                    return False, f"Message {i}: Image URL is not a string: {type(url)}"
                
                if not url.strip():
                    return False, f"Message {i}: Invalid or empty image URL"
                
                # Basic URL format check
                if not url.startswith(("http://", "https://", "data:image/")):
                    logger.warning(
                        f"Message {i}: Image URL doesn't start with http://, https://, or data:image/ - "
                        f"this may cause issues during training. URL: {url[:100]}"
                    )
                
                total_valid_urls += 1
    
    # Final check: if images are required, ensure we found at least one valid URL
    if require_images and total_valid_urls == 0:
        return False, "No image content found in any message"
    
    return True, None


def iter_vision_examples(
    source: Iterable[str],
    *,
    min_messages: int = 1,
    skip_empty: bool = True,
    require_images: bool = True,
    log_validation_errors: bool = False,
) -> Iterator[SFTExample]:
    """Iterate over vision SFT examples from JSONL source.
    
    Similar to iter_sft_examples but with vision-specific validation.
    
    Args:
        source: Iterable of JSONL lines
        min_messages: Minimum number of messages required
        skip_empty: Skip empty lines
        require_images: If True, skip examples without images
        log_validation_errors: If True, log validation failures
    
    Yields:
        Valid vision SFTExample objects
    """
    for line in source:
        if skip_empty and not line.strip():
            continue
        
        try:
            example = parse_jsonl_line(line, min_messages=min_messages)
            
            # Validate vision content if required
            if require_images:
                is_valid, error = validate_vision_example(example, require_images=True)
                if not is_valid:
                    if log_validation_errors:
                        logger.warning(f"Skipping invalid vision example: {error}")
                    continue
            
            yield example
            
        except (json.JSONDecodeError, SFTDataError) as exc:
            if log_validation_errors:
                logger.warning(f"Failed to parse vision example: {exc}")
            continue


__all__ = [
    "SFTDataError",
    "SFTExample",
    "SFTMessage",
    "SFTToolCall",
    "SFTToolDefinition",
    "collect_sft_jsonl_errors",
    "coerce_example",
    "iter_sft_examples",
    "load_jsonl",
    "parse_jsonl_line",
    "validate_jsonl_or_raise",
    # Reasoning utilities
    "extract_reasoning",
    "strip_reasoning",
    "message_has_reasoning",
    "validate_message_content",
    # Vision utilities
    "has_image_content",
    "message_has_image",
    "example_has_image",
    "count_images_in_content",
    "extract_image_urls",
    "validate_vision_example",
    "iter_vision_examples",
]
