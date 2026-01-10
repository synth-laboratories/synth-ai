#!/usr/bin/env python3
"""Analyze verifier payload to understand token consumption breakdown."""

import json
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return len(text) // 4


def analyze_size(obj: Any, name: str, depth: int = 0) -> dict:
    """Recursively analyze object size and structure."""
    indent = "  " * depth

    if isinstance(obj, str):
        size = len(obj)
        tokens = estimate_tokens(obj)
        is_image = obj.startswith("data:image/")

        result = {
            "name": name,
            "type": "string",
            "bytes": size,
            "tokens": tokens,
            "is_image": is_image,
            "preview": obj[:100] if not is_image else f"[IMAGE: {size:,} bytes]",
        }

        print(
            f"{indent}{name}: {size:,} bytes (~{tokens:,} tokens) {'⚠️  IMAGE' if is_image else ''}"
        )
        if not is_image and size > 1000:
            print(f"{indent}  Preview: {obj[:200]}...")

        return result

    elif isinstance(obj, list):
        total_bytes = 0
        total_tokens = 0
        items = []

        print(f"{indent}{name}: list with {len(obj)} items")
        for i, item in enumerate(obj):
            result = analyze_size(item, f"[{i}]", depth + 1)
            items.append(result)
            total_bytes += result.get("bytes", 0)
            total_tokens += result.get("tokens", 0)

        return {
            "name": name,
            "type": "list",
            "count": len(obj),
            "bytes": total_bytes,
            "tokens": total_tokens,
            "items": items,
        }

    elif isinstance(obj, dict):
        total_bytes = 0
        total_tokens = 0
        fields = {}

        print(f"{indent}{name}: dict with {len(obj)} keys: {list(obj.keys())}")
        for key, value in obj.items():
            result = analyze_size(value, key, depth + 1)
            fields[key] = result
            total_bytes += result.get("bytes", 0)
            total_tokens += result.get("tokens", 0)

        return {
            "name": name,
            "type": "dict",
            "keys": list(obj.keys()),
            "bytes": total_bytes,
            "tokens": total_tokens,
            "fields": fields,
        }

    else:
        text = str(obj)
        size = len(text)
        tokens = estimate_tokens(text)
        print(f"{indent}{name}: {type(obj).__name__} = {text[:100]}")
        return {
            "name": name,
            "type": type(obj).__name__,
            "bytes": size,
            "tokens": tokens,
            "value": obj,
        }


def analyze_trace_file(file_path: Path):
    """Analyze a captured trace file."""
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {file_path}")
    print(f"{'=' * 80}\n")

    data = json.loads(file_path.read_text())

    # Overall breakdown
    print("\n" + "=" * 80)
    print("TOP-LEVEL BREAKDOWN")
    print("=" * 80)

    breakdown = analyze_size(data, "verifier_payload", 0)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total payload bytes: {breakdown['bytes']:,}")
    print(f"Total estimated tokens: {breakdown['tokens']:,}")
    print(f"Gemini 2.5 Flash limit: 1,048,576 tokens")

    if breakdown["tokens"] > 1_048_576:
        print(f"⚠️  EXCEEDS LIMIT by {breakdown['tokens'] - 1_048_576:,} tokens")
    else:
        print(f"✓ Within limit ({1_048_576 - breakdown['tokens']:,} tokens remaining)")

    # Breakdown by major components
    print(f"\n{'=' * 80}")
    print("COMPONENT BREAKDOWN")
    print(f"{'=' * 80}")

    if "fields" in breakdown:
        components = []
        for key, field in breakdown["fields"].items():
            components.append((key, field.get("bytes", 0), field.get("tokens", 0)))

        components.sort(key=lambda x: x[1], reverse=True)

        for key, bytes_val, tokens_val in components:
            pct = (bytes_val / breakdown["bytes"] * 100) if breakdown["bytes"] > 0 else 0
            print(f"{key:30} {bytes_val:12,} bytes  {tokens_val:12,} tokens  {pct:5.1f}%")

    # Look for images
    print(f"\n{'=' * 80}")
    print("IMAGE DETECTION")
    print(f"{'=' * 80}")

    def find_images(obj, path=""):
        """Recursively find all images."""
        images = []

        if isinstance(obj, str) and obj.startswith("data:image/"):
            images.append((path, len(obj)))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                images.extend(find_images(value, f"{path}.{key}" if path else key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                images.extend(find_images(item, f"{path}[{i}]"))

        return images

    images = find_images(data)
    if images:
        print(f"Found {len(images)} images:")
        for path, size in images:
            print(f"  {path}: {size:,} bytes (~{estimate_tokens(str(size)):,} tokens)")
    else:
        print("✓ No base64 images found in payload")

    # Event history breakdown
    if "fields" in breakdown and "trace" in breakdown["fields"]:
        trace_fields = breakdown["fields"]["trace"].get("fields", {})
        if "event_history" in trace_fields:
            event_history = trace_fields["event_history"]
            print(f"\n{'=' * 80}")
            print(f"EVENT HISTORY ({event_history.get('count', 0)} events)")
            print(f"{'=' * 80}")

            if "items" in event_history:
                for i, event in enumerate(event_history["items"]):
                    event_bytes = event.get("bytes", 0)
                    event_tokens = event.get("tokens", 0)
                    print(f"Event {i}: {event_bytes:,} bytes (~{event_tokens:,} tokens)")

                    # Show breakdown
                    if "fields" in event:
                        for key, field in event["fields"].items():
                            field_bytes = field.get("bytes", 0)
                            if field_bytes > 1000:
                                print(f"  {key}: {field_bytes:,} bytes")


if __name__ == "__main__":
    # Check for captured trace
    trace_file = Path("/tmp/captured_trace.json")
    if not trace_file.exists():
        print(f"ERROR: No captured trace found at {trace_file}")
        print("Run the demo with trace capture enabled first.")
        exit(1)

    analyze_trace_file(trace_file)
