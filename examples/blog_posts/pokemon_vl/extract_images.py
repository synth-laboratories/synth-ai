#!/usr/bin/env python3
"""Extract images from pokemon_vl trace database or trace JSON file and save to images_gpt5 directory.

Usage:
    # From trace database:
    python extract_images.py --trace-db traces/v3/pokemon_vl_gpt5nano.db

    # From trace JSON file:
    python extract_images.py --trace-json trace.json
"""

import argparse
import base64
import json
import sqlite3
from pathlib import Path
from typing import Any

from synth_ai.tracing_v3.trace_utils import load_session_trace


def extract_image_urls_from_content(content: Any) -> list[str]:
    """Extract image URLs from message content."""
    urls = []
    
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" and "image_url" in part:
                    url = part["image_url"].get("url")
                    if isinstance(url, str) and url.startswith("data:image"):
                        urls.append(url)
                elif part.get("type") == "image":
                    img = part.get("image")
                    if isinstance(img, str) and img.startswith("data:image"):
                        urls.append(img)
    elif isinstance(content, str):
        # Check if it's a JSON string
        try:
            parsed = json.loads(content)
            return extract_image_urls_from_content(parsed)
        except:
            pass
    
    return urls


def extract_state_info_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """Extract state info from message metadata or content."""
    metadata = message.get("metadata", {})
    state = {}
    
    # Try to get state from metadata
    if "system_state_before" in metadata:
        state_before = metadata["system_state_before"]
        if isinstance(state_before, dict):
            obs = state_before.get("obs", {})
            state.update({
                "position": obs.get("position", "?"),
                "map_id": obs.get("map_id", "?"),
                "player_x": obs.get("player_x", "?"),
                "player_y": obs.get("player_y", "?"),
                "text_box_active": obs.get("text_box_active", False),
            })
    
    # Try to extract from content text
    content = message.get("content", "")
    if isinstance(content, str) and "position" in content:
        try:
            # Look for state summary in content
            if "State summary:" in content:
                parts = content.split("State summary:")
                if len(parts) > 1:
                    import ast
                    state_str = parts[1].split("'")[0] if "'" not in parts[1] else parts[1]
                    try:
                        state_dict = ast.literal_eval(state_str.split("'")[0] if "'" in state_str else state_str)
                        if isinstance(state_dict, dict):
                            state.update({
                                "position": state_dict.get("position", "?"),
                                "map_id": state_dict.get("map_id", "?"),
                                "player_x": state_dict.get("player_x", "?"),
                                "player_y": state_dict.get("player_y", "?"),
                                "text_box_active": state_dict.get("text_box_active", False),
                            })
                    except:
                        pass
        except:
            pass
    
    return state


def extract_images_from_trace_dict(trace: dict[str, Any], output_dir: Path):
    """Extract images from a trace dictionary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get messages from trace
    messages = trace.get("markov_blanket_message_history", []) or trace.get("messages", [])
    
    if not messages:
        print(f"  No messages found in trace")
        return 0
    
    print(f"  Found {len(messages)} messages")
    
    image_count = 0
    step_idx = 0
    for msg_idx, msg in enumerate(messages):
        # Extract images from message content
        content = msg.get("content", "")
        image_urls = extract_image_urls_from_content(content)
        
        if not image_urls:
            continue
        
        # Extract state info for filename
        state = extract_state_info_from_message(msg)
        
        for img_idx, img_url in enumerate(image_urls):
            # Extract base64 data
            if img_url.startswith("data:image"):
                # Format: data:image/png;base64,<data>
                parts = img_url.split(",", 1)
                if len(parts) != 2:
                    continue
                
                b64_data = parts[1]
                try:
                    img_data = base64.b64decode(b64_data)
                    
                    # Create filename
                    pos_str = f"{state.get('map_id', '?')}_{state.get('player_x', '?')},{state.get('player_y', '?')}"
                    textbox_str = "True" if state.get("text_box_active") else "False"
                    filename = f"step_{step_idx:03d}_pos_{pos_str}_textbox_{textbox_str}.png"
                    
                    filepath = output_dir / filename
                    filepath.write_bytes(img_data)
                    
                    print(f"  Saved: {filename}")
                    image_count += 1
                    step_idx += 1
                except Exception as e:
                    print(f"  Error decoding image: {e}")
                    continue
    
    return image_count


def extract_images_from_trace_db(trace_db: str, output_dir: Path, model_filter: str | None = None):
    """Extract images from trace database and save to output directory."""
    conn = sqlite3.connect(trace_db)
    conn.row_factory = sqlite3.Row
    
    # Get all session IDs
    query = "SELECT session_id, metadata FROM session_traces"
    if model_filter:
        query += " WHERE metadata LIKE ?"
        params = (f'%{model_filter}%',)
    else:
        params = ()
    
    rows = conn.execute(query, params).fetchall()
    
    if not rows:
        print(f"No traces found in {trace_db}")
        return
    
    print(f"Found {len(rows)} trace(s)")
    
    total_images = 0
    for row in rows:
        session_id = row["session_id"]
        print(f"\nProcessing session: {session_id}")
        
        try:
            trace = load_session_trace(conn, session_id)
        except Exception as e:
            print(f"  Error loading trace: {e}")
            continue
        
        count = extract_images_from_trace_dict(trace, output_dir)
        total_images += count
    
    conn.close()
    print(f"\n✓ Extracted {total_images} images to {output_dir}/")


def extract_images_from_trace_json(trace_json: Path, output_dir: Path):
    """Extract images from trace JSON file."""
    print(f"Loading trace from {trace_json}")
    
    with open(trace_json) as f:
        trace = json.load(f)
    
    # Handle trace wrapped in "session_trace" key
    if "session_trace" in trace:
        trace = trace["session_trace"]
    
    count = extract_images_from_trace_dict(trace, output_dir)
    print(f"\n✓ Extracted {count} images to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-db",
        help="Path to trace database",
    )
    parser.add_argument(
        "--trace-json",
        type=Path,
        help="Path to trace JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/blog_posts/pokemon_vl/images_gpt5",
        help="Output directory for images",
    )
    parser.add_argument(
        "--model-filter",
        help="Filter traces by model name (optional)",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.trace_json:
        extract_images_from_trace_json(args.trace_json, output_dir)
    elif args.trace_db:
        extract_images_from_trace_db(args.trace_db, output_dir, args.model_filter)
    else:
        parser.error("Must provide either --trace-db or --trace-json")


if __name__ == "__main__":
    main()


