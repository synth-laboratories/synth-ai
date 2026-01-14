#!/usr/bin/env python3
"""
Test using the Responses API format (what OpenCode uses).
This is different from Chat Completions API.
"""

import json
import os

import httpx

# OpenCode uses /responses endpoint
DIRECT_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o"

# Same tool schema but in Responses API format
TOOLS = [
    {
        "type": "function",
        "name": "read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string", "description": "Path to file"}},
            "required": ["filePath"],
        },
    },
    {
        "type": "function",
        "name": "edit",
        "description": "Edit a file by replacing old_string with new_string",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "Path to file"},
                "old_string": {"type": "string", "description": "Text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["filePath", "old_string", "new_string"],
        },
    },
]

FILE_CONTENT = """pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }

pub const ATTACK_1_TEXT: &str = "Does 10 damage times the amount of Energy attached to Tropius.";
"""

SYSTEM_PROMPT = """CRITICAL: Replace todo!() with working code using the edit tool.

If you see:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }
```

Use edit tool to change it to:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { (10 * attached_energy) as i32 }
```

DO NOT just read the file. You MUST call the edit tool.
"""


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Need OPENAI_API_KEY")
        return

    # Responses API uses "input" array with different role names
    input_messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Edit this file to implement grind_damage:\n{FILE_CONTENT}"},
    ]

    print("=" * 60)
    print("RESPONSES API TEST (like OpenCode)")
    print(f"URL: {DIRECT_URL}")
    print(f"Model: {MODEL}")
    print("=" * 60)

    for turn in range(3):
        print(f"\n--- Turn {turn + 1} ---")

        payload = {"model": MODEL, "input": input_messages, "tools": TOOLS, "tool_choice": "auto"}

        print(f"Sending {len(input_messages)} messages...")

        try:
            resp = httpx.post(
                DIRECT_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text[:500]}")
                return
            data = resp.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return

        # Responses API has different output format
        output = data.get("output", [])
        print(f"Output items: {len(output)}")

        for item in output:
            item_type = item.get("type")
            print(f"  Item type: {item_type}")

            if item_type == "message":
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "text":
                        print(f"    Text: {c.get('text', '')[:100]}...")

            elif item_type == "function_call":
                name = item.get("name")
                args = json.loads(item.get("arguments", "{}"))
                call_id = item.get("call_id")
                print(f"    Function: {name}({json.dumps(args)[:80]}...)")

                if name == "edit":
                    print("\n" + "=" * 60)
                    print("SUCCESS! Model called edit via Responses API!")
                    print(f"old_string: {args.get('old_string')}")
                    print(f"new_string: {args.get('new_string')}")
                    print("=" * 60)
                    return True

                # Add assistant output and tool result for next turn
                input_messages.append({"role": "assistant", "content": output})

                # Simulate tool result
                result = FILE_CONTENT if name == "read" else "OK"
                input_messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    print("\nFAILURE: No edit call after 3 turns via Responses API")
    return False


if __name__ == "__main__":
    main()
