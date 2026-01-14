#!/usr/bin/env python3
"""
Test the same thing but routing through the interceptor.
This helps isolate if the interceptor is causing issues.
"""

import json
import os

import httpx

INTERCEPTOR_URL = "http://localhost:8000/api/interceptor/v1/test-trial/chat/completions"
MODEL = "gpt-4o"  # or gpt-5.2

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string", "description": "Path to file"}},
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
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
"""


def main():
    api_key = os.environ.get("SYNTH_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Need SYNTH_API_KEY or OPENAI_API_KEY")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Edit this file:\n{FILE_CONTENT}"},
    ]

    print("=" * 60)
    print("INTERCEPTOR ROUTING TEST")
    print(f"URL: {INTERCEPTOR_URL}")
    print(f"Model: {MODEL}")
    print("=" * 60)

    for turn in range(3):
        print(f"\n--- Turn {turn + 1} ---")

        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }

        try:
            resp = httpx.post(
                INTERCEPTOR_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls", [])

        print(f"Content: {content or '(none)'}")
        print(f"Tool calls: {len(tool_calls)}")

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                args = json.loads(fn.get("arguments", "{}"))
                print(f"  -> {name}: {json.dumps(args)[:100]}...")

                if name == "edit":
                    print("\nSUCCESS! Model called edit via interceptor!")
                    return True

                # Add tool result
                messages.append(message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": f"File content:\n{FILE_CONTENT}" if name == "read" else "OK",
                    }
                )
        else:
            messages.append(message)

    print("\nFAILURE: No edit call after 3 turns")
    return False


if __name__ == "__main__":
    main()
