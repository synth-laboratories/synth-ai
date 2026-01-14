#!/usr/bin/env python3
"""
Minimal repro to test if the model calls edit tools directly.
Bypasses OpenCode entirely to isolate the issue.
"""

import json
import os

from openai import OpenAI

# Use the same model
MODEL = "gpt-4o"  # or "gpt-5.2" if you have access

# Minimal tools - just read and edit
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

# Simulated file content (what the model sees after "reading")
FILE_CONTENT = """//! Tropius δ - Dragon Frontiers #23

pub const SET: &str = "DF";
pub const NUMBER: u32 = 23;
pub const NAME: &str = "Tropius δ";

pub const ABILITY_1_NAME: &str = "Tropical Heal";
pub const ATTACK_1_NAME: &str = "Grind";
pub const ATTACK_1_DAMAGE: u32 = 10;

// Attack helper for Grind
pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }

pub const ATTACK_1_TEXT: &str = "Does 10 damage times the amount of Energy attached to Tropius.";
"""

SYSTEM_PROMPT = """You are an expert Rust developer implementing Pokemon TCG cards.

CRITICAL: The stub file contains `todo!()` macros that YOU MUST REPLACE with working code.

Example - if you see:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }
// ATTACK_1_TEXT: "Does 10 damage times the amount of Energy attached"
```

You MUST use the edit tool to change it to:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { (10 * attached_energy) as i32 }
```

REQUIRED WORKFLOW:
1. Read the stub file ONCE to find the todo!() functions
2. IMMEDIATELY use the edit tool to replace todo!() with working code
3. Do NOT read the file multiple times. After ONE read, you must EDIT.
"""

USER_PROMPT = """Implement the grind_damage function in src/df/cards/df_023_tropius.rs

The file contains:
- `pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }`
- `ATTACK_1_TEXT: "Does 10 damage times the amount of Energy attached to Tropius."`

You must EDIT the file to replace `todo!()` with the correct implementation.
"""


def simulate_tool_result(tool_name: str, args: dict) -> str:
    """Simulate tool execution"""
    if tool_name == "read":
        return f"File content:\n{FILE_CONTENT}"
    elif tool_name == "edit":
        return f"Successfully edited {args.get('filePath')}"
    return "Unknown tool"


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Need OPENAI_API_KEY environment variable")
        return False

    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    print("=" * 60)
    print("DIRECT MODEL CALL TEST")
    print(f"Model: {MODEL}")
    print("=" * 60)

    for turn in range(5):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())

        print(f"Content: {assistant_message.content or '(none)'}")

        if assistant_message.tool_calls:
            for tc in assistant_message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"Tool call: {tool_name}({json.dumps(tool_args, indent=2)})")

                # Check if it called edit
                if tool_name == "edit":
                    print("\n" + "=" * 60)
                    print("SUCCESS! Model called edit tool!")
                    print(f"old_string: {tool_args.get('old_string')}")
                    print(f"new_string: {tool_args.get('new_string')}")
                    print("=" * 60)
                    return True

                # Simulate tool result
                result = simulate_tool_result(tool_name, tool_args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            print("No tool calls - model finished")
            break

    print("\n" + "=" * 60)
    print("FAILURE: Model never called edit tool after 5 turns")
    print("=" * 60)
    return False


if __name__ == "__main__":
    main()
