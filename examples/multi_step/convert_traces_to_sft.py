#!/usr/bin/env python3
"""Convert Crafter trace format to SFT format with messages[] structure."""

import json
import sys
from pathlib import Path

def convert_trace_to_sft(trace: dict) -> dict:
    """Convert a single trace to SFT format."""
    # Extract dialogue from trace
    dialogue = trace.get("dialogue", [])
    assistant = trace.get("assistant", {})
    
    # Build messages list
    messages = []
    
    # Add dialogue history
    for msg in dialogue:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add assistant response if present
    if assistant:
        content = assistant.get("content", "")
        tool_calls = assistant.get("tool_calls", [])
        
        # If there are tool calls, format them
        if tool_calls:
            # Convert tool calls to a simple text format for SFT
            tool_text = "\n".join([
                f"Tool: {tc['name']}\nArguments: {json.dumps(tc.get('arguments', {}))}"
                for tc in tool_calls
            ])
            content = f"{content}\n\n{tool_text}".strip()
        
        messages.append({
            "role": "assistant",
            "content": content
        })
    
    return {"messages": messages}

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_traces_to_sft.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(f"{input_path.stem}_sft_format.jsonl")
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Converting {input_path} → {output_path}")
    
    converted = 0
    skipped = 0
    
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line_no, line in enumerate(f_in, 1):
            try:
                trace = json.loads(line.strip())
                sft_entry = convert_trace_to_sft(trace)
                
                # Only write if we have messages
                if sft_entry["messages"]:
                    f_out.write(json.dumps(sft_entry) + "\n")
                    converted += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"Warning: Skipping line {line_no}: {e}")
                skipped += 1
    
    print(f"✅ Converted {converted} entries, skipped {skipped}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()

