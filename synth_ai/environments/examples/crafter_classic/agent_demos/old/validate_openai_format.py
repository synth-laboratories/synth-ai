#!/usr/bin/env python3
"""
Validate that JSONL files are compatible with OpenAI fine-tuning format
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def validate_openai_format(file_path: Path) -> tuple[bool, List[str]]:
    """
    Validate JSONL file for OpenAI fine-tuning compatibility.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    line_count = 0
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                
                # Parse JSON
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                
                # Check required structure
                if not isinstance(data, dict):
                    errors.append(f"Line {line_num}: Root must be a dictionary")
                    continue
                
                # Check for messages key
                if 'messages' not in data:
                    errors.append(f"Line {line_num}: Missing 'messages' key")
                    continue
                
                messages = data['messages']
                if not isinstance(messages, list):
                    errors.append(f"Line {line_num}: 'messages' must be a list")
                    continue
                
                if len(messages) < 2:
                    errors.append(f"Line {line_num}: Need at least 2 messages (system/user + assistant)")
                    continue
                
                # Validate each message
                roles_seen = []
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {line_num}, message {msg_idx}: Message must be a dictionary")
                        continue
                    
                    # Check required fields
                    if 'role' not in msg:
                        errors.append(f"Line {line_num}, message {msg_idx}: Missing 'role'")
                        continue
                    
                    role = msg['role']
                    roles_seen.append(role)
                    
                    if role not in ['system', 'user', 'assistant', 'tool']:
                        errors.append(f"Line {line_num}, message {msg_idx}: Invalid role '{role}'")
                    
                    # Check content or tool_calls
                    if role == 'assistant':
                        # Assistant messages can have content, tool_calls, or both
                        if 'content' not in msg and 'tool_calls' not in msg:
                            errors.append(f"Line {line_num}, message {msg_idx}: Assistant message needs 'content' or 'tool_calls'")
                        
                        # Validate tool_calls structure
                        if 'tool_calls' in msg:
                            tool_calls = msg['tool_calls']
                            if not isinstance(tool_calls, list):
                                errors.append(f"Line {line_num}, message {msg_idx}: 'tool_calls' must be a list")
                            else:
                                for tc_idx, tc in enumerate(tool_calls):
                                    if not isinstance(tc, dict):
                                        errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Must be a dict")
                                        continue
                                    
                                    # Check tool call structure
                                    if 'id' not in tc:
                                        errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Missing 'id'")
                                    if 'type' not in tc:
                                        errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Missing 'type'")
                                    if 'function' not in tc:
                                        errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Missing 'function'")
                                    else:
                                        func = tc['function']
                                        if not isinstance(func, dict):
                                            errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: 'function' must be a dict")
                                        else:
                                            if 'name' not in func:
                                                errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Missing function 'name'")
                                            if 'arguments' not in func:
                                                errors.append(f"Line {line_num}, message {msg_idx}, tool_call {tc_idx}: Missing function 'arguments'")
                    else:
                        # Other roles must have content
                        if 'content' not in msg:
                            errors.append(f"Line {line_num}, message {msg_idx}: {role} message missing 'content'")
                
                # Check message order
                if roles_seen[-1] != 'assistant':
                    errors.append(f"Line {line_num}: Last message must be from assistant (found {roles_seen[-1]})")
                
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, errors
    
    print(f"Validated {line_count} examples")
    return len(errors) == 0, errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_openai_format.py <jsonl_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Validating OpenAI format for: {file_path}")
    print("=" * 60)
    
    is_valid, errors = validate_openai_format(file_path)
    
    if is_valid:
        print("✅ File is valid for OpenAI fine-tuning!")
        
        # Show sample
        with open(file_path, 'r') as f:
            first_line = f.readline()
            example = json.loads(first_line)
            
        print("\nSample example structure:")
        print(f"- Number of messages: {len(example['messages'])}")
        print(f"- Message roles: {[msg['role'] for msg in example['messages']]}")
        
        # Check if assistant has tool calls
        last_msg = example['messages'][-1]
        if 'tool_calls' in last_msg:
            print(f"- Assistant uses tool calls: Yes")
            print(f"- Number of tool calls: {len(last_msg['tool_calls'])}")
            if last_msg['tool_calls']:
                print(f"- First tool: {last_msg['tool_calls'][0]['function']['name']}")
        else:
            print(f"- Assistant uses tool calls: No")
    else:
        print(f"❌ File has {len(errors)} validation errors:")
        for i, error in enumerate(errors[:10]):  # Show first 10 errors
            print(f"  {i+1}. {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())