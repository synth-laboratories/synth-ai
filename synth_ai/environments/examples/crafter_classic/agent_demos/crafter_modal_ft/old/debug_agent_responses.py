#!/usr/bin/env python3
"""Debug why agent is not using multiple actions."""

import json
import sys
from pathlib import Path

# Read the latest run output from stdin or file
if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        output = f.read()
else:
    # Check for latest log file
    log_files = list(Path(".").glob("crafter_run_*.log"))
    if not log_files:
        print("No log files found. Run with: python test_crafter_react_agent_lm_synth.py --model 'Qwen/Qwen2.5-14B-Instruct' --episodes 1 --max-steps 10 --verbose 2>&1 | tee crafter_run.log")
        exit(1)
    
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    with open(latest_log) as f:
        output = f.read()

# Parse tool calls
tool_calls = []
single_action_count = 0
multi_action_count = 0

for line in output.split('\n'):
    if "🔧 Turn" in line and "Tool Call:" in line:
        turn_info = {"turn": line}
        tool_calls.append(turn_info)
    elif "Actions:" in line and tool_calls:
        actions_str = line.strip().split("Actions:")[1].strip()
        try:
            # Parse the action list
            actions = eval(actions_str) if actions_str else []
            tool_calls[-1]["actions"] = actions
            tool_calls[-1]["action_count"] = len(actions)
            
            if len(actions) == 1:
                single_action_count += 1
            elif len(actions) > 1:
                multi_action_count += 1
        except:
            tool_calls[-1]["actions"] = "parse_error"
            tool_calls[-1]["action_count"] = 0

print("🔍 AGENT ACTION ANALYSIS\n")
print(f"Total tool calls: {len(tool_calls)}")
print(f"Single action calls: {single_action_count}")
print(f"Multi-action calls: {multi_action_count}")
print(f"Average actions per call: {sum(tc.get('action_count', 0) for tc in tool_calls) / len(tool_calls) if tool_calls else 0:.2f}")

# Show distribution
action_counts = {}
for tc in tool_calls:
    count = tc.get('action_count', 0)
    action_counts[count] = action_counts.get(count, 0) + 1

print("\nAction count distribution:")
for count in sorted(action_counts.keys()):
    print(f"  {count} actions: {action_counts[count]} times")

# Show examples of multi-action calls
print("\n📋 Multi-action examples:")
multi_examples = [tc for tc in tool_calls if tc.get('action_count', 0) > 1]
for example in multi_examples[:5]:
    print(f"  {example['turn']}")
    print(f"    Actions: {example['actions']}")

# Check for response parsing issues
print("\n🔍 Response preview analysis:")
response_previews = []
for line in output.split('\n'):
    if "📝 Raw response preview:" in line:
        preview = line.split("preview:")[1].strip()
        response_previews.append(preview)

if response_previews:
    print(f"Found {len(response_previews)} response previews")
    # Check if responses mention multiple actions
    multi_action_mentions = 0
    for preview in response_previews[:5]:
        if any(word in preview.lower() for word in ['multiple', 'sequence', 'then', 'after']):
            multi_action_mentions += 1
        print(f"  - {preview[:100]}...")
    
    print(f"\nResponses mentioning sequences: {multi_action_mentions}/{len(response_previews[:5])}")