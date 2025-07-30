#!/usr/bin/env python3
"""
Prepare and Validate JSONL Data for Vertex AI Fine-tuning
=========================================================
This script validates and prepares JSONL data for Gemini fine-tuning on Vertex AI.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
from collections import defaultdict
import random


def validate_jsonl_line(line: str, line_num: int) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a single JSONL line for Vertex AI compatibility."""
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError as e:
        return False, f"Line {line_num}: Invalid JSON - {e}", {}
    
    # Check required structure
    if "messages" not in data:
        return False, f"Line {line_num}: Missing 'messages' field", {}
    
    messages = data["messages"]
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be a list", {}
    
    if len(messages) < 2:
        return False, f"Line {line_num}: Need at least 2 messages (user and assistant)", {}
    
    # Validate message structure
    valid_roles = {"user", "assistant", "system"}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Line {line_num}: Message {i} must be a dict", {}
        
        if "role" not in msg:
            return False, f"Line {line_num}: Message {i} missing 'role'", {}
        
        if "content" not in msg:
            return False, f"Line {line_num}: Message {i} missing 'content'", {}
        
        if msg["role"] not in valid_roles:
            return False, f"Line {line_num}: Message {i} invalid role '{msg['role']}'", {}
    
    # Check conversation flow
    if messages[-1]["role"] != "assistant":
        return False, f"Line {line_num}: Last message must be from assistant", {}
    
    # Calculate token estimate (rough)
    total_chars = sum(len(msg["content"]) for msg in messages)
    token_estimate = total_chars // 4  # Rough estimate
    
    return True, "OK", {
        "num_messages": len(messages),
        "roles": [msg["role"] for msg in messages],
        "token_estimate": token_estimate,
        "total_chars": total_chars
    }


def analyze_jsonl_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a JSONL file for Vertex AI compatibility."""
    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "invalid_lines": 0,
        "errors": [],
        "token_distribution": defaultdict(int),
        "message_count_distribution": defaultdict(int),
        "role_patterns": defaultdict(int),
        "total_tokens_estimate": 0
    }
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            stats["total_lines"] += 1
            is_valid, error_msg, line_stats = validate_jsonl_line(line, line_num)
            
            if is_valid:
                stats["valid_lines"] += 1
                stats["total_tokens_estimate"] += line_stats["token_estimate"]
                
                # Bucket token counts
                tokens = line_stats["token_estimate"]
                if tokens < 100:
                    stats["token_distribution"]["<100"] += 1
                elif tokens < 500:
                    stats["token_distribution"]["100-500"] += 1
                elif tokens < 1000:
                    stats["token_distribution"]["500-1000"] += 1
                elif tokens < 2000:
                    stats["token_distribution"]["1000-2000"] += 1
                else:
                    stats["token_distribution"]["2000+"] += 1
                
                # Message count distribution
                msg_count = line_stats["num_messages"]
                stats["message_count_distribution"][msg_count] += 1
                
                # Role patterns
                role_pattern = "->".join(line_stats["roles"])
                stats["role_patterns"][role_pattern] += 1
            else:
                stats["invalid_lines"] += 1
                stats["errors"].append(error_msg)
                if len(stats["errors"]) > 10:
                    stats["errors"].append("... (truncated)")
                    break
    
    return stats


def create_subset(input_path: Path, output_path: Path, num_examples: int, 
                  shuffle: bool = True, seed: int = 42):
    """Create a subset of the JSONL file."""
    # Read all valid lines
    valid_lines = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                is_valid, _, _ = validate_jsonl_line(line, len(valid_lines) + 1)
                if is_valid:
                    valid_lines.append(line.strip())
    
    # Sample subset
    if shuffle:
        random.seed(seed)
        random.shuffle(valid_lines)
    
    subset = valid_lines[:num_examples]
    
    # Write subset
    with open(output_path, 'w') as f:
        for line in subset:
            f.write(line + '\n')
    
    print(f"✅ Created subset with {len(subset)} examples at {output_path}")
    return len(subset)


def convert_for_vertex_ai(input_path: Path, output_path: Path, 
                         add_system_prompt: bool = True):
    """Convert JSONL to Vertex AI format with optional enhancements."""
    converted_count = 0
    
    system_prompt = """You are an expert Crafter player. Your goal is to achieve as many objectives as possible efficiently.

Key objectives: collect resources, craft tools (pickaxe → stone pickaxe → iron pickaxe), make iron sword, survive.

Always think step-by-step about your current situation and plan your next action carefully."""
    
    with open(input_path, 'r') as inf, open(output_path, 'w') as outf:
        for line in inf:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                messages = data["messages"]
                
                # Optionally add system prompt
                if add_system_prompt and messages[0]["role"] != "system":
                    messages = [{"role": "system", "content": system_prompt}] + messages
                
                # Ensure proper format
                formatted_data = {"messages": messages}
                
                outf.write(json.dumps(formatted_data) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"⚠️  Skipping line due to error: {e}")
    
    print(f"✅ Converted {converted_count} examples to {output_path}")
    return converted_count


def estimate_training_cost(stats: Dict[str, Any], price_per_million: float = 4.0):
    """Estimate Vertex AI training cost."""
    total_tokens = stats["total_tokens_estimate"]
    total_millions = total_tokens / 1_000_000
    estimated_cost = total_millions * price_per_million
    
    return {
        "total_tokens": total_tokens,
        "total_millions": round(total_millions, 2),
        "estimated_cost_usd": round(estimated_cost, 2),
        "price_per_million": price_per_million
    }


def print_analysis_report(stats: Dict[str, Any], cost_estimate: Dict[str, Any]):
    """Print a detailed analysis report."""
    print("\n" + "=" * 60)
    print("📊 VERTEX AI FINE-TUNING DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\n✅ Valid examples: {stats['valid_lines']}")
    print(f"❌ Invalid examples: {stats['invalid_lines']}")
    print(f"📝 Total lines: {stats['total_lines']}")
    print(f"✔️  Validation rate: {stats['valid_lines']/stats['total_lines']*100:.1f}%")
    
    if stats['errors']:
        print(f"\n⚠️  First few errors:")
        for error in stats['errors'][:5]:
            print(f"   - {error}")
    
    print(f"\n📊 Token Distribution:")
    for bucket, count in sorted(stats['token_distribution'].items()):
        print(f"   {bucket} tokens: {count} examples")
    
    print(f"\n💬 Message Patterns:")
    for pattern, count in sorted(stats['role_patterns'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {pattern}: {count} examples")
    
    print(f"\n💰 Cost Estimate:")
    print(f"   Total tokens: {cost_estimate['total_tokens']:,}")
    print(f"   Token millions: {cost_estimate['total_millions']}")
    print(f"   Estimated cost: ${cost_estimate['estimated_cost_usd']} USD")
    print(f"   (at ${cost_estimate['price_per_million']}/million tokens)")
    
    print("\n📝 Recommendations:")
    if stats['valid_lines'] < 100:
        print("   ⚠️  Dataset is small. Consider generating more examples.")
    if stats['valid_lines'] > 10000:
        print("   💡 Large dataset. Consider creating a smaller subset for initial tests.")
    if cost_estimate['estimated_cost_usd'] > 100:
        print("   💰 High estimated cost. Consider using a subset for initial experiments.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate JSONL for Vertex AI")
    parser.add_argument("jsonl_path", type=Path, help="Path to JSONL file")
    parser.add_argument("--validate", action="store_true", help="Validate the JSONL file")
    parser.add_argument("--create-subset", type=int, help="Create subset with N examples")
    parser.add_argument("--convert", action="store_true", help="Convert to Vertex AI format")
    parser.add_argument("--add-system", action="store_true", help="Add system prompt to messages")
    parser.add_argument("--output", type=Path, help="Output path for converted/subset file")
    
    args = parser.parse_args()
    
    if not args.jsonl_path.exists():
        sys.exit(f"❌ File not found: {args.jsonl_path}")
    
    # Always run validation
    print(f"🔍 Analyzing {args.jsonl_path}...")
    stats = analyze_jsonl_file(args.jsonl_path)
    cost_estimate = estimate_training_cost(stats)
    
    if args.validate or (not args.create_subset and not args.convert):
        print_analysis_report(stats, cost_estimate)
    
    # Create subset if requested
    if args.create_subset:
        output_path = args.output or args.jsonl_path.with_name(
            f"{args.jsonl_path.stem}_subset_{args.create_subset}.jsonl"
        )
        create_subset(args.jsonl_path, output_path, args.create_subset)
    
    # Convert if requested
    if args.convert:
        output_path = args.output or args.jsonl_path.with_name(
            f"{args.jsonl_path.stem}_vertex.jsonl"
        )
        convert_for_vertex_ai(args.jsonl_path, output_path, args.add_system)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()