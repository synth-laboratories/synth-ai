#!/usr/bin/env python3
"""
Filter traces from DuckDB to create OpenAI SFT-ready .jsonl files
Supports two modes:
1. Trajectory-level filtering: Include entire trajectories above a score threshold
2. Window-based filtering: Extract high-scoring windows of actions
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import os
import sys
import toml

# Add synth_ai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager


def create_histogram(data: List[float], bins: int = 20, width: int = 60, height: int = 15, 
                    title: str = "", x_label: str = "", y_label: str = "") -> str:
    """Create a beautiful ASCII histogram."""
    if not data:
        return "No data to display"
    
    # Create histogram
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1
    
    # Normalize heights
    if max_count > 0:
        heights = [int(c * height / max_count) for c in counts]
    else:
        heights = [0] * len(counts)
    
    # Build the plot
    lines = []
    
    # Title
    if title:
        lines.append(f"\n{title.center(width + 10)}")
        lines.append("=" * (width + 10))
    
    # Y-axis label
    if y_label:
        lines.append(f"{y_label}")
    
    # Plot area with y-axis
    for y in range(height, 0, -1):
        # Y-axis value
        y_val = int(max_count * y / height)
        line = f"{y_val:>6} │"
        
        # Bars
        for h in heights:
            if h >= y:
                line += "█"
            else:
                line += " "
        
        lines.append(line)
    
    # X-axis
    lines.append(f"{'':>6} └" + "─" * len(heights))
    
    # X-axis labels
    x_labels_line = " " * 8
    min_val, max_val = min(data), max(data)
    
    # Add labels at key positions
    label_positions = [0, len(heights)//4, len(heights)//2, 3*len(heights)//4, len(heights)-1]
    for i, pos in enumerate(label_positions):
        if pos < len(edges) - 1:
            val = edges[pos]
            label = f"{val:.1f}"
            # Calculate position
            target_pos = 8 + pos
            if i == 0:
                x_labels_line = label + x_labels_line[len(label):]
            elif i == len(label_positions) - 1:
                start = max(0, target_pos - len(label))
                x_labels_line = x_labels_line[:start] + label
            else:
                start = max(0, target_pos - len(label)//2)
                end = min(len(x_labels_line), start + len(label))
                if start < len(x_labels_line):
                    x_labels_line = x_labels_line[:start] + label[:end-start] + x_labels_line[end:]
    
    lines.append(x_labels_line)
    
    # X-axis label
    if x_label:
        lines.append(f"\n{x_label.center(width + 10)}")
    
    return "\n".join(lines)


def create_bar_chart(categories: List[str], values: List[int], width: int = 60, 
                     title: str = "", show_values: bool = True) -> str:
    """Create a horizontal bar chart."""
    if not categories or not values:
        return "No data to display"
    
    max_val = max(values) if values else 1
    lines = []
    
    # Title
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * (width + 20))
    
    # Find longest category name for alignment
    max_cat_len = max(len(cat) for cat in categories)
    
    # Create bars
    for cat, val in zip(categories, values):
        # Normalize bar length
        bar_len = int(val * width / max_val) if max_val > 0 else 0
        bar = "█" * bar_len
        
        # Format line
        if show_values:
            line = f"{cat:<{max_cat_len}} │ {bar} {val}"
        else:
            line = f"{cat:<{max_cat_len}} │ {bar}"
        
        lines.append(line)
    
    return "\n".join(lines)


def filter_traces_from_duckdb(
    db_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    Filter traces from DuckDB based on configuration.
    
    Returns:
        Tuple of (num_examples, statistics_dict)
    """
    mode = config.get("mode", "trajectory")
    filters = config.get("filters", {})
    
    # Extract filtering parameters
    min_reward = filters.get("min_total_reward", 0.0)
    min_achievements = filters.get("min_achievements", 0)
    max_cost = filters.get("max_cost", float('inf'))
    max_tokens = filters.get("max_tokens", float('inf'))
    
    statistics = {
        "total_sessions": 0,
        "filtered_sessions": 0,
        "total_examples": 0,
        "reward_distribution": [],
        "token_distribution": [],
        "cost_distribution": [],
        "model_distribution": defaultdict(int)
    }
    
    with FinetuningDataExtractor(db_path) as extractor:
        # Get all sessions
        all_sessions_query = "SELECT session_id FROM session_traces"
        all_sessions = extractor.db_manager.query_traces(all_sessions_query)
        statistics["total_sessions"] = len(all_sessions)
        
        # Filter sessions based on criteria
        filtered_sessions = []
        
        for session_id in all_sessions['session_id']:
            metrics = extractor.get_session_metrics(session_id)
            
            # Apply filters
            if metrics['total_reward'] < min_reward:
                continue
            if metrics['total_cost'] > max_cost:
                continue
            if metrics['total_tokens'] > max_tokens:
                continue
            
            # Check achievements if required
            if min_achievements > 0:
                achievement_sessions = extractor.filter_by_achievements(min_achievements)
                if session_id not in achievement_sessions:
                    continue
            
            filtered_sessions.append(session_id)
            
            # Collect statistics
            statistics["reward_distribution"].append(metrics['total_reward'])
            statistics["token_distribution"].append(metrics['total_tokens'])
            statistics["cost_distribution"].append(metrics['total_cost'])
        
        statistics["filtered_sessions"] = len(filtered_sessions)
        
        # Extract training data
        if mode == "trajectory":
            training_data = extractor.extract_openai_format(
                session_ids=filtered_sessions,
                min_reward=min_reward
            )
        else:  # window mode
            # For window mode, we need to implement window extraction
            # For now, use trajectory mode
            training_data = extractor.extract_openai_format(
                session_ids=filtered_sessions,
                min_reward=min_reward
            )
        
        statistics["total_examples"] = len(training_data)
        
        # Write to output file
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        # Get model distribution
        model_query = """
            SELECT model_name, COUNT(*) as count
            FROM events
            WHERE event_type = 'cais' AND model_name IS NOT NULL
            GROUP BY model_name
        """
        model_stats = extractor.db_manager.query_traces(model_query)
        for _, row in model_stats.iterrows():
            statistics["model_distribution"][row['model_name']] = row['count']
    
    return len(training_data), statistics


def print_statistics(stats: Dict[str, Any]):
    """Print filtering statistics with visualizations."""
    print("\n" + "="*80)
    print("FILTERING STATISTICS")
    print("="*80)
    
    # Basic counts
    print(f"\nTotal sessions in database: {stats['total_sessions']}")
    print(f"Sessions after filtering: {stats['filtered_sessions']}")
    print(f"Training examples generated: {stats['total_examples']}")
    
    filter_rate = (stats['filtered_sessions'] / stats['total_sessions'] * 100) if stats['total_sessions'] > 0 else 0
    print(f"Filter pass rate: {filter_rate:.1f}%")
    
    # Reward distribution
    if stats['reward_distribution'] and any(not np.isnan(x) for x in stats['reward_distribution']):
        valid_rewards = [x for x in stats['reward_distribution'] if not np.isnan(x)]
        if valid_rewards:
            print(create_histogram(
                valid_rewards,
                bins=20,
                title="Reward Distribution",
                x_label="Total Reward",
                y_label="Count"
            ))
            
            print(f"\nReward statistics:")
            print(f"  Min: {min(valid_rewards):.2f}")
            print(f"  Max: {max(valid_rewards):.2f}")
            print(f"  Mean: {np.mean(valid_rewards):.2f}")
            print(f"  Median: {np.median(valid_rewards):.2f}")
    else:
        print("\nNo valid reward data to display.")
    
    # Token distribution
    if stats['token_distribution'] and any(not np.isnan(x) for x in stats['token_distribution']):
        valid_tokens = [x for x in stats['token_distribution'] if not np.isnan(x)]
        if valid_tokens:
            print(create_histogram(
                valid_tokens,
                bins=20,
                title="Token Usage Distribution",
                x_label="Total Tokens",
                y_label="Count"
            ))
    
    # Model distribution
    if stats['model_distribution']:
        models = list(stats['model_distribution'].keys())
        counts = list(stats['model_distribution'].values())
        print(create_bar_chart(
            models,
            counts,
            title="Model Usage",
            show_values=True
        ))
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter traces from DuckDB for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use default config
  python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -o ft_data/training.jsonl
  
  # Use custom config file
  python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -c filter_config.toml
  
  # Override config parameters
  python filter_traces_sft_duckdb.py -d crafter_traces.duckdb --min-reward 5.0 --max-cost 0.1
        """
    )
    
    parser.add_argument('-d', '--database', required=True, help='Path to DuckDB database')
    parser.add_argument('-o', '--output', default='ft_data/training.jsonl', help='Output JSONL file')
    parser.add_argument('-c', '--config', help='Configuration TOML file')
    
    # Filter overrides
    parser.add_argument('--mode', choices=['trajectory', 'window'], help='Filtering mode')
    parser.add_argument('--min-reward', type=float, help='Minimum total reward')
    parser.add_argument('--min-achievements', type=int, help='Minimum achievements')
    parser.add_argument('--max-cost', type=float, help='Maximum cost')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens')
    
    parser.add_argument('--dry-run', action='store_true', help='Show statistics without writing output')
    
    args = parser.parse_args()
    
    # Load config
    config = {
        "mode": "trajectory",
        "filters": {
            "min_total_reward": 1.0,
            "min_achievements": 0,
            "max_cost": 10.0,
            "max_tokens": 100000
        }
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            loaded_config = toml.load(f)
            config.update(loaded_config)
    
    # Apply command-line overrides
    if args.mode:
        config["mode"] = args.mode
    if args.min_reward is not None:
        config["filters"]["min_total_reward"] = args.min_reward
    if args.min_achievements is not None:
        config["filters"]["min_achievements"] = args.min_achievements
    if args.max_cost is not None:
        config["filters"]["max_cost"] = args.max_cost
    if args.max_tokens is not None:
        config["filters"]["max_tokens"] = args.max_tokens
    
    print(f"Using database: {args.database}")
    print(f"Output file: {args.output}")
    print(f"Mode: {config['mode']}")
    print(f"Filters: {json.dumps(config['filters'], indent=2)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No output will be written")
    
    # Run filtering
    num_examples, stats = filter_traces_from_duckdb(
        args.database,
        args.output if not args.dry_run else "/dev/null",
        config
    )
    
    # Print statistics
    print_statistics(stats)
    
    if not args.dry_run:
        print(f"\n✅ Successfully wrote {num_examples} training examples to {args.output}")
    else:
        print(f"\n✅ Would write {num_examples} training examples (dry run)")


if __name__ == "__main__":
    main()