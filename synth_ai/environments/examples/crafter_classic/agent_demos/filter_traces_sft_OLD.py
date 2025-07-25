#!/usr/bin/env python3
"""
Filter traces to create OpenAI SFT-ready .jsonl files
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
    if not categories:
        return "No data to display"
    
    max_val = max(values) if values else 1
    max_label_len = max(len(cat) for cat in categories)
    
    lines = []
    
    # Title
    if title:
        lines.append(f"\n{title}")
        lines.append("=" * (width + max_label_len + 15))
    
    # Bars
    for cat, val in zip(categories, values):
        bar_width = int(val * width / max_val) if max_val > 0 else 0
        bar = "█" * bar_width
        
        if show_values:
            line = f"{cat:<{max_label_len}} │ {bar} {val}"
        else:
            line = f"{cat:<{max_label_len}} │ {bar}"
        
        lines.append(line)
    
    return "\n".join(lines)


def display_analysis_results(scores: List[float], achievements_per_trace: Dict[str, List[str]], 
                           window_scores: List[Tuple[str, int, int, float]] = None):
    """Display beautiful analysis results."""
    # Don't clear screen in analyze mode
    # os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " CRAFTER TRACE ANALYSIS RESULTS ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 1. Trajectory Score Distribution
    # For discrete scores, create custom histogram
    if scores:
        score_counts = {}
        for s in scores:
            score_counts[int(s)] = score_counts.get(int(s), 0) + 1
        
        # Create bar chart for scores
        max_score = int(max(scores))
        max_count = max(score_counts.values())
        
        print("\n" + "📊 Trajectory Score Distribution".center(70))
        print("=" * 70)
        print("Traces")
        
        # Y-axis scale
        for y in range(max_count, 0, -max(1, max_count // 10)):
            line = f"{y:>6} │"
            for score in range(max_score + 1):
                count = score_counts.get(score, 0)
                # Each score gets 10 characters width
                bar_height = int(count * 10 / max_count) if max_count > 0 else 0
                if count >= y:
                    line += " ████████ "
                else:
                    line += "          "
            print(line)
        
        # X-axis
        print(f"{'':>6} └" + "─" * (10 * (max_score + 1)))
        
        # X-axis labels
        x_labels = "       "
        for score in range(max_score + 1):
            x_labels += f"    {score}     "
        print(x_labels)
        print("\n" + "Number of Achievements (Score)".center(70))
    
    # Statistics box
    print("\n┌─ Statistics ─────────────────────────┐")
    print(f"│ Total traces: {len(scores):<23}│")
    print(f"│ Mean score: {np.mean(scores):<25.2f}│")
    print(f"│ Median score: {np.median(scores):<23.1f}│")
    print(f"│ Max score: {max(scores):<26.0f}│")
    print(f"│ Traces with score > 0: {sum(1 for s in scores if s > 0):<14}│")
    print("└──────────────────────────────────────┘")
    
    # 2. Achievement Distribution
    all_achievements = []
    for achievements in achievements_per_trace.values():
        all_achievements.extend(achievements)
    
    from collections import Counter
    achievement_counts = Counter(all_achievements)
    
    if achievement_counts:
        top_achievements = achievement_counts.most_common(10)
        categories = [ach for ach, _ in top_achievements]
        values = [count for _, count in top_achievements]
        
        print("\n" + create_bar_chart(
            categories,
            values,
            width=40,
            title="🏆 Top 10 Achievements Unlocked",
            show_values=True
        ))
    
    # 3. Window Analysis (if provided)
    if window_scores:
        window_score_values = [score for _, _, _, score in window_scores]
        unique_window_scores = sorted(set(window_score_values))
        
        print("\n┌─ Window Analysis ────────────────────┐")
        print(f"│ Total windows analyzed: {len(window_scores):<13}│")
        print(f"│ Windows with score > 0: {sum(1 for s in window_score_values if s > 0):<13}│")
        print(f"│ Unique score values: {unique_window_scores}".ljust(39) + "│")
        print("└──────────────────────────────────────┘")
    
    # 4. Filtering Recommendations
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " FILTERING RECOMMENDATIONS ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    score_thresholds = [1, 2, 3]
    print("\n┌─ Trajectory Filtering ───────────────┐")
    for threshold in score_thresholds:
        count = sum(1 for s in scores if s >= threshold)
        pct = count / len(scores) * 100 if scores else 0
        print(f"│ Score ≥ {threshold}: {count:>3} traces ({pct:>5.1f}%)      │")
    print("└──────────────────────────────────────┘")
    
    if window_scores:
        print("\n┌─ Window Filtering ───────────────────┐")
        window_thresholds = [1, 2]
        for threshold in window_thresholds:
            count = sum(1 for s in window_score_values if s >= threshold)
            pct = count / len(window_scores) * 100 if window_scores else 0
            print(f"│ Score ≥ {threshold}: {count:>3} windows ({pct:>5.1f}%)     │")
        print("└──────────────────────────────────────┘")


def load_trace(trace_file: Path) -> Dict[str, Any]:
    """Load a trace file."""
    with open(trace_file, 'r') as f:
        return json.load(f)


def extract_trajectory_score(trace: Dict[str, Any]) -> float:
    """Extract the trajectory score from a trace."""
    # Look for episode results in metadata
    metadata = trace.get('session_metadata', [])
    
    # Handle list format with metadata_type
    if isinstance(metadata, list):
        # Find episode_results in the list
        for item in metadata:
            if isinstance(item, dict) and item.get('metadata_type') == 'episode_results':
                episode_results = item.get('data', {})
                break
        else:
            episode_results = {}
    else:
        episode_results = metadata.get('episode_results', {})
    
    # Use number of achievements as the primary score
    num_achievements = episode_results.get('num_achievements', 0)
    
    # Could also use shaped reward if available
    # total_reward = episode_results.get('total_reward', 0)
    
    return float(num_achievements)


def extract_llm_calls(trace: Dict[str, Any], hook_config: Optional[Dict[str, Any]] = None) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """Extract all LLM calls from a trace with their turn numbers and event metadata.
    
    Returns list of (turn_number, llm_record, event) tuples.
    """
    llm_calls = []
    
    # Get events that contain LLM calls
    events = trace.get('event_history', [])
    
    exclude_hooks = hook_config.get('exclude_hooks', []) if hook_config else []
    include_hooks = hook_config.get('include_hooks', []) if hook_config else []
    
    for event in events:
        # Look for CAISEvents from the agent
        if event.get('system_instance_id', '').startswith('crafter-react-agent'):
            # Check hook filtering
            event_hooks = event.get('hooks_triggered', [])
            
            # Skip if any exclude hooks were triggered
            if exclude_hooks and any(hook in event_hooks for hook in exclude_hooks):
                continue
            
            # Skip if include_hooks specified but none were triggered
            if include_hooks and not any(hook in event_hooks for hook in include_hooks):
                continue
            
            # Get the LLM call records
            llm_records = event.get('llm_call_records', [])
            turn = event.get('time_record', {}).get('message_time', 0)
            
            for record in llm_records:
                if record:
                    llm_calls.append((turn, record, event))
    
    return llm_calls


def calculate_window_score(trace: Dict[str, Any], start_turn: int, end_turn: int) -> float:
    """Calculate score for a window of turns."""
    # Count achievements unlocked in this window
    achievements_before = set()
    achievements_after = set()
    
    # Get messages to track achievement changes
    messages = trace.get('message_history', [])
    
    for message in messages:
        turn = message.get('time_record', {}).get('message_time', -1)
        if message.get('message_type') == 'observation':
            obs = message.get('content', {}).get('payload', {})
            achievements = obs.get('achievements_status', {})
            
            if turn == start_turn - 1:
                # Achievements before window
                achievements_before = {k for k, v in achievements.items() if v}
            elif turn == end_turn:
                # Achievements after window
                achievements_after = {k for k, v in achievements.items() if v}
    
    # Score is number of new achievements unlocked in window
    new_achievements = achievements_after - achievements_before
    return len(new_achievements)


def convert_to_openai_format(llm_call: Dict[str, Any], quality_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert an LLM call record to OpenAI fine-tuning format with quality filtering."""
    # Extract messages
    messages = llm_call.get('messages', [])
    
    # Extract the completion (assistant's response)
    response = llm_call.get('response', {})
    choices = response.get('choices', [])
    
    if choices:
        assistant_message = choices[0].get('message', {})
        content = assistant_message.get('content', '')
        
        # Apply quality filters if provided
        if quality_config:
            # Check minimum response length
            min_length = quality_config.get('min_response_length', 0)
            if content and len(content) < min_length:
                return None
            elif not content and min_length > 0:
                return None
            
            # Check if tool calls required
            require_tools = quality_config.get('require_tool_calls', False)
            tool_calls = assistant_message.get('tool_calls', [])
            if require_tools and not tool_calls:
                return None
            
            # Check excluded keywords
            exclude_keywords = quality_config.get('exclude_keywords', [])
            if content and exclude_keywords and any(keyword.lower() in content.lower() for keyword in exclude_keywords):
                return None
        
        # Build the completion message
        completion = {
            "role": "assistant",
            "content": content,
        }
        
        # Add tool calls if present
        tool_calls = assistant_message.get('tool_calls', [])
        if tool_calls:
            completion["tool_calls"] = tool_calls
        
        # Create the training example
        return {
            "messages": messages + [completion]
        }
    
    return None


def filter_by_trajectory_score(traces_dir: Path, output_file: Path, config: Dict[str, Any]):
    """Filter entire trajectories by score threshold with hook and quality filtering."""
    examples = []
    trace_scores = []
    included_count = 0
    hook_filtered_count = 0
    quality_filtered_count = 0
    trace_contributions = {}  # Track which traces contributed examples
    included_trace_scores = {}  # Track scores of included traces
    
    score_threshold = config.get('trajectory_filtering', {}).get('score_threshold', 2.0)
    hook_config = config.get('hook_filtering', {})
    quality_config = config.get('quality_filtering', {})
    
    # Process all trace files
    trace_files = sorted(traces_dir.glob("*.json"))
    print(f"Processing {len(trace_files)} trace files...")
    
    for trace_file in trace_files:
        trace = load_trace(trace_file)
        score = extract_trajectory_score(trace)
        trace_scores.append((trace_file.name, score))
        
        if score >= score_threshold:
            # Extract all LLM calls from this trajectory
            llm_calls = extract_llm_calls(trace, hook_config)
            initial_count = len(llm_calls)
            
            trajectory_examples = []
            for turn, llm_call, event in llm_calls:
                example = convert_to_openai_format(llm_call, quality_config)
                if example:
                    trajectory_examples.append(example)
                else:
                    quality_filtered_count += 1
            
            if trajectory_examples:
                trace_contributions[trace_file.name] = len(trajectory_examples)
                included_trace_scores[trace_file.name] = score
            
            hook_filtered_count += initial_count - len(llm_calls)
            examples.extend(trajectory_examples)
            included_count += 1
    
    # Save examples
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✓ Included {included_count}/{len(trace_files)} traces (score ≥ {score_threshold})")
    if hook_filtered_count > 0:
        print(f"✓ Filtered out {hook_filtered_count} events due to hook exclusions")
    if quality_filtered_count > 0:
        print(f"✓ Filtered out {quality_filtered_count} events due to quality filters")
    print(f"✓ Extracted {len(examples)} training examples from {len(trace_contributions)} unique traces")
    
    return trace_contributions, included_trace_scores


def filter_by_window_score(traces_dir: Path, output_file: Path, config: Dict[str, Any]):
    """Filter by sliding window with greedy extraction and hook/quality filtering."""
    all_examples = []
    window_stats = defaultdict(int)
    traces_with_windows = 0
    hook_filtered_count = 0
    quality_filtered_count = 0
    trace_contributions = {}  # Track which traces contributed examples
    window_scores_by_trace = defaultdict(list)  # Track window scores per trace
    trace_overall_scores = {}  # Track overall trace scores
    
    window_config = config.get('window_filtering', {})
    window_size = window_config.get('window_size', 5)
    score_threshold = window_config.get('score_threshold', 1.0)
    hook_config = config.get('hook_filtering', {})
    quality_config = config.get('quality_filtering', {})
    
    # Process all trace files
    trace_files = sorted(traces_dir.glob("*.json"))
    print(f"Processing {len(trace_files)} trace files with window_size={window_size}...")
    
    for trace_file in trace_files:
        trace = load_trace(trace_file)
        trace_score = extract_trajectory_score(trace)  # Get overall trace score
        llm_calls = extract_llm_calls(trace, hook_config)
        
        if not llm_calls:
            continue
        
        # Extract examples using greedy window approach
        examples = []
        used_turns = set()
        found_window = False
        trace_window_count = 0
        
        # Get max turn number
        max_turn = max(turn for turn, _, _ in llm_calls)
        
        # Try all possible windows
        for start in range(0, max_turn - window_size + 2):
            end = start + window_size - 1
            
            # Skip if any turn in window already used
            if any(t in used_turns for t in range(start, end + 1)):
                continue
            
            # Calculate window score
            score = calculate_window_score(trace, start, end)
            
            if score >= score_threshold:
                # Extract LLM calls from this window
                window_examples = []
                for turn, llm_call, event in llm_calls:
                    if start <= turn <= end:
                        example = convert_to_openai_format(llm_call, quality_config)
                        if example:
                            window_examples.append(example)
                        else:
                            quality_filtered_count += 1
                
                if window_examples:
                    examples.extend(window_examples)
                    trace_window_count += len(window_examples)
                    window_scores_by_trace[trace_file.name].append(score)
                    # Mark all turns in window as used
                    for t in range(start, end + 1):
                        used_turns.add(t)
                    
                    window_stats[score] += 1
                    found_window = True
        
        if found_window:
            traces_with_windows += 1
            trace_contributions[trace_file.name] = trace_window_count
            trace_overall_scores[trace_file.name] = trace_score
        
        all_examples.extend(examples)
    
    # Save examples
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    total_windows = sum(window_stats.values())
    print(f"\n✓ Found qualifying windows in {traces_with_windows}/{len(trace_files)} traces")
    print(f"✓ Extracted {total_windows} windows (score ≥ {score_threshold})")
    if hook_filtered_count > 0:
        print(f"✓ Filtered out {hook_filtered_count} events due to hook exclusions")
    if quality_filtered_count > 0:
        print(f"✓ Filtered out {quality_filtered_count} events due to quality filters") 
    print(f"✓ Generated {len(all_examples)} training examples from {len(trace_contributions)} unique traces")
    
    return trace_contributions, window_scores_by_trace, trace_overall_scores


def analyze_traces(traces_dir: Path, analyze_windows: bool = True, hook_config: Optional[Dict[str, Any]] = None):
    """Analyze traces to help choose thresholds."""
    scores = []
    achievements_per_trace = {}
    window_scores = []
    
    trace_files = sorted(traces_dir.glob("*.json"))
    
    print(f"Analyzing {len(trace_files)} trace files...")
    
    for trace_file in trace_files:
        trace = load_trace(trace_file)
        score = extract_trajectory_score(trace)
        scores.append(score)
        
        # Get achievement details
        metadata = trace.get('session_metadata', [])
        
        # Handle list format with metadata_type
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict) and item.get('metadata_type') == 'episode_results':
                    episode_results = item.get('data', {})
                    break
            else:
                episode_results = {}
        else:
            episode_results = metadata.get('episode_results', {})
        
        achievements = episode_results.get('achievements', {})
        unlocked = [k for k, v in achievements.items() if v]
        achievements_per_trace[trace_file.name] = unlocked
        
        # Analyze windows if requested
        if analyze_windows:
            llm_calls = extract_llm_calls(trace, hook_config)
            if llm_calls:
                max_turn = max(turn for turn, _, _ in llm_calls)
                # Check all possible 5-turn windows
                for start in range(0, max_turn - 4):
                    end = start + 4  # 5-turn window
                    window_score = calculate_window_score(trace, start, end)
                    if window_score > 0:  # Only track windows with achievements
                        window_scores.append((trace_file.name, start, end, window_score))
    
    # Display results
    display_analysis_results(scores, achievements_per_trace, window_scores if analyze_windows else None)


def main():
    parser = argparse.ArgumentParser(description="Filter traces for SFT data")
    parser.add_argument("traces_dir", type=Path, help="Directory containing trace files")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to TOML configuration file")
    parser.add_argument("--analyze", action="store_true", help="Analyze traces to help choose thresholds")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode for choosing thresholds")
    
    # Trajectory filtering options
    parser.add_argument("--trajectory-threshold", type=float, default=None,
                        help="Minimum trajectory score for inclusion")
    
    # Window filtering options
    parser.add_argument("--window-size", type=int, default=None,
                        help="Window size for window-based filtering")
    parser.add_argument("--window-threshold", type=float, default=None,
                        help="Minimum window score for inclusion")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        config = toml.load(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Look for default config in same directory as traces
        default_config = args.traces_dir.parent / "filter_config.toml"
        if default_config.exists():
            config = toml.load(default_config)
            print(f"Loaded configuration from {default_config}")
    
    # Override config with command line arguments
    if args.trajectory_threshold is not None:
        config.setdefault('trajectory_filtering', {})['score_threshold'] = args.trajectory_threshold
    if args.window_size is not None:
        config.setdefault('window_filtering', {})['window_size'] = args.window_size
    if args.window_threshold is not None:
        config.setdefault('window_filtering', {})['score_threshold'] = args.window_threshold
    
    if not args.traces_dir.exists():
        print(f"Error: Traces directory not found: {args.traces_dir}")
        return
    
    if args.analyze:
        hook_config = config.get('hook_filtering', {})
        analyze_traces(args.traces_dir, hook_config=hook_config)
        return
    
    # Interactive mode or direct filtering
    traj_threshold = config.get('trajectory_filtering', {}).get('score_threshold')
    window_threshold = config.get('window_filtering', {}).get('score_threshold')
    
    if args.interactive or (traj_threshold is None and window_threshold is None):
        # First show analysis
        hook_config = config.get('hook_filtering', {})
        analyze_traces(args.traces_dir, hook_config=hook_config)
        
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " INTERACTIVE THRESHOLD SELECTION ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Get thresholds interactively
        print("\nBased on the analysis above, please choose filtering thresholds:")
        
        if traj_threshold is None:
            while True:
                try:
                    traj_input = input("\n📊 Trajectory score threshold (e.g., 2.0): ")
                    traj_threshold = float(traj_input)
                    config.setdefault('trajectory_filtering', {})['score_threshold'] = traj_threshold
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        if window_threshold is None:
            while True:
                try:
                    window_input = input("📊 Window score threshold (e.g., 1.0): ")
                    window_threshold = float(window_input)
                    config.setdefault('window_filtering', {})['score_threshold'] = window_threshold
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        print(f"\nYou selected:")
        print(f"  • Trajectory threshold: {traj_threshold}")
        print(f"  • Window threshold: {window_threshold}")
        
        # Get custom filenames
        print(f"\nOutput file names:")
        traj_filename = input("📁 Trajectory output filename (default: trajectory_score.jsonl): ").strip()
        if not traj_filename:
            traj_filename = "trajectory_score.jsonl"
        elif not traj_filename.endswith('.jsonl'):
            traj_filename += '.jsonl'
        
        window_filename = input("📁 Window output filename (default: window_score.jsonl): ").strip()
        if not window_filename:
            window_filename = "window_score.jsonl"
        elif not window_filename.endswith('.jsonl'):
            window_filename += '.jsonl'
        
        # Store filenames in config for later use
        config.setdefault('output', {})['trajectory_file'] = traj_filename
        config.setdefault('output', {})['window_file'] = window_filename
        
        print(f"\nFiles will be saved as:")
        print(f"  • Trajectory data: {traj_filename}")
        print(f"  • Window data: {window_filename}")
        
        confirm = input("\nProceed with filtering? (y/n): ")
        if confirm.lower() != 'y':
            print("Filtering cancelled.")
            return
    
    # Ensure we have defaults if still None
    if traj_threshold is None:
        config.setdefault('trajectory_filtering', {})['score_threshold'] = 2.0
    if window_threshold is None:
        config.setdefault('window_filtering', {})['score_threshold'] = 1.0
    
    # Show configuration summary if hook or quality filtering enabled
    hook_config = config.get('hook_filtering', {})
    quality_config = config.get('quality_filtering', {})
    
    if hook_config.get('exclude_hooks') or hook_config.get('include_hooks') or quality_config:
        print("\n" + "Configuration Summary".center(50))
        print("=" * 50)
        
        if hook_config.get('exclude_hooks'):
            print(f"Excluding events with hooks: {hook_config['exclude_hooks']}")
        if hook_config.get('include_hooks'):
            print(f"Including only events with hooks: {hook_config['include_hooks']}")
        
        if quality_config.get('min_response_length'):
            print(f"Min response length: {quality_config['min_response_length']}")
        if quality_config.get('require_tool_calls'):
            print("Requiring tool calls in responses")
        if quality_config.get('exclude_keywords'):
            print(f"Excluding keywords: {quality_config['exclude_keywords']}")
    
    # Create ft_dataset directory
    # Get the agent_demos directory (where this script is located)
    script_dir = Path(__file__).parent
    ft_dataset_dir = script_dir / "ft_dataset"
    ft_dataset_dir.mkdir(exist_ok=True)
    
    # Get output file names from config
    output_config = config.get('output', {})
    traj_filename = output_config.get('trajectory_file', 'trajectory_score.jsonl')
    window_filename = output_config.get('window_file', 'window_score.jsonl')
    
    # Run both filtering methods
    print("\nRunning trajectory-based filtering...")
    print("=" * 50)
    traj_contributions, traj_trace_scores = filter_by_trajectory_score(
        args.traces_dir,
        ft_dataset_dir / traj_filename,
        config
    )
    
    print("\n\nRunning window-based filtering...")
    print("=" * 50)
    window_contributions, window_scores_by_trace, window_trace_scores = filter_by_window_score(
        args.traces_dir,
        ft_dataset_dir / window_filename,
        config
    )
    
    # Compare results
    traj_file = ft_dataset_dir / "trajectory_score.jsonl"
    window_file = ft_dataset_dir / "window_score.jsonl"
    
    if traj_file.exists() and window_file.exists():
        traj_count = sum(1 for _ in open(traj_file))
        window_count = sum(1 for _ in open(window_file))
        
        # Calculate yield rates
        total_traces = len(list(args.traces_dir.glob("*.json")))
        
        # For trajectory: count traces above threshold
        traj_threshold = config.get('trajectory_filtering', {}).get('score_threshold', 2.0)
        included_traces = 0
        for trace_file in args.traces_dir.glob("*.json"):
            trace = load_trace(trace_file)
            score = extract_trajectory_score(trace)
            if score >= traj_threshold:
                included_traces += 1
        
        traj_yield = (included_traces / total_traces * 100) if total_traces > 0 else 0
        
        # For windows: count traces with qualifying windows
        window_size = config.get('window_filtering', {}).get('window_size', 5)
        window_threshold = config.get('window_filtering', {}).get('score_threshold', 1.0)
        hook_config = config.get('hook_filtering', {})
        
        traces_with_windows = 0
        for trace_file in args.traces_dir.glob("*.json"):
            trace = load_trace(trace_file)
            llm_calls = extract_llm_calls(trace, hook_config)
            if llm_calls:
                max_turn = max(turn for turn, _, _ in llm_calls)
                for start in range(0, max_turn - window_size + 2):
                    end = start + window_size - 1
                    score = calculate_window_score(trace, start, end)
                    if score >= window_threshold:
                        traces_with_windows += 1
                        break
        
        window_yield = (traces_with_windows / total_traces * 100) if total_traces > 0 else 0
        
        print("\n\nComparison:")
        print("=" * 50)
        print(f"Trajectory-based: {traj_count} examples ({traj_yield:.1f}% of traces)")
        
        # Show trajectory contribution distribution
        if traj_contributions:
            traj_unique_count = len(traj_contributions)
            print(f"  └─ From {traj_unique_count} unique traces")
            
            # Show distribution of examples per trace
            example_counts = list(traj_contributions.values())
            if example_counts:
                avg_examples = sum(example_counts) / len(example_counts)
                min_examples = min(example_counts)
                max_examples = max(example_counts)
                print(f"  └─ Examples per trace: min={min_examples}, avg={avg_examples:.1f}, max={max_examples}")
            
            # Show trace score distribution for included traces
            if traj_trace_scores:
                trace_score_counts = {}
                for score in traj_trace_scores.values():
                    score_int = int(score)
                    trace_score_counts[score_int] = trace_score_counts.get(score_int, 0) + 1
                
                print("  └─ Trace score distribution:")
                for score in sorted(trace_score_counts.keys()):
                    count = trace_score_counts[score]
                    print(f"      Score {score}: {count} traces")
        
        print(f"\nWindow-based: {window_count} examples ({window_yield:.1f}% of traces)")
        
        # Show window contribution distribution
        if window_contributions:
            window_unique_count = len(window_contributions)
            print(f"  └─ From {window_unique_count} unique traces")
            
            # Show distribution of examples per trace
            example_counts = list(window_contributions.values())
            if example_counts:
                avg_examples = sum(example_counts) / len(example_counts)
                min_examples = min(example_counts)
                max_examples = max(example_counts)
                print(f"  └─ Examples per trace: min={min_examples}, avg={avg_examples:.1f}, max={max_examples}")
            
            # Show trace score distribution for traces with windows
            if window_trace_scores:
                trace_score_counts = {}
                for score in window_trace_scores.values():
                    score_int = int(score)
                    trace_score_counts[score_int] = trace_score_counts.get(score_int, 0) + 1
                
                print("  └─ Trace score distribution:")
                for score in sorted(trace_score_counts.keys()):
                    count = trace_score_counts[score]
                    print(f"      Score {score}: {count} traces")
            
            # Show window score distribution
            all_window_scores = []
            for scores in window_scores_by_trace.values():
                all_window_scores.extend(scores)
            
            if all_window_scores:
                score_counts = {}
                for score in all_window_scores:
                    score_counts[int(score)] = score_counts.get(int(score), 0) + 1
                
                print("  └─ Window score distribution:")
                for score in sorted(score_counts.keys()):
                    count = score_counts[score]
                    print(f"      Score {score}: {count} windows")
        
        print(f"\nOutput files saved to: {ft_dataset_dir}/")
        
        # Generate metadata
        print("\nGenerating metadata...")
        generate_metadata(args.traces_dir, ft_dataset_dir, config)


def generate_metadata(traces_dir: Path, output_dir: Path, config: Dict[str, Any]):
    """Generate comprehensive metadata for the filtered datasets."""
    metadata = {
        "dataset_creation": {
            "source_traces_dir": traces_dir.name,
            "num_source_traces": len(list(traces_dir.glob("*.json"))),
            "filtering_methods": ["trajectory_score", "window_score"],
            "config_used": config
        }
    }
    
    traj_threshold = config.get('trajectory_filtering', {}).get('score_threshold', 2.0)
    window_threshold = config.get('window_filtering', {}).get('score_threshold', 1.0)
    window_size = config.get('window_filtering', {}).get('window_size', 5)
    hook_config = config.get('hook_filtering', {})
    quality_config = config.get('quality_filtering', {})
    
    # Analyze trajectory filtering
    traj_file = output_dir / "trajectory_score.jsonl"
    if traj_file.exists():
        traj_examples = sum(1 for _ in open(traj_file))
        
        # Count included traces
        included_traces = set()
        trace_scores = {}
        achievements_by_trace = {}
        
        for trace_file in traces_dir.glob("*.json"):
            trace = load_trace(trace_file)
            score = extract_trajectory_score(trace)
            trace_scores[trace_file.name] = score
            
            if score >= traj_threshold:
                included_traces.add(trace_file.name)
                # Get achievements
                metadata_list = trace.get('session_metadata', [])
                if isinstance(metadata_list, list):
                    for item in metadata_list:
                        if isinstance(item, dict) and item.get('metadata_type') == 'episode_results':
                            episode_results = item.get('data', {})
                            break
                    else:
                        episode_results = {}
                else:
                    episode_results = metadata_list.get('episode_results', {})
                
                achievements = episode_results.get('achievements', {})
                unlocked = [k for k, v in achievements.items() if v]
                achievements_by_trace[trace_file.name] = unlocked
        
        metadata["trajectory_filtering"] = {
            "threshold": traj_threshold,
            "total_traces": len(trace_scores),
            "included_traces": len(included_traces),
            "excluded_traces": len(trace_scores) - len(included_traces),
            "yield_rate": (len(included_traces) / len(trace_scores) * 100) if trace_scores else 0,
            "total_examples": traj_examples,
            "avg_examples_per_trace": traj_examples / len(included_traces) if included_traces else 0
        }
    
    # Analyze window filtering  
    window_file = output_dir / "window_score.jsonl"
    if window_file.exists():
        window_examples = sum(1 for _ in open(window_file))
        
        # Count traces with qualifying windows
        traces_with_windows = set()
        window_count = 0
        
        for trace_file in traces_dir.glob("*.json"):
            trace = load_trace(trace_file)
            llm_calls = extract_llm_calls(trace, hook_config)
            
            if llm_calls:
                max_turn = max(turn for turn, _, _ in llm_calls)
                for start in range(0, max_turn - window_size + 2):
                    end = start + window_size - 1
                    score = calculate_window_score(trace, start, end)
                    if score >= window_threshold:
                        traces_with_windows.add(trace_file.name)
                        window_count += 1
        
        metadata["window_filtering"] = {
            "window_size": window_size,
            "threshold": window_threshold,
            "total_traces": len(list(traces_dir.glob("*.json"))),
            "traces_with_qualifying_windows": len(traces_with_windows),
            "total_windows_extracted": window_count,
            "total_examples": window_examples,
            "avg_examples_per_window": window_size
        }
    
    # Save metadata
    metadata_file = config.get('output', {}).get('metadata_file', 'metadata.json')
    with open(output_dir / metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to {output_dir}/{metadata_file}")


if __name__ == "__main__":
    main()