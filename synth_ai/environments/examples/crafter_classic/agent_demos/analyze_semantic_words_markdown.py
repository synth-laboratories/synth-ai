#!/usr/bin/env python3
"""
Run Crafter agent and analyze semantic map words - output as markdown tables only.

This script:
1. Runs a Crafter agent for multiple episodes
2. Extracts all unique words from the semantic map observations
3. Outputs analysis as markdown tables (no plotting dependencies)

Usage:
    python analyze_semantic_words_markdown.py --model gemini-1.5-flash --episodes 3
"""

import argparse
import asyncio
import json
import re

# Import the Crafter agent
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(str(Path(__file__).parent))
from test_crafter_react_agent import run_crafter_episodes


def extract_words_from_semantic_map(observation: str) -> set[str]:
    """Extract meaningful words from a semantic map observation string."""
    if not observation or "semantic_map" not in observation.lower():
        return set()
    
    # Look for patterns like object names in the semantic map
    # Common Crafter objects/entities
    crafter_words = {
        # Resources
        'wood', 'stone', 'coal', 'iron', 'diamond', 'water',
        # Animals  
        'cow', 'pig', 'skeleton', 'zombie',
        # Structures/Objects
        'tree', 'grass', 'furnace', 'table', 'bed', 'chest',
        'house', 'fence', 'door', 'wall',
        # Tools
        'axe', 'pickaxe', 'sword', 'shovel',
        # Food
        'bread', 'meat', 'apple',
        # Environment
        'mountain', 'river', 'forest', 'desert', 'cave',
        'lava', 'sand', 'dirt', 'path',
        # Actions/States
        'crafting', 'mining', 'building', 'farming',
        'health', 'hunger', 'energy'
    }
    
    # Extract words using regex - look for alphabetic words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', observation.lower())
    
    # Filter to keep only meaningful Crafter-related words
    found_words = set()
    for word in words:
        if word in crafter_words or any(cw in word for cw in crafter_words):
            found_words.add(word)
    return found_words

def analyze_episode_traces(traces_data: list[dict]) -> dict[str, int]:
    """Analyze traces to extract semantic map words."""
    word_counter = Counter()
    
    for episode_data in traces_data:
        if 'observations' in episode_data:
            for obs in episode_data['observations']:
                if isinstance(obs, dict):
                    # Look for semantic map in observation
                    obs_str = str(obs)
                    words = extract_words_from_semantic_map(obs_str)
                    word_counter.update(words)
                elif isinstance(obs, str):
                    words = extract_words_from_semantic_map(obs)
                    word_counter.update(words)
    
    return dict(word_counter)

def generate_markdown_report(word_counts: dict[str, int], model: str, episodes: int) -> str:
    """Generate a markdown report of the semantic map analysis."""
    if not word_counts:
        return "# Semantic Map Analysis\n\n**No words found in semantic maps!**\n"
    
    total_words = sum(word_counts.values())
    unique_words = len(word_counts)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Generate markdown
    md = f"""# Semantic Map Word Analysis
    
**Model:** {model}  
**Episodes:** {episodes}  
**Generated:** {timestamp}  

## Summary

- **Total word occurrences:** {total_words}
- **Unique words discovered:** {unique_words}
- **Average occurrences per word:** {total_words/unique_words:.1f}

## Top Words by Frequency

| Rank | Word | Count | Percentage |
|------|------|-------|------------|
"""
    
    # Top 15 words table
    for i, (word, count) in enumerate(sorted_words[:15], 1):
        percentage = (count / total_words) * 100
        md += f"| {i:2d} | {word} | {count} | {percentage:.1f}% |\n"
    
    # Word categories
    categories = {
        "Resources": ['wood', 'stone', 'coal', 'iron', 'diamond', 'water'],
        "Animals": ['cow', 'pig', 'skeleton', 'zombie'],
        "Structures": ['tree', 'furnace', 'table', 'house', 'chest', 'fence', 'door'],
        "Tools": ['axe', 'pickaxe', 'sword', 'shovel'],
        "Environment": ['mountain', 'river', 'forest', 'desert', 'cave', 'lava', 'grass'],
        "Food": ['bread', 'meat', 'apple']
    }
    
    md += "\n## Words by Category\n\n"
    
    for category, words in categories.items():
        found_words = [(w, word_counts[w]) for w in words if w in word_counts]
        if found_words:
            md += f"### {category}\n\n"
            md += "| Word | Count |\n|------|-------|\n"
            for word, count in sorted(found_words, key=lambda x: x[1], reverse=True):
                md += f"| {word} | {count} |\n"
            md += "\n"
    
    # Frequency distribution
    freq_counts = Counter(word_counts.values())
    md += "## Frequency Distribution\n\n"
    md += "| Frequency | Number of Words |\n|-----------|----------------|\n"
    for freq in sorted(freq_counts.keys(), reverse=True):
        md += f"| {freq} | {freq_counts[freq]} |\n"
    
    # All words alphabetically
    md += "\n## All Words (Alphabetical)\n\n"
    md += "| Word | Count |\n|------|-------|\n"
    for word in sorted(word_counts.keys()):
        md += f"| {word} | {word_counts[word]} |\n"
    
    return md

async def main():
    parser = argparse.ArgumentParser(description="Analyze semantic map words - markdown output only")
    parser.add_argument("--model", default="gemini-1.5-flash", 
                       help="Model to use for agent (default: gemini-1.5-flash)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to run (default: 3)")
    parser.add_argument("--max-turns", type=int, default=50,
                       help="Maximum turns per episode (default: 50)")
    parser.add_argument("--output-dir", default="semantic_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    print(f"🚀 Running {args.episodes} episodes with {args.model}")
    print("📊 Will analyze semantic map words and generate markdown report")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run the agent episodes
    try:
        print("\n🎮 Starting Crafter episodes...")
        traces_result = await run_crafter_episodes(
            model_name=args.model,
            num_episodes=args.episodes,
            max_turns=args.max_turns,
            difficulty="easy",
            base_seed=1000
        )
        
        print(f"✅ Completed {args.episodes} episodes")
        
        # Analyze semantic map words
        print("\n🔍 Analyzing semantic map words...")
        word_counts = analyze_episode_traces(traces_result)
        
        # Generate markdown report
        print("\n📝 Generating markdown report...")
        markdown_report = generate_markdown_report(word_counts, args.model, args.episodes)
        
        # Save markdown report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"semantic_analysis_{args.model}_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(markdown_report)
        
        print(f"💾 Markdown report saved to: {report_file}")
        
        # Also save raw data as JSON
        analysis_data = {
            "model": args.model,
            "episodes": args.episodes,
            "timestamp": timestamp,
            "word_counts": word_counts,
            "total_unique_words": len(word_counts),
            "total_word_occurrences": sum(word_counts.values())
        }
        
        json_file = output_dir / f"word_data_{args.model}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"💾 Raw data saved to: {json_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("SEMANTIC MAP WORD ANALYSIS SUMMARY")
        print("="*60)
        
        if word_counts:
            total_words = sum(word_counts.values())
            unique_words = len(word_counts)
            print(f"Total word occurrences: {total_words}")
            print(f"Unique words discovered: {unique_words}")
            
            # Top 10 most common words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 10 most frequent words:")
            for i, (word, count) in enumerate(sorted_words[:10], 1):
                print(f"{i:2d}. {word:<12} ({count} times)")
        else:
            print("No semantic map words found!")
        
        print(f"\n📄 Full analysis available in: {report_file}")
        print("\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
