#!/usr/bin/env python3
"""
Simple semantic word extraction from Crafter observations.
Just extract and count words - no complex agent running needed.
"""

import re
from collections import Counter
from pathlib import Path
import json

def extract_semantic_words(text: str) -> dict:
    """Extract semantic words and return counts."""
    
    # Common Crafter entities to look for
    target_words = {
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
        'lava', 'sand', 'dirt', 'path'
    }
    
    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_counts = Counter()
    
    for word in words:
        if word in target_words:
            word_counts[word] += 1
    
    return dict(word_counts)

def analyze_existing_traces():
    """Look for existing trace files and analyze them."""
    import glob
    
    # Look for trace files in multiple locations
    trace_patterns = [
        "synth_ai/environments/examples/crafter_classic/agent_demos/traces/*.json",
        "synth_ai/environments/examples/crafter_custom/agent_demos/traces/*.json",
        "*/traces/*.json",
        "traces/*.json"
    ]
    
    all_words = Counter()
    total_files = 0
    
    for pattern in trace_patterns:
        for json_file in glob.glob(pattern):
            total_files += 1
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    # Convert entire data to string and extract words
                    text = str(data)
                    words = extract_semantic_words(text)
                    all_words.update(words)
                    if total_files <= 3:  # Show first few files for debugging
                        print(f"Analyzed: {json_file}")
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    
    print(f"Analyzed {total_files} total trace files")
    return dict(all_words)

def demo_with_sample_observations():
    """Demo with some sample Crafter observations."""
    sample_observations = [
        "Player sees tree, stone, grass, water nearby. Inventory: wood=3, stone=1",
        "Found cow near river. Mountain visible to north. Has axe in hand.",
        "Built furnace next to house. Coal burning. Diamond ore in cave detected.",
        "Zombie approaching! Skeleton behind tree. Sword ready. Health low.",
        "Crafting table placed. Bread cooking. Pig wandering in forest.",
        "Door opened to chest. Iron tools inside. Fence around garden."
    ]
    
    all_words = Counter()
    for obs in sample_observations:
        words = extract_semantic_words(obs)
        all_words.update(words)
    
    return dict(all_words)

def main():
    print("🔍 Simple Semantic Word Analysis")
    print("=" * 40)
    
    # Try to analyze existing traces first
    word_counts = analyze_existing_traces()
    
    # If no traces found, use demo data
    if not word_counts:
        print("No trace files found, using demo observations...")
        word_counts = demo_with_sample_observations()
    
    if not word_counts:
        print("No semantic words found!")
        return
    
    # Calculate totals
    total_occurrences = sum(word_counts.values())
    
    print(f"Total word occurrences: {total_occurrences}")
    print(f"Unique words found: {len(word_counts)}")
    print()
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Word Distribution:")
    print("-" * 20)
    
    for word, count in sorted_words:
        percentage = (count / total_occurrences) * 100 if total_occurrences > 0 else 0
        print(f'"{word}": {count}/{total_occurrences} ({percentage:.1f}%)')

if __name__ == "__main__":
    main()