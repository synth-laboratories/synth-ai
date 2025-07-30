#!/usr/bin/env python3
"""
Display statistics about Crafter datasets
"""

import json
from pathlib import Path
from collections import Counter
import sys


def analyze_dataset(dataset_path: Path):
    """Analyze a single dataset"""
    # Load metadata
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load instances
    with open(dataset_path / "instances.json", "r") as f:
        instances = json.load(f)
    
    print(f"\nDataset: {metadata['name']}")
    print(f"Description: {metadata['description']}")
    print(f"Total instances: {metadata['num_instances']}")
    print(f"Train/Val/Test split: {len(instances) - len(metadata['split_info']['val_instance_ids']) - len(metadata['split_info']['test_instance_ids'])}/{len(metadata['split_info']['val_instance_ids'])}/{len(metadata['split_info']['test_instance_ids'])}")
    
    # Analyze by difficulty
    difficulties = Counter(inst['metadata']['difficulty'] for inst in instances)
    print(f"\nInstances by difficulty:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count} ({count/len(instances)*100:.1f}%)")
    
    # Analyze by impetus type
    impetus_types = Counter()
    for inst in instances:
        instructions = inst['impetus']['instructions'].lower()
        if 'speedrun' in instructions:
            impetus_types['speedrun'] += 1
        elif 'focus on' in instructions:
            impetus_types['focused'] += 1
        else:
            impetus_types['general'] += 1
    
    print(f"\nInstances by type:")
    for type_name, count in sorted(impetus_types.items()):
        print(f"  {type_name}: {count} ({count/len(instances)*100:.1f}%)")
    
    # Analyze achievement focuses
    focus_counts = Counter()
    speedrun_targets = Counter()
    
    for inst in instances:
        if inst['impetus'].get('achievement_focus'):
            for ach in inst['impetus']['achievement_focus']:
                focus_counts[ach] += 1
        
        if 'speedrun' in inst['impetus']['instructions'].lower():
            # Extract speedrun target
            instructions = inst['impetus']['instructions']
            if ':' in instructions:
                target = instructions.split(':')[1].strip().split(' ')[0]
                speedrun_targets[target] += 1
    
    if focus_counts:
        print(f"\nTop achievement focuses:")
        for ach, count in focus_counts.most_common(10):
            print(f"  {ach}: {count}")
    
    if speedrun_targets:
        print(f"\nSpeedrun targets:")
        for target, count in speedrun_targets.most_common():
            print(f"  {target}: {count}")
    
    # Sample some instances
    print(f"\nSample instances:")
    for i, inst in enumerate(instances[:3]):
        print(f"\n  Instance {i+1}:")
        print(f"    ID: {inst['id']}")
        print(f"    Difficulty: {inst['metadata']['difficulty']}")
        print(f"    Seed: {inst['metadata']['world_seed']}")
        print(f"    Instructions: {inst['impetus']['instructions'][:80]}...")


def main():
    dataset_dir = Path("dataset")
    
    print("Crafter Dataset Statistics")
    print("=" * 60)
    
    # Find all datasets
    datasets = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]
    
    if not datasets:
        print("No datasets found in dataset/")
        return
    
    print(f"Found {len(datasets)} dataset(s)")
    
    for dataset_path in sorted(datasets):
        analyze_dataset(dataset_path)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()