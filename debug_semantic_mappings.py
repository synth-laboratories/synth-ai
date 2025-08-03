#!/usr/bin/env python3
"""
Debug script to see what's actually being mapped to specific words like 
pickaxe, furnace, table in the traces.
"""

import json
import glob
import re
from collections import defaultdict

def find_word_contexts(word, max_examples=5):
    """Find actual contexts where a specific word appears in traces."""
    contexts = []
    file_count = 0
    
    for pattern in ["synth_ai/environments/examples/crafter_custom/agent_demos/traces/*.json"]:
        for json_file in glob.glob(pattern):
            if file_count >= 10:  # Limit files to check
                break
            file_count += 1
            
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Convert to string and look for our word
                text = str(data).lower()
                if word in text:
                    # Find contexts around the word
                    pattern = f".{{0,100}}{word}.{{0,100}}"
                    matches = re.findall(pattern, text)
                    
                    for match in matches[:max_examples]:
                        contexts.append({
                            'file': json_file.split('/')[-1],
                            'context': match.strip(),
                            'word': word
                        })
                        
                        if len(contexts) >= max_examples:
                            return contexts
                            
            except Exception as e:
                continue
    
    return contexts

def analyze_semantic_id_mappings():
    """Try to get the actual semantic ID mappings from Crafter."""
    try:
        import crafter
        import itertools
        
        print("🔍 Checking Crafter's internal semantic mappings...")
        
        dummyenv = crafter.Env()
        
        # Get material IDs
        print("\nMaterial IDs:")
        for name, idx in dummyenv._world._mat_ids.items():
            if name and ('pickaxe' in str(name).lower() or 'furnace' in str(name).lower() or 'table' in str(name).lower()):
                print(f"  {idx}: {name}")
        
        # Get object IDs  
        print("\nObject IDs:")
        for name, idx in dummyenv._sem_view._obj_ids.items():
            if name and ('pickaxe' in str(name).lower() or 'furnace' in str(name).lower() or 'table' in str(name).lower()):
                print(f"  {idx}: {name}")
        
        # Show all mappings
        print("\nAll Material Mappings (first 20):")
        for i, (name, idx) in enumerate(dummyenv._world._mat_ids.items()):
            if i >= 20:
                break
            print(f"  {idx}: {name}")
            
        print("\nAll Object Mappings (first 20):")
        for i, (name, idx) in enumerate(dummyenv._sem_view._obj_ids.items()):
            if i >= 20:
                break
            print(f"  {idx}: {name}")
        
        dummyenv.close()
        
    except ImportError:
        print("❌ Crafter not available for direct inspection")
    except Exception as e:
        print(f"❌ Error inspecting Crafter: {e}")

def main():
    print("🔍 Debugging Semantic Word Mappings")
    print("=" * 50)
    
    # Check actual mappings from Crafter
    analyze_semantic_id_mappings()
    
    # Find specific contexts for suspicious words
    suspicious_words = ['pickaxe', 'furnace', 'table']
    
    for word in suspicious_words:
        print(f"\n🔎 Finding contexts for '{word}':")
        print("-" * 30)
        
        contexts = find_word_contexts(word)
        
        if contexts:
            for i, ctx in enumerate(contexts, 1):
                print(f"{i}. File: {ctx['file']}")
                print(f"   Context: ...{ctx['context'][:150]}...")
                print()
        else:
            print(f"   No contexts found for '{word}'")
    
    # Also check some high-frequency words for comparison
    print(f"\n🔎 Finding contexts for 'stone' (for comparison):")
    print("-" * 30)
    
    stone_contexts = find_word_contexts('stone', max_examples=2)
    for i, ctx in enumerate(stone_contexts, 1):
        print(f"{i}. File: {ctx['file']}")
        print(f"   Context: ...{ctx['context'][:150]}...")
        print()

if __name__ == "__main__":
    main()