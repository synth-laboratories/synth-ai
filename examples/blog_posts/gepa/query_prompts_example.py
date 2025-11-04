"""
Example script showing how to query prompt learning job results.

Usage:
    python query_prompts_example.py pl_9c58b711c2644083
"""

import os
import sys
from pprint import pprint

from synth_ai.learning import get_prompts, get_prompt_text, get_scoring_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_prompts_example.py <job_id>")
        print("Example: python query_prompts_example.py pl_9c58b711c2644083")
        sys.exit(1)
    
    job_id = sys.argv[1]
    
    # Get credentials from environment
    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("SYNTH_API_KEY")
    
    if not api_key:
        print("Error: SYNTH_API_KEY environment variable not set")
        sys.exit(1)
    
    print(f"Querying job: {job_id}")
    print(f"Backend: {base_url}")
    print("=" * 80)
    
    # Get all prompts and metadata
    print("\n📊 Fetching prompt results...")
    results = get_prompts(job_id, base_url, api_key)
    
    # Print best score
    if results.best_score is not None:
        print(f"\n🏆 Best Score: {results.best_score:.3f} ({results.best_score * 100:.1f}%)")
    
    # Print top-K prompts with scores
    top_prompts = results.top_prompts
    if top_prompts:
        print(f"\n📝 Top {len(top_prompts)} Prompts:")
        print("=" * 80)
        for prompt_info in sorted(top_prompts, key=lambda p: p.get("rank", 999)):
            rank = prompt_info["rank"]
            train_accuracy = prompt_info.get("train_accuracy")
            val_accuracy = prompt_info.get("val_accuracy")
            
            print(f"\nRank #{rank}:")
            if train_accuracy is not None:
                print(f"  Train Accuracy: {train_accuracy:.3f} ({train_accuracy * 100:.1f}%)")
            if val_accuracy is not None:
                print(f"  Val Accuracy:   {val_accuracy:.3f} ({val_accuracy * 100:.1f}%)")
            print(f"  Prompt Text:")
            print("  " + "-" * 76)
            full_text = prompt_info.get("full_text", "")
            for line in full_text.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 76)
    
    # Get scoring summary
    print("\n📈 Scoring Summary:")
    print("=" * 80)
    summary = get_scoring_summary(job_id, base_url, api_key)
    
    print(f"Best Train Accuracy:     {summary['best_train_accuracy']:.3f} ({summary['best_train_accuracy'] * 100:.1f}%)")
    if summary['best_val_accuracy']:
        print(f"Best Val Accuracy:       {summary['best_val_accuracy']:.3f} ({summary['best_val_accuracy'] * 100:.1f}%)")
    print(f"Mean Train Accuracy:     {summary['mean_train_accuracy']:.3f} ({summary['mean_train_accuracy'] * 100:.1f}%)")
    print(f"Candidates Tried:        {summary['num_candidates_tried']}")
    print(f"Frontier Candidates:     {summary['num_frontier_candidates']}")
    
    print(f"\nScore Distribution:")
    for bin_range, count in summary['score_distribution'].items():
        bar = "█" * count
        print(f"  {bin_range}: {count:3d} {bar}")
    
    # Quick access to best prompt text only
    print("\n💡 Quick access to best prompt:")
    print("=" * 80)
    best_text = get_prompt_text(job_id, base_url, api_key, rank=1)
    if best_text:
        print(best_text)
    else:
        print("Best prompt text not available yet (job may still be running)")
    
    print("\n" + "=" * 80)
    print("✅ Query complete!")


if __name__ == "__main__":
    main()

