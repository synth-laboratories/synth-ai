"""Example: Running prompt learning jobs programmatically.

This example shows how to use the SDK to run MIPRO or GEPA optimization
in a Python script instead of via CLI.
"""

import os
from synth_ai.api.train.prompt_learning import PromptLearningJob


def main() -> None:
    """Run a prompt learning job programmatically."""
    # Initialize job from config
    job = PromptLearningJob.from_config(
        config_path="examples/blog_posts/gepa/configs/banking77_gepa_local.toml",
        backend_url=os.environ.get("BACKEND_BASE_URL", "https://api.usesynth.ai"),
        api_key=os.environ["SYNTH_API_KEY"],
    )
    
    # Submit job
    print("Submitting job...")
    job_id = job.submit()
    print(f"Job submitted: {job_id}")
    
    # Poll until complete with status callback
    def on_status(status: dict) -> None:
        print(f"Status: {status.get('status', 'unknown')}")
        if 'best_score' in status:
            print(f"Best score so far: {status['best_score']}")
    
    print("Polling until complete...")
    final_status = job.poll_until_complete(
        timeout=3600.0,
        interval=5.0,
        on_status=on_status,
    )
    
    print(f"\nJob completed with status: {final_status.get('status')}")
    
    # Get results
    print("\nFetching results...")
    results = job.get_results()
    
    print(f"Best score: {results.get('best_score')}")
    print(f"Top prompts: {len(results.get('top_prompts', []))}")
    
    # Get best prompt text
    best_prompt = job.get_best_prompt_text(rank=1)
    if best_prompt:
        print(f"\nBest prompt:\n{best_prompt}")


if __name__ == "__main__":
    main()

