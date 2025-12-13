#!/usr/bin/env python3
"""CLI runner for Graph Optimization jobs.

Usage:
    # Run with config file:
    python -m synth_ai.products.graph_gepa.run --config config.toml
    
    # Or use the synth CLI:
    synth graph-optimization run --config config.toml
    
    # With custom backend:
    python -m synth_ai.products.graph_gepa.run --config config.toml --backend http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from .config import GraphOptimizationConfig
from .client import GraphOptimizationClient


async def run_optimization(
    config_path: str,
    backend_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Run graph optimization from config file.
    
    Args:
        config_path: Path to TOML config file
        backend_url: Backend API URL
        api_key: Optional API key
        verbose: Print verbose output
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load config
    try:
        config = GraphOptimizationConfig.from_toml(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}", file=sys.stderr)
        return 1
    
    print("=" * 60)
    print(f"Graph Optimization ({config.algorithm}) - {config.dataset_name}")
    print("=" * 60)
    print()
    print(f"  Graph type: {config.graph_type.value}")
    print(f"  Graph structure: {config.graph_structure.value}")
    print(f"  Generations: {config.evolution.num_generations}")
    print(f"  Children per gen: {config.evolution.children_per_generation}")
    print(f"  Train seeds: {len(config.seeds.train)}")
    print(f"  Validation seeds: {len(config.seeds.validation)}")
    print(f"  Proposer model: {config.proposer.model}")
    print()
    
    print(f"  Algorithm: {config.algorithm}")
    
    # Run optimization
    async with GraphOptimizationClient(backend_url, api_key) as client:
        # Start job
        print("🚀 Starting optimization job...")
        try:
            job_id = await client.start_job(config)
            print(f"   Job ID: {job_id}")
        except Exception as e:
            print(f"❌ Failed to start job: {e}", file=sys.stderr)
            return 1
        
        # Stream events
        print()
        print("📊 Progress:")
        current_gen = 0
        best_score = 0.0
        
        try:
            async for event in client.stream_events(job_id):
                event_type = event.get("type", "")
                data = event.get("data", {})
                
                if event_type == "job_started":
                    print(f"   ✓ Job started for dataset: {data.get('dataset_name')}")
                
                elif event_type == "context_loaded":
                    print(f"   ✓ Context loaded: {data.get('task_name', 'unknown')}")
                
                elif event_type == "initial_graph_generated":
                    yaml_len = data.get("yaml_length", 0)
                    print(f"   ✓ Initial graph generated ({yaml_len} chars)")
                
                elif event_type == "generation_started":
                    gen = data.get("generation", 0)
                    current_gen = gen
                    print()
                    print(f"   🧬 Generation {gen}")
                
                elif event_type == "candidate_evaluated":
                    cid = data.get("candidate_id", "?")[:8]
                    score = data.get("score", 0)
                    print(f"      Candidate {cid}: {score:.4f}")
                    if score > best_score:
                        best_score = score
                
                elif event_type == "generation_completed":
                    gen = data.get("generation", 0)
                    gen_best = data.get("best_score", 0)
                    print(f"      → Generation {gen} complete. Best: {gen_best:.4f}")
                
                elif event_type == "job_completed":
                    final_score = data.get("best_score", 0)
                    print()
                    print(f"   ✅ Optimization complete!")
                    print(f"      Best score: {final_score:.4f}")
                
                elif event_type == "job_failed":
                    error = data.get("error", "Unknown error")
                    print()
                    print(f"   ❌ Job failed: {error}")
                    return 1
                
                elif verbose:
                    print(f"   [{event_type}] {data}")
        
        except Exception as e:
            print(f"\n❌ Stream error: {e}", file=sys.stderr)
            # Try to get final result anyway
        
        # Get final result
        print()
        print("-" * 60)
        try:
            result = await client.get_result(job_id)
            print("📋 Final Results:")
            print(f"   Status: {result.get('status')}")
            print(f"   Best score: {result.get('best_score', 0):.4f}")
            print(f"   Generations: {result.get('generations_completed', 0)}")
            print(f"   Candidates evaluated: {result.get('total_candidates_evaluated', 0)}")
            print(f"   Duration: {result.get('duration_seconds', 0):.1f}s")
            
            if result.get("policy_id"):
                print(f"   Policy ID: {result.get('policy_id')}")
            
            if result.get("error"):
                print(f"   ⚠️ Error: {result.get('error')}")
            
            # Save best YAML if available
            if result.get("best_yaml"):
                output_dir = Path(config_path).parent / "output"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"best_graph_{job_id}.yaml"
                with open(output_file, "w") as f:
                    f.write(result["best_yaml"])
                print(f"   📁 Saved: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Could not retrieve final result: {e}", file=sys.stderr)
            return 1
    
    print()
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run graph optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config:
    python -m synth_ai.products.graph_gepa.run --config config.toml
    
    # Run with custom backend:
    python -m synth_ai.products.graph_gepa.run --config config.toml --backend http://api.example.com
    
    # Verbose output:
    python -m synth_ai.products.graph_gepa.run --config config.toml --verbose
        """,
    )
    
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--backend", "-b",
        default=os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000"),
        help="Backend API URL (default: $SYNTH_BACKEND_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("SYNTH_API_KEY"),
        help="API key for authentication (default: $SYNTH_API_KEY)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output",
    )
    
    args = parser.parse_args()
    
    return asyncio.run(run_optimization(
        config_path=args.config,
        backend_url=args.backend,
        api_key=args.api_key,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    sys.exit(main())

