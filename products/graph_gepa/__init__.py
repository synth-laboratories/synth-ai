"""Graph Optimization - Optimize LLM workflow graphs.

This product provides tools for optimizing LLM-based workflow graphs
using various algorithms. It can optimize both:

- **Policy graphs**: Graphs that solve tasks (e.g., multi-hop QA, reasoning)
- **Verifier graphs**: Graphs that judge/score existing results

Algorithms:
- `graph_gepa`: Grammatical Evolution for graph structure optimization

Usage:
    # From command line:
    synth graph-optimization run --config config.toml
    
    # Or programmatically:
    from synth_ai.products.graph_gepa import GraphOptimizationClient, GraphOptimizationConfig
    
    config = GraphOptimizationConfig.from_toml("config.toml")
    async with GraphOptimizationClient(base_url, api_key) as client:
        job_id = await client.start_job(config)
        async for event in client.stream_events(job_id):
            print(event)
"""

from .config import GraphOptimizationConfig
from .client import GraphOptimizationClient

__all__ = ["GraphOptimizationConfig", "GraphOptimizationClient"]

