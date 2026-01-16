"""GraphEvolve - Optimize LLM workflow graphs.

This product provides tools for optimizing LLM-based workflow graphs
using evolutionary algorithms. It can optimize both:

- **Policy graphs**: Graphs that solve tasks (e.g., multi-hop QA, reasoning)
- **Verifier graphs**: Graphs that verify/score existing results

Algorithms:
- `graph_evolve`: Evolutionary optimization for graph structure

Usage:
    # From command line:
    synth graph-optimization run --config config.toml

    # Or programmatically:
    from synth_ai.products.graph_evolve import GraphOptimizationClient, GraphOptimizationConfig

    config = GraphOptimizationConfig.from_toml("config.toml")
    async with GraphOptimizationClient(base_url, synth_user_key) as client:
        job_id = await client.start_job(config)
        async for event in client.stream_events(job_id):
            print(event)
"""

from .client import GraphOptimizationClient
from .config import GraphOptimizationConfig
from .converters import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    convert_openai_sft,
    preview_conversion,
)

__all__ = [
    "GraphOptimizationConfig",
    "GraphOptimizationClient",
    # Converters
    "convert_openai_sft",
    "preview_conversion",
    "ConversionResult",
    "ConversionWarning",
    "ConversionError",
]
