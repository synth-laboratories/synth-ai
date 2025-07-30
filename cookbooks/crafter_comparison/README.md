# Crafter Model Comparison Cookbook

This cookbook demonstrates how to run parallel experiments comparing different language models on the Crafter environment, with robust timeout handling and performance analysis.

## Overview

This cookbook runs episodes of the Crafter game environment with different language models (gpt-4o-mini and gpt-4.1-mini) in parallel, collecting performance metrics and analyzing the results.

## Features

- **Parallel episode execution**: Runs multiple episodes simultaneously for faster experimentation
- **Timeout handling**: 
  - Turn-level timeout (20s per LLM call)
  - Episode-level timeout (180s total)
  - Action execution timeout (5s)
- **Progress tracking**: Real-time progress bars showing steps across all episodes
- **Performance comparison**: Analyzes achievements, invalid action rates, and model usage statistics
- **Deterministic seeding**: Uses consecutive seeds for reproducible experiments

## Prerequisites

1. Ensure the Crafter environment service is running:
   ```bash
   cd synth-ai/
   bash serve.sh
   ```

2. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

Run the comparison script:
```bash
./run_comparison.sh
```

Or with custom parameters:
```bash
python compare_models.py \
    --episodes 10 \
    --max-turns 100 \
    --difficulty easy \
    --models gpt-4o-mini gpt-4.1-mini \
    --base-seed 1000 \
    --turn-timeout 30.0 \
    --episode-timeout 300.0
```

## Parameters

- `--episodes`: Number of episodes per model (default: 5)
- `--max-turns`: Maximum turns per episode (default: 50)
- `--difficulty`: Game difficulty - easy, medium, hard (default: easy)
- `--models`: Models to test (default: gpt-4o-mini gpt-4.1-mini)
- `--base-seed`: Starting seed for episodes (default: 1000)
- `--turn-timeout`: Timeout per turn in seconds (default: 20.0)
- `--episode-timeout`: Total timeout per episode in seconds (default: 180.0)

## Output

The script produces:
1. Real-time progress bars showing episode execution
2. Performance summary table comparing models
3. Achievement frequency analysis
4. Model usage statistics (filtered to current experiment only)
5. JSON file with detailed results

## Example Output

```
📊 Analysis Results:
================================================================================

📈 Model Performance Summary:
Model                Avg Achievements   Max Achievements   Invalid Rate    Success Rate   
--------------------------------------------------------------------------------------
gpt-4o-mini            1.80 ± 1.33                    4            0.00%          100.00%
gpt-4.1-mini           1.80 ± 1.17                    3            0.00%          100.00%

🏆 Achievement Frequencies:
Achievement                 gpt-4.1-   gpt-4o-m
-----------------------------------------------
collect_drink               1/5   ( 20%)   3/5   ( 60%)
collect_sapling             2/5   ( 40%)   2/5   ( 40%)
collect_wood                4/5   ( 80%)   2/5   ( 40%)
```

## Implementation Details

The comparison uses:
- Async/await for parallel episode execution
- Session-based tracing with v3 architecture
- Structured output tools for consistent LLM interactions
- SQLite database for tracking model usage and costs