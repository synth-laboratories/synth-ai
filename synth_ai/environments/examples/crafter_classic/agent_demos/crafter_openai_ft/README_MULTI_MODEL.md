# Multi-Model Crafter Evaluation

This script runs Crafter rollouts for multiple language models and provides comprehensive performance comparisons.

## Features

### Models Tested
- **gpt-4o-mini** - OpenAI's efficient small model
- **gpt-4.1-mini** - OpenAI's improved mini model  
- **gpt-4.1-nano** - OpenAI's ultra-efficient model
- **gemini-1.5-flash** - Google's fast Gemini model
- **gemini-2.5-flash-lite** - Google's newest lightweight model
- **qwen3/32b** - Alibaba's large parameter model

### Metrics Analyzed
- **Invalid Action Rates** - How often models choose invalid actions
- **Achievement Frequencies** - Achievement unlocking patterns by game step
- **Achievement Counts** - Total and unique achievements per model
- **Performance Metrics** - Reward, steps, success rates, duration
- **Model Rankings** - Comparative performance rankings

### Data Collection
- **DuckDB Integration** - All traces stored in structured database
- **Experiment Tracking** - Each model run tracked as separate experiment
- **V2 Tracing** - Advanced tracing with hooks for achievement detection
- **Detailed Logging** - Step-by-step action and state logging

## Quick Start

### Prerequisites

1. **Crafter Service Running**
   ```bash
   # Make sure Crafter service is running on localhost:8901
   # See Crafter documentation for setup
   ```

2. **Dependencies Installed**
   ```bash
   pip install httpx tqdm numpy pandas duckdb
   ```

### Basic Usage

```bash
# Run with default settings (10 episodes per model)
python run_rollouts_for_models_and_compare.py

# Quick test with fewer episodes
python run_rollouts_for_models_and_compare.py --episodes 3 --max-turns 50

# Test specific models only
python run_rollouts_for_models_and_compare.py --models gpt-4o-mini gemini-1.5-flash

# Use custom service URL and database
python run_rollouts_for_models_and_compare.py \
    --service-url http://localhost:8902 \
    --database my_experiments.duckdb
```

### ⚡ High-Performance Async Mode

The script runs **fully asynchronously** for maximum performance:

```bash
# Episodes run concurrently within each model (default behavior)
python run_rollouts_for_models_and_compare.py --episodes 10

# Also run multiple models concurrently (fastest)
python run_rollouts_for_models_and_compare.py --concurrent-models

# Limit concurrent models to avoid overwhelming system
python run_rollouts_for_models_and_compare.py --concurrent-models --max-concurrent-models 2

# Maximum performance: concurrent models + more episodes
python run_rollouts_for_models_and_compare.py \
    --concurrent-models \
    --max-concurrent-models 3 \
    --episodes 20
```

**Performance Comparison:**
- **Sequential**: ~60 minutes for 6 models × 10 episodes
- **Concurrent Episodes**: ~15 minutes for 6 models × 10 episodes  
- **Concurrent Models + Episodes**: ~5 minutes for 6 models × 10 episodes

## Configuration

### Command Line Options

```bash
python run_rollouts_for_models_and_compare.py --help

Options:
  --episodes N               Number of episodes per model (default: 10)
  --max-turns N              Maximum turns per episode (default: 100)
  --models MODEL [...]       Space-separated list of models to test
  --database PATH            Database file path (default: crafter_multi_model_traces.duckdb)
  --service-url URL          Crafter service URL (default: http://localhost:8901)
  --concurrent-models        Run models concurrently (default: sequential)
  --max-concurrent-models N  Maximum concurrent models (default: 3)
```

### Configuration File

Edit `multi_model_config.toml` to customize default settings:

```toml
[experiment]
episodes = 10
max_turns = 100
difficulty = "easy"

[services]
crafter_service_url = "http://localhost:8901"
database_path = "crafter_multi_model_traces.duckdb"
```

## Output Format

### Terminal Output

The script provides real-time progress and comprehensive results:

```
🚀 STARTING MULTI-MODEL CRAFTER EVALUATION
   Models: ['gpt-4o-mini', 'gpt-4.1-mini', ...]
   Episodes per model: 10
   Max turns per episode: 100

🚀 Starting experiment for gpt-4o-mini
📍 Episode 1/10 for gpt-4o-mini
gpt-4o-mini Ep 1: 100%|████████| 100/100 [02:15<00:00, 0.74it/s]
...

====================================================================================================
🏆 MULTI-MODEL CRAFTER EVALUATION RESULTS
====================================================================================================

📊 PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Model                Episodes   Success%   Avg Reward   Avg Steps    Avg Duration
--------------------------------------------------------------------------------
gpt-4o-mini          10         100.0%     15.30        87.2         142.5s
gpt-4.1-mini          10         100.0%     18.75        91.4         156.8s
...

🚫 INVALID ACTION ANALYSIS
--------------------------------------------------------------------------------
Model                Avg Invalid%    Total Invalid   Total Actions  
--------------------------------------------------------------------------------
gpt-4o-mini          12.45%          108             867
gpt-4.1-mini          8.32%          76              913
...

🏅 ACHIEVEMENT ANALYSIS
--------------------------------------------------------------------------------
Model                Total Ach.   Unique Ach.   Early (0-25)  Mid (26-50)   Late (51+)  
--------------------------------------------------------------------------------
gpt-4o-mini          34           12            8             15            11
gpt-4.1-mini          42           15            12            18            12
...

🥇 MODEL RANKINGS
--------------------------------------------------
By Average Reward:
  1. gpt-4.1-mini: 18.75
  2. gemini-1.5-flash: 16.42
  3. gpt-4o-mini: 15.30
  ...

By Invalid Action Rate (lower is better):
  1. gemini-2.5-flash-lite: 5.23%
  2. gpt-4.1-mini: 8.32%
  3. qwen3/32b: 11.67%
  ...
```

### File Outputs

1. **Results JSON** - Complete results with analysis
   ```
   traces_multi_model/multi_model_results_20240101_120000.json
   ```

2. **DuckDB Database** - All traces and experiment data
   ```
   crafter_multi_model_traces.duckdb
   ```

3. **Individual Traces** - Session traces for each episode
   ```
   traces_multi_model/trace_episode_0.json
   traces_multi_model/trace_episode_1.json
   ...
   ```

## Advanced Analysis

### Database Queries

After running experiments, you can analyze the data further:

```python
import duckdb

# Connect to results database
conn = duckdb.connect("crafter_multi_model_traces.duckdb")

# Get experiment summary
experiments = conn.execute("""
    SELECT e.name, e.id, e.created_at, 
           COUNT(st.session_id) as session_count
    FROM experiments e
    LEFT JOIN session_traces st ON e.id = st.experiment_id
    GROUP BY e.id, e.name, e.created_at
    ORDER BY e.created_at DESC
""").df()

print(experiments)

# Analyze model performance
model_performance = conn.execute("""
    SELECT 
        e.name as experiment_name,
        AVG(st.num_timesteps) as avg_steps,
        COUNT(st.session_id) as total_sessions
    FROM experiments e
    JOIN session_traces st ON e.id = st.experiment_id
    GROUP BY e.name
    ORDER BY avg_steps DESC
""").df()

print(model_performance)
```

### Custom Analysis

Use the existing comparison tools:

```python
# Use compare_experiments.py for detailed analysis
from compare_experiments import main as compare_main

# Use filter_traces_sft_duckdb.py for filtering high-performance traces
from filter_traces_sft_duckdb import main as filter_main
```

## Architecture

### Components

1. **Experiment Runner** - Orchestrates multi-model evaluation
2. **Episode Runner** - Handles individual episode execution
3. **Trace Manager** - Manages DuckDB storage and retrieval
4. **Analysis Engine** - Computes metrics and comparisons
5. **Result Formatter** - Generates human-readable output

### Data Flow

```
Models → Episodes → Actions → Environment → Rewards/Achievements → Traces → Database → Analysis → Results
```

### Async Architecture

The script uses a **multi-level async design** for optimal performance:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model 1       │    │   Model 2       │    │   Model 3       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Episode 1   │ │    │ │ Episode 1   │ │    │ │ Episode 1   │ │
│ │ Episode 2   │ │    │ │ Episode 2   │ │    │ │ Episode 2   │ │
│ │ Episode 3   │ │    │ │ Episode 3   │ │    │ │ Episode 3   │ │
│ │ ...         │ │    │ │ ...         │ │    │ │ ...         │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│   (concurrent)  │    │   (concurrent)  │    │   (concurrent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      concurrent             concurrent             concurrent
```

**Level 1: Episode Concurrency** (Always enabled)
- All episodes for a model run concurrently
- Each episode has its own SessionTracer
- Database writes use retry logic for contention handling

**Level 2: Model Concurrency** (Optional with `--concurrent-models`)
- Multiple models run simultaneously
- Semaphore limits concurrent models to prevent resource exhaustion
- Progress tracking across all concurrent operations

### Integration Points

- **LM Class** - Language model interface with v2 tracing
- **Crafter Service** - Environment service API
- **DuckDB** - Trace storage and querying
- **Session Tracer** - Advanced tracing with hooks
- **Achievement Hooks** - Automated achievement detection

## Troubleshooting

### Common Issues

1. **Service Connection Errors**
   ```
   ❌ HTTP request failed after 3 attempts
   ```
   - Check if Crafter service is running
   - Verify service URL is correct
   - Check network connectivity

2. **Model API Errors**
   ```
   ❌ LM call failed: API key not found
   ```
   - Ensure API keys are set in environment
   - Check model names are correct
   - Verify model access permissions

3. **Database Errors**
   ```
   Warning: Could not create experiment in DB
   ```
   - Check write permissions for database file
   - Ensure DuckDB is properly installed
   - Check disk space

### Debug Mode

For detailed debugging:

```bash
# Enable verbose output
python run_rollouts_for_models_and_compare.py --episodes 1 --max-turns 10

# Check individual traces
ls traces_multi_model/
cat traces_multi_model/trace_episode_0.json
```

## Performance Optimization

### For Large-Scale Evaluation

1. **Reduce Episodes for Testing**
   ```bash
   --episodes 3 --max-turns 25
   ```

2. **Run Subset of Models**
   ```bash
   --models gpt-4o-mini gemini-1.5-flash
   ```

3. **Parallel Execution**
   - Models run sequentially (for fair comparison)
   - Episodes within model run sequentially (for trace integrity)
   - Consider running multiple instances for different model subsets

### Memory Management

- Database auto-commits after each session
- Traces stored efficiently in compressed format
- Memory usage scales with episode length, not total episodes

## Integration with Existing Tools

This script integrates seamlessly with existing analysis tools:

- **compare_experiments.py** - For detailed experiment comparison
- **filter_traces_sft_duckdb.py** - For extracting training data
- **Existing DuckDB infrastructure** - All standard queries work

Use the experiment IDs output by the script with existing comparison tools for deeper analysis. 