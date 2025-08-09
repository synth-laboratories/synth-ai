# Crafter Fine-tuning Workflow Guide

This directory contains a complete pipeline for fine-tuning models on Crafter environment data using Modal/Synth services.

## 🎯 Overview

The workflow consists of several key components:
1. **Data Generation**: Generate and filter Crafter traces
2. **Fine-tuning**: Train models using Modal/Synth services
3. **Evaluation**: Test fine-tuned models
4. **Analysis**: Compare experiments and analyze results

## 📋 Prerequisites

- Modal account with GPU access
- Synth API key: `sk-test-11111111111111111111111111111111`
- Environment service running on `http://localhost:8901`
- DuckDB for trace filtering and analysis

## 🚀 Quick Start

### 1. Generate Training Data

```bash
# Run Crafter ReAct agent to generate traces
python test_crafter_react_agent_openai.py --model gpt-4o-mini --episodes 100

# Filter traces for quality
python filter_traces_sft_duckdb.py --input-dir traces_v2_synth --output crafter_quality_sft.jsonl --min-achievements 3
```

### 2. Fine-tune Models

```bash
# Use Modal/Synth service
python kick_off_ft_modal.py crafter_quality_sft.jsonl --model Qwen/Qwen2.5-7B-Instruct --epochs 3

# Or use OpenAI service
python kick_off_ft_oai.py crafter_quality_sft.jsonl --model gpt-4o-mini
```

### 3. Evaluate Fine-tuned Models

```bash
# Test fine-tuned model
python test_crafter_react_agent_openai.py --model ft:qwen2-5-7b-instruct:org-test123:timestamp --episodes 10
```

## 📁 Tool Descriptions

### 1. `test_crafter_react_agent_openai.py`

**Purpose**: Generate Crafter traces using ReAct agents with OpenAI models.

**Usage**:
```bash
python test_crafter_react_agent_openai.py \
  --model gpt-4o-mini \
  --episodes 100 \
  --max-turns 30 \
  --difficulty easy \
  --config crafter_config.toml
```

**Configuration** (`crafter_config.toml`):
```toml
[eval]
model_name = "gpt-4o-mini"
episodes = 100
max_steps = 30
difficulty = "easy"
seed = 42

[service]
base_url = "http://localhost:8901"
timeout = 30.0

[output]
save_traces = true
save_detailed_results = true

[openai]
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"
```

**Output**:
- Traces saved to `traces_v2_synth/` directory
- Achievement statistics and analysis
- Detailed episode logs

### 2. `filter_traces_sft_duckdb.py`

**Purpose**: Filter and convert traces to SFT training format using DuckDB.

**Usage**:
```bash
python filter_traces_sft_duckdb.py \
  --input-dir traces_v2_synth \
  --output crafter_quality_sft.jsonl \
  --min-achievements 3 \
  --min-reward 10 \
  --max-turns 25 \
  --stats-only
```

**Filtering Options**:
- `--min-achievements`: Minimum achievements per episode
- `--min-reward`: Minimum reward score
- `--max-turns`: Maximum turns per episode
- `--require-termination`: Only include completed episodes
- `--stats-only`: Show statistics without creating output file

**Output Format** (JSONL):
```json
{"messages": [{"role": "system", "content": "You are playing Crafter..."}, {"role": "user", "content": "Observation: ..."}, {"role": "assistant", "content": "I'll move right and collect wood."}]}
```

### 3. `kick_off_ft_modal.py`

**Purpose**: Fine-tune models using Modal/Synth unified service.

**Usage**:
```bash
python kick_off_ft_modal.py crafter_quality_sft.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5
```

**Supported Models**:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

**Features**:
- Automatic token analysis
- Progress monitoring
- Model testing after completion
- Usage examples generation

### 4. `kick_off_ft_oai.py`

**Purpose**: Fine-tune models using OpenAI's API.

**Usage**:
```bash
python kick_off_ft_oai.py crafter_quality_sft.jsonl \
  --model gpt-4o-mini \
  --epochs 3 \
  --subset 1000
```

**Features**:
- Subset creation for different training sizes
- OpenAI API integration
- Job monitoring and status tracking

### 5. `compare_experiments.py`

**Purpose**: Compare multiple fine-tuning experiments and analyze results.

**Usage**:
```bash
python compare_experiments.py \
  --experiments experiments.json \
  --output comparison_report.html
```

**Experiment Configuration** (`experiments.json`):
```json
{
  "experiments": [
    {
      "name": "baseline-7b",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "training_data": "crafter_baseline.jsonl",
      "epochs": 3,
      "achievements": [2.1, 2.3, 2.0],
      "instances": ["easy", "medium", "hard"],
      "version": "v1.0"
    },
    {
      "name": "improved-7b",
      "model": "ft:qwen2-5-7b-instruct:org-test123:timestamp",
      "training_data": "crafter_improved.jsonl",
      "epochs": 5,
      "achievements": [3.2, 3.5, 3.1],
      "instances": ["easy", "medium", "hard"],
      "version": "v1.1"
    }
  ]
}
```

**Analysis Components**:

#### Achievements Analysis
- **Form**: List of achievement counts per episode
- **Metrics**: Average, median, standard deviation
- **Visualization**: Histograms, box plots

#### Instances Analysis
- **Form**: Difficulty levels (easy, medium, hard)
- **Metrics**: Success rates, completion times
- **Visualization**: Bar charts, heatmaps

#### Version Tracking
- **Form**: Incrementing version numbers (v1.0, v1.1, etc.)
- **Features**: Git integration, changelog tracking
- **Hooks**: Automated version bumping

#### Evaluation Hooks

##### `plot_hook_frequency.py`
```bash
python plot_hook_frequency.py \
  --traces traces_v2_synth/ \
  --output hook_analysis.png
```
- Analyzes tool call frequency
- Identifies common action patterns
- Generates action distribution plots

##### `seed_analysis_summary.py`
```bash
python seed_analysis_summary.py \
  --seeds 42,43,44,45 \
  --output seed_report.json
```
- Analyzes performance across different seeds
- Identifies seed-dependent behaviors
- Generates seed stability metrics

##### `analyze_enhanced_hooks.py`
```bash
python analyze_enhanced_hooks.py \
  --traces traces_v2_synth/ \
  --enhancements enhanced_hooks.json
```
- Analyzes custom hook implementations
- Measures hook effectiveness
- Generates enhancement recommendations

#### Custom Evaluation Pipelines

##### `custom_eval_pipelines.py`
```python
# Example custom evaluation
def evaluate_crafter_rules_understanding(traces):
    """Evaluate if model understands Crafter rules."""
    rule_violations = []
    for trace in traces:
        # Check for rule violations
        if has_invalid_action(trace):
            rule_violations.append(trace)
    return len(rule_violations) / len(traces)

def evaluate_time_efficiency(traces):
    """Evaluate if model uses time efficiently."""
    wasted_turns = []
    for trace in traces:
        # Count wasted actions
        wasted = count_wasted_actions(trace)
        wasted_turns.append(wasted)
    return sum(wasted_turns) / len(traces)
```

### 6. Efficient Comparison Tools

#### `efficient_compare.py`
```bash
python efficient_compare.py \
  --baseline baseline_results.json \
  --improved improved_results.json \
  --metrics achievements,time_efficiency,rule_understanding \
  --output comparison_summary.md
```

**Features**:
- Statistical significance testing
- Effect size calculations
- Confidence intervals
- Automated report generation

## 🔧 Advanced Configuration

### Environment Setup

```bash
# Set environment variables
export SYNTH_API_KEY="sk-test-11111111111111111111111111111111"
export MODAL_BASE_URL="https://synth-laboratories--unified-ft-service-fastapi-app.modal.run"

# Install dependencies
pip install modal httpx duckdb numpy pandas matplotlib seaborn
```

### Custom Evaluation Metrics

Create custom evaluation functions in `custom_eval_pipelines.py`:

```python
def custom_crafter_eval(traces):
    """Custom Crafter evaluation metrics."""
    return {
        "achievement_rate": calculate_achievement_rate(traces),
        "efficiency_score": calculate_efficiency(traces),
        "rule_understanding": evaluate_rule_understanding(traces),
        "time_management": evaluate_time_management(traces)
    }
```

### Batch Processing

```bash
# Process multiple experiments
for model in "7b" "14b" "32b"; do
    python kick_off_ft_modal.py crafter_quality_sft.jsonl \
      --model "Qwen/Qwen2.5-${model}-Instruct" \
      --epochs 3 \
      --suffix "batch-${model}"
done
```

## 📊 Analysis and Visualization

### Achievement Analysis
```bash
python analyze_achievements.py \
  --traces traces_v2_synth/ \
  --output achievement_analysis.html
```

### Action Pattern Analysis
```bash
python analyze_action_patterns.py \
  --traces traces_v2_synth/ \
  --output action_patterns.png
```

### Training Progress Analysis
```bash
python analyze_training_progress.py \
  --job-ids ftjob-abc123,ftjob-def456 \
  --output training_analysis.json
```

## 🚨 Troubleshooting

### Common Issues

1. **Modal Service Unavailable**
   ```bash
   # Check service health
   curl https://synth-laboratories--unified-ft-service-fastapi-app.modal.run/health
   ```

2. **Upload Failures**
   ```bash
   # Check file format
   python validate_jsonl.py crafter_quality_sft.jsonl
   ```

3. **Training Failures**
   ```bash
   # Check job status
   curl -H "Authorization: Bearer sk-test-11111111111111111111111111111111" \
     https://synth-laboratories--unified-ft-service-fastapi-app.modal.run/v1/fine_tuning/jobs/ftjob-abc123
   ```

### Debug Tools

```bash
# Validate training data
python validate_training_data.py crafter_quality_sft.jsonl

# Check model compatibility
python check_model_compatibility.py Qwen/Qwen2.5-7B-Instruct

# Analyze trace quality
python analyze_trace_quality.py traces_v2_synth/
```

## 📈 Performance Optimization

### Data Quality
- Filter for high-achievement episodes
- Remove episodes with rule violations
- Balance action distributions

### Training Efficiency
- Use appropriate model sizes
- Optimize hyperparameters
- Monitor training progress

### Evaluation Strategy
- Use multiple seeds for robustness
- Test on different difficulty levels
- Track performance over time

## 🔄 Workflow Automation

### Complete Pipeline Script
```bash
#!/bin/bash
# complete_crafter_ft_pipeline.sh

echo "🎮 Starting Crafter Fine-tuning Pipeline"

# 1. Generate traces
echo "📊 Generating traces..."
python test_crafter_react_agent_openai.py --episodes 1000

# 2. Filter traces
echo "🔍 Filtering traces..."
python filter_traces_sft_duckdb.py --min-achievements 3 --output crafter_quality_sft.jsonl

# 3. Fine-tune models
echo "🚀 Fine-tuning models..."
for model in "7b" "14b"; do
    python kick_off_ft_modal.py crafter_quality_sft.jsonl --model "Qwen/Qwen2.5-${model}-Instruct"
done

# 4. Evaluate results
echo "📈 Evaluating results..."
python compare_experiments.py --experiments experiments.json

echo "✅ Pipeline complete!"
```

## 📚 Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Synth API Documentation](https://synth-laboratories.com/docs)
- [Crafter Environment Guide](https://github.com/danijar/crafter)
- [Fine-tuning Best Practices](https://platform.openai.com/docs/guides/fine-tuning)

## 🤝 Contributing

To add new evaluation metrics or analysis tools:

1. Create new analysis function in `custom_eval_pipelines.py`
2. Add configuration options to experiment configs
3. Update comparison tools to include new metrics
4. Add visualization support if needed
5. Update this README with new features

## 📝 Changelog

- **v1.0**: Initial release with basic fine-tuning workflow
- **v1.1**: Added comprehensive evaluation tools
- **v1.2**: Enhanced comparison and analysis features
- **v1.3**: Added custom evaluation pipelines and hooks 