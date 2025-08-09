# Crafter Fine-Tuning Workflow

This guide walks you through the complete workflow: generating rollouts, filtering by achievements, fine-tuning, and comparing pre/post performance.

## Overview

The workflow consists of 4 main steps:
1. **Generate Initial Rollouts** - Create agent traces using base model
2. **Filter by Achievements** - Select high-quality traces for training
3. **Fine-tune Model** - Train on filtered data
4. **Compare Performance** - Evaluate pre/post fine-tuning

## Prerequisites

- Python 3.8+
- Required packages: `httpx`, `duckdb`, `pandas`, `matplotlib`
- Access to Crafter environment
- OpenAI API key (for fine-tuning)

## Step 1: Generate Initial Rollouts

First, generate agent traces using the base model:

```bash
python test_crafter_react_agent_openai.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --episodes 100 \
    --max-steps 50 \
    --difficulty "easy" \
    --output-dir "./traces/initial_rollouts"
```

**Key Parameters:**
- `--model`: Base model to use for initial rollouts
- `--episodes`: Number of episodes to run (more = better data)
- `--max-steps`: Maximum steps per episode
- `--difficulty`: Environment difficulty level
- `--output-dir`: Directory to save trace files

**Output:** Trace files in JSONL format with agent interactions, achievements, and rewards.

## Step 2: Filter Traces by Achievements

Filter the rollouts to keep only high-quality traces:

```bash
python filter_traces_sft_duckdb.py \
    --input "./traces/initial_rollouts/all_traces.json" \
    --output "./ft_data/filtered_traces.jsonl" \
    --min-achievements 3 \
    --min-reward 10.0 \
    --max-turns 30 \
    --stats-only
```

**Filtering Criteria:**
- `--min-achievements`: Minimum number of achievements unlocked
- `--min-reward`: Minimum total reward achieved
- `--max-turns`: Maximum number of turns (shorter = more efficient)
- `--stats-only`: Show statistics without filtering (for exploration)

**Quality Metrics:**
- **Achievement Rate**: % of episodes with 3+ achievements
- **Average Reward**: Mean reward across episodes
- **Efficiency**: Episodes with high reward in few turns

**Example Output:**
```
📊 Quality Statistics:
- Total traces: 1000
- High quality traces: 234 (23.4%)
- Average achievements: 2.1
- Average reward: 8.7
- Filtered traces: 156 (15.6%)
```

## Step 3: Fine-tune Model

Use the filtered data to fine-tune the model:

```bash
python kick_off_ft_oai.py \
    --training-file "./ft_data/filtered_traces.jsonl" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --suffix "crafter-achievements" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5
```

**Fine-tuning Parameters:**
- `--training-file`: Path to filtered training data
- `--model`: Base model to fine-tune
- `--suffix`: Unique identifier for the fine-tuned model
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for training

**Training Process:**
1. Uploads training data to OpenAI
2. Creates fine-tuning job
3. Monitors training progress
4. Returns fine-tuned model ID

**Example Output:**
```
🚀 Starting fine-tuning job...
✅ Job created: ftjob-abc123
📊 Status: running
⏱️  Estimated completion: 2 hours
✅ Training completed!
🎯 Fine-tuned model: ft:Qwen/Qwen2.5-7B-Instruct:org-123:timestamp
```

## Step 4: Generate Post-FT Rollouts

Run the same evaluation with the fine-tuned model:

```bash
python test_crafter_react_agent_openai.py \
    --model "ft:Qwen/Qwen2.5-7B-Instruct:org-123:timestamp" \
    --episodes 100 \
    --max-steps 50 \
    --difficulty "easy" \
    --output-dir "./traces/post_ft_rollouts"
```

## Step 5: Compare Pre/Post Performance

Compare the performance between base and fine-tuned models:

```bash
python compare_experiments.py \
    --experiment-1 "base_model" \
    --traces-1 "./traces/initial_rollouts/all_traces.json" \
    --experiment-2 "fine_tuned_model" \
    --traces-2 "./traces/post_ft_rollouts/all_traces.json" \
    --output-dir "./comparison_results"
```

**Comparison Metrics:**
- **Achievement Rate**: % of episodes with 3+ achievements
- **Average Reward**: Mean reward across episodes
- **Efficiency**: Reward per turn ratio
- **Action Patterns**: Most common actions taken
- **Success Rate**: Episodes reaching specific goals

**Example Output:**
```
📊 Performance Comparison:

Base Model:
- Achievement Rate: 23.4%
- Average Reward: 8.7
- Efficiency: 0.29 reward/turn

Fine-tuned Model:
- Achievement Rate: 41.2% (+17.8%)
- Average Reward: 12.3 (+3.6)
- Efficiency: 0.41 reward/turn (+0.12)

🎯 Improvement: +75% achievement rate, +41% average reward
```

## Advanced Configuration

### Custom Achievement Filtering

Filter by specific achievement types:

```bash
python filter_traces_sft_duckdb.py \
    --input "./traces/initial_rollouts/all_traces.json" \
    --output "./ft_data/advanced_filtered.jsonl" \
    --min-achievements 3 \
    --achievement-types "craft_wood_pickaxe,craft_wood_sword" \
    --min-achievement-types 2
```

### Batch Processing

Process multiple experiments:

```bash
# Generate rollouts for multiple models
for model in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"; do
    python test_crafter_react_agent_openai.py \
        --model "$model" \
        --episodes 50 \
        --output-dir "./traces/${model//\//_}"
done

# Compare all models
python compare_experiments.py \
    --experiment-1 "7B_base" \
    --traces-1 "./traces/Qwen_Qwen2.5-7B-Instruct/all_traces.json" \
    --experiment-2 "14B_base" \
    --traces-2 "./traces/Qwen_Qwen2.5-14B-Instruct/all_traces.json" \
    --output-dir "./multi_model_comparison"
```

### Custom Evaluation Metrics

Create custom evaluation pipelines:

```python
# custom_eval_pipelines.py
def custom_achievement_analysis(traces):
    """Analyze specific achievement patterns."""
    achievement_counts = {}
    for trace in traces:
        for achievement in trace.get('final_achievements', {}):
            achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
    return achievement_counts

# Use in comparison
python compare_experiments.py \
    --custom-eval "custom_achievement_analysis" \
    --experiment-1 "base" \
    --traces-1 "./traces/base/all_traces.json" \
    --experiment-2 "ft" \
    --traces-2 "./traces/ft/all_traces.json"
```

## Troubleshooting

### Common Issues

**Low Quality Traces:**
- Increase `--episodes` for more data
- Lower filtering criteria (`--min-achievements`, `--min-reward`)
- Check environment difficulty settings

**Fine-tuning Failures:**
- Verify training data format (JSONL with messages)
- Check OpenAI API quota and limits
- Ensure base model is supported

**Comparison Errors:**
- Verify trace file paths exist
- Check trace file format compatibility
- Ensure both experiments have sufficient data

### Debug Commands

```bash
# Check trace file format
python filter_traces_sft_duckdb.py --input traces.json --stats-only

# Validate training data
python -c "
import json
with open('training_data.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if 'messages' not in data:
            print(f'Error at line {i}: missing messages')
"

# Test model loading
python test_crafter_react_agent_openai.py --model "your-model" --episodes 1
```

## Performance Optimization

### Data Quality
- **More Episodes**: 100+ episodes for reliable statistics
- **Diverse Seeds**: Use different random seeds for variety
- **Quality Filtering**: Focus on high-achievement traces

### Training Efficiency
- **Batch Size**: Start with 4, increase if memory allows
- **Learning Rate**: 2e-5 works well for most cases
- **Epochs**: 2-3 epochs usually sufficient

### Evaluation Strategy
- **Consistent Settings**: Same difficulty, max_steps across experiments
- **Statistical Significance**: 50+ episodes per experiment
- **Multiple Metrics**: Look at both achievement rate and efficiency

## File Structure

```
crafter_openai_ft/
├── test_crafter_react_agent_openai.py    # Generate rollouts
├── filter_traces_sft_duckdb.py           # Filter by quality
├── kick_off_ft_oai.py                    # Fine-tune model
├── compare_experiments.py                 # Compare performance
├── traces/                                # Rollout data
│   ├── initial_rollouts/
│   └── post_ft_rollouts/
├── ft_data/                              # Training data
└── old/                                  # Archived scripts
```

## Next Steps

1. **Scale Up**: Increase episodes and model sizes
2. **Custom Metrics**: Add domain-specific evaluation
3. **Hyperparameter Tuning**: Optimize training parameters
4. **Multi-Task Learning**: Train on multiple environments
5. **Deployment**: Integrate fine-tuned models into production

This workflow provides a complete pipeline from data generation to model improvement, enabling systematic evaluation of fine-tuning effectiveness.
