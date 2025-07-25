# Gemini Fine-tuning Guide for Crafter

This guide explains how to generate fine-tuning data using Gemini models and train custom models on Vertex AI.

## Overview

The Gemini fine-tuning pipeline consists of:
1. **Data Generation**: Use Gemini models to play Crafter and generate high-quality trajectories
2. **Filtering**: Select trajectories with good achievement scores
3. **Formatting**: Convert to Vertex AI-compatible JSONL format
4. **Fine-tuning**: Train custom Gemini models on Vertex AI

## Prerequisites

1. **Synth-AI Environment Service** running on port 8901:
   ```bash
   cd ../../../..
   python -m synth_ai.environments.service.app
   ```

2. **Google Cloud Setup**:
   - Active GCP project with billing enabled
   - Vertex AI API enabled
   - Storage bucket for training data
   - Sufficient quota for Gemini fine-tuning

3. **API Keys**:
   - Set `GOOGLE_API_KEY` environment variable for Gemini API access

## Quick Start

### 1. Test the Setup

```bash
# Run tests to verify everything works
python test_gemini_generation.py
```

### 2. Generate Training Data

```bash
# Generate 100 trajectories (default)
python generate_ft_data_gemini.py --config gemini_ft_config.toml

# Generate with custom settings
python generate_ft_data_gemini.py --num-rollouts 500 --model gemini-2.5-flash

# Filter existing trajectories
python generate_ft_data_gemini.py --filter-only --min-achievements 4
```

### 3. Validate and Prepare Data

```bash
# Validate JSONL format
python prepare_vertex_ft.py ft_data_gemini/crafter_gemini_ft.jsonl --validate

# Create a subset for testing
python prepare_vertex_ft.py ft_data_gemini/crafter_gemini_ft.jsonl --create-subset 1000

# Add system prompts
python prepare_vertex_ft.py ft_data_gemini/crafter_gemini_ft.jsonl --convert --add-system
```

### 4. Start Fine-tuning on Vertex AI

```bash
# Upload to GCS and start fine-tuning
python kick_off_ft_gemini.py ft_data_gemini/crafter_gemini_ft.jsonl \
  --project YOUR_PROJECT_ID \
  --bucket YOUR_GCS_BUCKET \
  --model gemini-1.0-flash \
  --display-name crafter-gemini-expert \
  --epochs 3
```

## Configuration

### `gemini_ft_config.toml`

```toml
[generation]
model_name = "gemini-2.5-flash"  # Model for data generation
num_rollouts = 100               # Number of episodes
max_turns = 30                   # Max steps per episode
difficulty = "easy"              # Crafter difficulty

[quality]
min_score_threshold = 2.0        # Minimum trajectory score
min_achievements = 3             # Minimum achievements
enable_thinking = true           # Enable chain-of-thought
thinking_budget = 15000          # Thinking token budget
```

## Data Generation Process

### 1. Trajectory Generation
- Uses `GeminiCrafterAgent` with synth-ai's LM class
- Enables thinking/reasoning for better decision-making
- Tracks all LLM calls, actions, and observations

### 2. Quality Filtering
- **Score-based**: Trajectories scored by achievements and efficiency
- **Achievement-based**: Minimum number of game objectives completed
- **Action quality**: Penalizes invalid or repetitive actions

### 3. JSONL Conversion
- Extracts user-assistant message pairs from trajectories
- Formats for Vertex AI compatibility
- Preserves thinking/reasoning when available

## Output Format

The generated JSONL follows Vertex AI format:
```json
{
  "messages": [
    {"role": "user", "content": "=== Current State ===\nPosition: (5, 10)\nHealth: 9/10..."},
    {"role": "assistant", "content": "I need to collect wood to craft a pickaxe..."}
  ]
}
```

## Cost Estimation

Training costs depend on:
- Number of training examples
- Total tokens (input + output)
- Model size (gemini-1.0-flash)

Use `prepare_vertex_ft.py --validate` to get cost estimates before training.

## Best Practices

1. **Start Small**: Test with 100-500 trajectories first
2. **Quality over Quantity**: Better to have fewer high-quality examples
3. **Monitor Costs**: Check token counts and estimated costs
4. **Iterative Improvement**: Analyze results and adjust filters
5. **Use Subsets**: Test with smaller datasets before full training

## Troubleshooting

### "Service not running"
Start the synth-ai environment service:
```bash
cd ../../../..
python -m synth_ai.environments.service.app
```

### "API key not set"
Set your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key"
```

### "Invalid JSONL format"
Validate and fix with:
```bash
python prepare_vertex_ft.py your_file.jsonl --validate
python prepare_vertex_ft.py your_file.jsonl --convert
```

## Advanced Usage

### Custom Filtering
Modify `filter_high_quality_trajectories()` in `generate_ft_data_gemini.py` to implement custom quality metrics.

### Different Models
Test with various Gemini models:
- `gemini-2.5-flash`: Fast, good reasoning
- `gemini-2.5-pro`: Best quality, slower
- `gemini-2-flash`: Latest, experimental
- `gemini-1.5-flash`: Stable, widely available

### Batch Processing
For large-scale generation:
```python
# In generate_ft_data_gemini.py
config.num_rollouts = 1000  # Generate many trajectories
# Run with asyncio concurrency limits
```

## Next Steps

After fine-tuning completes:
1. Test the fine-tuned model
2. Compare performance with base model
3. Iterate on data generation strategy
4. Deploy for production use