# ADAS Cookbooks

Example datasets and test scripts for ADAS (Automated Design of Agentic Systems) workflow optimization.

## What is ADAS?

ADAS is a simplified "Workflows API" for prompt optimization that:
- Uses JSON datasets instead of TOML configs
- Auto-generates task apps from datasets (no user-managed task apps)
- Has built-in judge configurations (rubric, contrastive, gold_examples)
- Wraps GEPA internally for the actual optimization

## Examples

### 1. Style Matching (Contrastive Judge)

Demonstrates optimizing a prompt to match a specific writing style using gold examples.

**Dataset:** `style_matching_dataset.json`
- 5 tasks (essay outlines to expand)
- 5 gold outputs (reference essays showing desired style)
- Contrastive judge mode (compares generated vs gold)

**Run:**
```bash
# CLI
uvx synth-ai train --type adas --dataset products/adas/cookbooks/style_matching_dataset.json --poll

# Python
uv run python products/adas/cookbooks/test_style_matching.py
```

### 2. Banking77 Intent Classification (Rubric Judge)

Demonstrates optimizing a prompt for intent classification using explicit rubric criteria.

**Dataset:** `banking77_dataset.json`
- 10 tasks (customer queries to classify)
- 10 gold outputs (correct intent labels)
- Rubric judge mode (scores against criteria)

**Run:**
```bash
# CLI
uvx synth-ai train --type adas --dataset products/adas/cookbooks/banking77_dataset.json --poll

# Python
uv run python products/adas/cookbooks/test_banking77.py
```

### 3. Nano Banana Image Style Matching (Contrastive Judge)

Demonstrates optimizing image generation prompts using Nano Banana (gemini-2.5-flash-image) with a contrastive VLM judge to match Pokemon art style.

This example is **planned but not yet shipped** as a runnable script in this repo.
When it lands, it will follow the same pattern as the other cookbooks: authenticated
ADAS endpoints only, no local-only hacks.

**Gold Set:** `lambdalabs/pokemon-blip-captions` (HuggingFace)
- 5 tasks (Pokemon-style creature generation)
- 5 gold outputs (real Pokemon images from HuggingFace dataset)
- Contrastive judge mode with VLM (gpt-4o) comparing generated images to gold examples
- Policy model: `gemini-2.5-flash-image` (Nano Banana)

**Run (future):**
```bash
# Python (downloads gold images from HuggingFace, generates dataset, runs optimization)
uv run python products/adas/test_image_style_matching.py
```

**Requirements:**
- `GEMINI_API_KEY` - Required for Nano Banana image generation
- `SYNTH_API_KEY` - Required for ADAS job submission
- `datasets` and `pillow` packages: `pip install datasets pillow`
- VLM-capable judge model (gpt-4o, gemini-2.5-flash, etc.) for image comparison

**How it works:**
1. Script loads 5 Pokemon images from `lambdalabs/pokemon-blip-captions` on HuggingFace
2. Images are encoded as base64 data URLs and saved as gold outputs
3. ADAS optimizes the prompt to generate images matching Pokemon art style
4. Contrastive VLM judge compares generated images to gold Pokemon references

## Dataset Format

ADAS uses `ADASTaskSet` JSON format:

```json
{
  "version": "1.0",
  "metadata": {
    "name": "my-dataset",
    "description": "..."
  },
  "initial_prompt": "Your initial system prompt...",
  "tasks": [
    {
      "id": "task1",
      "input": {"key": "value"}
    }
  ],
  "gold_outputs": [
    {
      "task_id": "task1",
      "output": {"key": "value"}
    }
  ],
  "default_rubric": {
    "outcome": {
      "criteria": [
        {
          "name": "accuracy",
          "description": "...",
          "weight": 1.0
        }
      ]
    }
  },
  "judge_config": {
    "mode": "rubric",
    "model": "llama-3.3-70b-versatile",
    "provider": "groq"
  }
}
```

## Judge Modes

- **rubric**: Scores against explicit criteria (good for classification, structured tasks)
- **contrastive**: Compares generated output to gold examples (good for style matching, creative tasks)
- **gold_examples**: Uses gold outputs as few-shot context for judge (hybrid approach)

## Requirements

- `SYNTH_API_KEY` - Your Synth API key
- `BACKEND_BASE_URL` (optional) - Backend URL override for dev/local; defaults to production
