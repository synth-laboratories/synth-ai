# Image Style Matching Demo

This demo uses GraphGen to optimize a workflow for generating Pokemon-style images using `gemini-2.5-flash-image`.

## Quick Start

### Option 1: Run via Script

```bash
cd /path/to/synth-ai
export SYNTH_API_KEY=your_api_key_here  # Optional - will auto-generate demo key if not set
export SYNTH_BACKEND_URL=http://127.0.0.1:8000  # Optional - override backend base URL

uv run python demos/image_style_matching/run_notebook.py
```

This executes `graphgen_image_style_matching.ipynb` using papermill and saves:
- Executed notebook to `demo_prod_executed.ipynb`
- Generated images to `results/`

### Option 2: Run Interactively in Jupyter

```bash
cd /path/to/synth-ai
jupyter notebook demos/image_style_matching/graphgen_image_style_matching.ipynb
```

## What the Demo Does

1. Creates a dataset with Pokemon-style image generation tasks
2. Runs GraphGen optimization to find the best prompt workflow (~2-5 minutes)
3. Downloads the optimized graph
4. Runs inference on test inputs (wolf, fox, rabbit)
5. Saves generated images to `results/`

## Results

After running, `results/` will contain:

```
results/
├── optimized_graph.txt    # The optimized prompt workflow
├── test_1_wolf.png        # Generated wolf image (~1.6 MB)
├── test_2_fox.png         # Generated fox image (~1.6 MB)
└── test_3_rabbit.png      # Generated rabbit image (~1.5 MB)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Your Synth API key (optional - will auto-generate demo key) |
| `SYNTH_BACKEND_URL` | Override backend base URL (e.g. `http://127.0.0.1:8000`) |
| `BACKEND_BASE_URL` | Custom backend URL (default: `https://api.usesynth.ai`) |

## Model Configuration

- **Policy Model**: `gemini-2.5-flash-image` (image generation)
- **Judge Model**: `gpt-4.1-nano` (contrastive evaluation)
