# Web Design Style Prompt Optimization with GEPA

This demo uses **GEPA** (Genetic Evolution of Prompt Architecture) to optimize a style system prompt that guides Gemini 2.5 Flash Image to generate visually accurate webpage screenshots.

## The Problem

Functional descriptions alone produce poor visual fidelity when generating webpage images:

- **Baseline Score: 3.2/10** (tested on 5 Astral pages)
- **Root Cause**: Functional descriptions describe *what* content is on the page, but not *how* it should look visually
- **Missing Information**: Color palettes, typography, layout spacing, brand identity

### Failure Patterns Identified

From `VERIFICATION_ANALYSIS.md`:

| Metric | Baseline Score | Key Issues |
|--------|---------------|------------|
| Color Scheme & Branding | 2.4/10 | Dark theme bias, wrong accent colors |
| Typography & Styling | 3.8/10 | Wrong font sizes and weights |
| Layout & Spacing | 4.0/10 | Incorrect padding and margins |
| Visual Elements | 2.8/10 | Missing logos, wrong icons |
| Overall Fidelity | 3.0/10 | Unrecognizable as original site |

**Critical Finding**: 100% of generated pages used dark themes despite originals being light-themed.

## The Solution: GEPA Optimization

GEPA optimizes a style system prompt that provides the missing visual guidance:

```
Baseline Prompt (generic):
"Create a modern, professional webpage design."
↓
GEPA Optimization (3 generations, 50 evaluations)
↓
Optimized Prompt (brand-specific):
"Use light backgrounds (#F8F8F8) with dark text.
Green CTAs (#51CF66). Purple sections (#5D1F7B).
Large bold headings. Generous spacing..."
```

### Expected Improvements

| Approach | Expected Score | Fixes |
|----------|---------------|-------|
| Baseline (functional only) | 3.2/10 | - |
| + Color palette | 5.0/10 | Color scheme, some visual elements |
| + Style keywords | 6.5/10 | Typography, spacing refinement |
| + GEPA optimization | **7.5-8.5/10** | Brand consistency, visual polish |

## Files

```
demos/web-design/
├── README.md                    # This file
├── run_demo.py                  # Main demo script
├── gepa_config.toml             # GEPA optimization configuration
├── verify_generation.py         # Original verification script
├── run_batch_verification.py   # Batch verification script
├── VERIFICATION_ANALYSIS.md    # Analysis of baseline results
├── create_hf_dataset.py         # Build + (optionally) push the dataset to Hugging Face Hub
├── verification_results/        # Baseline verification outputs
└── optimization_results/        # GEPA optimization outputs
```

## Using a public Hugging Face dataset (recommended)

The demo is designed to **download screenshots + descriptions from a public Hugging Face dataset**
instead of relying on git-committed images.

By default, `run_demo.py` uses:

- `JoshPurtell/web-design-screenshots`

To override, set:

- `SYNTH_WEB_DESIGN_DATASET=org/web-design-screenshots` (your public dataset repo id)
- (optional) `SYNTH_WEB_DESIGN_DATASET_REVISION=<git sha or tag>` for reproducibility
- (optional) `SYNTH_WEB_DESIGN_MAX_EXAMPLES=8` to cap the number of images used (defaults to 8)
- (optional) `SYNTH_WEB_DESIGN_MAX_IMAGE_PIXELS=12000000` to skip oversized screenshots (defaults to 12MP)
- (optional) `SYNTH_WEB_DESIGN_CACHE_DIR=...` to control where downsampled images are cached
- (optional) For dataset publishing: slice tall pages into multiple segments so 384px-downsampled images stay readable.

To force **local disk mode** (no Hub download), set:

- `SYNTH_WEB_DESIGN_DATASET=local`

## Dataset

**Source**: 20 startup websites, 620 total screenshots
**Focus**: Astral (astral.sh) - 37 pages with functional descriptions

Why Astral?
- ✓ Best coverage (37 pages)
- ✓ Consistent visual brand (light theme, green/purple accents)
- ✓ Modern tech startup aesthetic
- ✓ Clear brand identity to learn

## How It Works

### 1. Task App (`task_app.py`)

A Local API server that:

```python
Input:
  - functional_description (str): "The page has a header with..."
  - original_image_path (str): "/path/to/astral_homepage.png"
  - style_prompt (str): "Use light backgrounds with..." ← OPTIMIZED BY GEPA

Process:
  1. Combine functional_description + style_prompt
  2. Generate image with Gemini 2.5 Flash Image
  3. Compare to original using vision model

Output:
  - color_scheme_branding (0-10)
  - typography_styling (0-10)
  - layout_spacing (0-10)
  - visual_elements (0-10)
  - overall_visual_fidelity (0-10)
  - average_score (0-10) ← PRIMARY METRIC
```

### 2. GEPA Config (`gepa_config.toml`)

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8103"

[prompt_learning.gepa.population]
initial_size = 6                # Start with 6 random variants
num_generations = 3             # Evolve for 3 generations
children_per_generation = 4     # Create 4 children per generation

[prompt_learning.gepa.rollout]
budget = 50                     # 50 total evaluations
minibatch_size = 5              # Evaluate on 5 examples per candidate

[prompt_learning.gepa.evaluation]
seeds = [0..14]                 # 15 training examples
validation_seeds = [15..22]     # 8 validation examples
```

### 3. Demo Script (`run_demo.py`)

```bash
python3 run_demo.py                    # Run full optimization
python3 run_demo.py --max-examples 10  # Smaller dataset (faster)
python3 run_demo.py --local            # Use local backend
```

**Workflow**:
1. Load dataset (from Hub by default via `SYNTH_WEB_DESIGN_DATASET`)
2. Materialize a small subset of images into a local cache (outside the repo) for task app access
3. Start task app server on port 8103
4. Submit GEPA job to Synth backend
5. Poll for completion (10-30 minutes)
6. Display optimized style prompt
7. Save results to `optimization_results/`

## Publishing the dataset to the Hub

From a machine that has the screenshots available locally:

```bash
# HF_TOKEN must have write access to the org dataset repo
export HF_TOKEN=...

# Example: build the dataset from a JSON of descriptions that reference local screenshot paths
uv run python demos/web-design/create_hf_dataset.py \
  --descriptions-file demos/web-design/all_functional_descriptions.json \
  --push --repo-name org/web-design-screenshots
```

## Requirements

### API Keys

```bash
# .env file
SYNTH_API_KEY=sk_live_...
GEMINI_API_KEY=AIzaSy...
```

### Python Dependencies

```bash
pip install google-genai datasets pillow httpx
```

Or use the project's existing environment (uv, poetry, etc.)

## Running the Demo

### Quick Start

```bash
cd /path/to/synth-ai

# Ensure synth-ai is importable (pick one):
uv sync
# OR:
python -m pip install -e .

uv run python demos/web-design/run_demo.py
```

### Expected Output

```
================================================================================
GEPA DEMO: WEB DESIGN STYLE PROMPT OPTIMIZATION
================================================================================
Backend: https://api.usesynth.ai
Task App: http://127.0.0.1:8103
================================================================================

Using SYNTH_API_KEY: sk_live_ace8b968a529...

================================================================================
PREPARING ASTRAL DATASET
================================================================================
Found 37 valid Astral pages
Using 23 examples for optimization

Saving images to task_images/...
✓ Saved 23 images
✓ Dataset info saved to optimization_results/dataset_info.json

================================================================================
STARTING TASK APP SERVER
================================================================================
Starting task app on http://127.0.0.1:8103
This may take a few seconds...

✓ Task app is ready at http://127.0.0.1:8103

================================================================================
RUNNING GEPA OPTIMIZATION
================================================================================
Using config: gepa_config.toml

Submitting GEPA job...
✓ Job submitted: pl_a1b2c3d4e5f6

Polling for completion (this may take 10-30 minutes)...

[Generation 1/3] Evaluating 6 initial candidates...
[Generation 1/3] Best score: 4.2/10
[Generation 2/3] Evaluating 4 children...
[Generation 2/3] Best score: 5.8/10
[Generation 3/3] Evaluating 4 children...
[Generation 3/3] Best score: 7.3/10

================================================================================
OPTIMIZATION RESULTS
================================================================================

Status: SUCCEEDED
Best Score: 7.3/10

================================================================================
OPTIMIZED STYLE PROMPT
================================================================================
You are generating a professional startup website screenshot for Astral.

CRITICAL VISUAL REQUIREMENTS:
- Background: Use off-white (#F8F8F8) or light grey (#F5F5F5) backgrounds
- Text: Dark grey (#242220) or black text for maximum contrast
- Accent Colors:
  * Bright green (#51CF66, #47CE77) for call-to-action buttons
  * Deep purple (#5D1F7B, #1A1A2E) for promotional sections
  * Light purple (#F0ECF6) for subtle background variations

TYPOGRAPHY:
- Headings: Large, bold, sans-serif fonts (2-3x body text size)
- Body: Clean, readable sans-serif with generous line-height
- Hierarchy: Clear size differences between h1, h2, h3, body

LAYOUT:
- Generous white space and padding between sections
- Wide margins (not edge-to-edge content)
- Clear visual separation between page sections
- Professional, uncluttered appearance

BRAND IDENTITY:
- Modern, minimal, tech-forward aesthetic
- Light, airy feel with high contrast
- Trustworthy and professional tone

Apply these guidelines precisely to match Astral's visual brand.

✓ Saved to: optimization_results/optimized_style_prompt.txt
✓ Full results: optimization_results/gepa_results.json

================================================================================
CLEANUP
================================================================================
Stopping task app server...
✓ Task app stopped

================================================================================
DEMO COMPLETE
================================================================================
Total time: 1247.3s (20.8 min)
Dataset: 23 Astral pages
Result: succeeded
Best score: 7.3/10
```

## Baseline vs Optimized Comparison

### Baseline (Functional Description Only)

**Average Score: 3.2/10**

Generated image characteristics:
- Dark theme (#1A2B3C backgrounds)
- Blue accent colors (#4596F1)
- Generic typography
- Wrong brand identity

**Example Issues**:
- "Background: Original #F8F8F8, Generated #1A2B3C"
- "CTA button: Original green (#51CF66), Generated blue (#4596F1)"
- "Logo missing or wrong colors"

### After GEPA Optimization

**Average Score: 7.3/10** (target)

Generated image characteristics:
- Light theme (#F8F8F8 backgrounds) ✓
- Green accent colors (#51CF66) ✓
- Proper typography hierarchy ✓
- Matches brand identity ✓

**Improvements**:
- Color scheme accuracy: 2.4 → 7.5/10
- Typography matching: 3.8 → 7.8/10
- Visual fidelity: 3.0 → 7.2/10

## Next Steps

### 1. Test Optimized Prompt

```python
from verify_generation import generate_image_from_description

# Load optimized prompt
with open('optimization_results/optimized_style_prompt.txt') as f:
    style_prompt = f.read()

# Generate with optimized prompt
image_bytes = generate_image_from_description_with_style(
    description=example['functional_description'],
    style_prompt=style_prompt,
    api_key=GEMINI_API_KEY
)
```

### 2. Run Batch Verification

```bash
# Test optimized prompt on more examples
python3 run_batch_verification_with_optimized_prompt.py
```

### 3. Extend to Other Sites

- Try optimization on other sites (Linear, Stripe, etc.)
- Compare site-specific vs generic optimized prompts
- Multi-site optimization (optimize across all 20 sites)

### 4. Advanced Experiments

- **Prompt length vs quality**: Does longer prompt = better score?
- **Cross-site transfer**: Does Astral prompt work for Linear?
- **Multi-objective optimization**: Optimize for speed + quality
- **Few-shot learning**: Include example screenshots in prompt

## Related Files

- `VERIFICATION_ANALYSIS.md`: Detailed analysis of baseline failures
- `verification_results/`: Baseline verification outputs (5 examples)
- `run_batch_verification.py`: Batch verification script
- `verify_generation.py`: Single-example verification

## Troubleshooting

### Task app fails to start

```bash
# Check port 8103 is available
lsof -i :8103

# Kill existing process
kill -9 <PID>
```

### GEPA job fails

```bash
# Check backend connectivity
curl https://api.usesynth.ai/health

# Verify API key
echo $SYNTH_API_KEY

# Check task app is accessible from backend
# (May need tunnel if using production backend)
```

### Low scores even after optimization

- Check `optimization_results/gepa_results.json` for details
- Increase `num_generations` in `gepa_config.toml`
- Increase `rollout_budget` for more evaluations
- Try different `mutation_rate` and `crossover_rate` values

## References

- GEPA Paper: [Coming soon]
- Gemini 2.5 Flash Image: https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.5-flash-image
- Synth Prompt Learning: https://docs.usesynth.ai/prompt-learning
