# MTG Artist Style Matching Demo

This demo optimizes prompts to generate images matching the distinctive style of famous **Magic: The Gathering** artists — **without naming the artist**.

## The Challenge

Given reference images from an MTG artist's work, find a prompt that:
1. Generates images matching their distinctive visual style
2. **Does NOT mention the artist's name** (verifier returns 0 if it does)

This forces the optimizer to discover descriptive style characteristics (lighting, color palette, brushwork, mood) rather than using the artist's name as a shortcut.

## Featured Artists (18 total)

| Key | Artist | Style |
|-----|--------|-------|
| `seb_mckinnon` | **Seb McKinnon** | Moody, gothic, painterly surrealism |
| `rebecca_guay` | **Rebecca Guay** | Delicate watercolor/ink, pre-Raphaelite aesthetics |
| `terese_nielsen` | **Terese Nielsen** | Saturated, iconic, high-contrast fantasy lighting |
| `john_avon` | **John Avon** | Luminous landscapes, atmospheric depth |
| `nils_hamm` | **Nils Hamm** | Eerie atmospheric realism, unsettling undertones |
| `rk_post` | **rk post** | Dark fantasy, sharp silhouettes, metal-album energy |
| `kev_walker` | **Kev Walker** | Gritty dynamic figures, classic modern Magic look |
| `ron_spencer` | **Ron Spencer** | Grotesque horror creatures, organic textures |
| `mark_tedin` | **Mark Tedin** | Old-school surreal alien fantasy, cosmic weirdness |
| `quinton_hoover` | **Quinton Hoover** | Loose expressive brushwork, dreamy character pieces |
| `zoltan_boros` | **Zoltan Boros** | Dramatic realism, cinematic scenes |
| `steve_argyle` | **Steve Argyle** | Polished character portraits, crisp fantasy |
| `wayne_reynolds` | **Wayne Reynolds** | Dense linework, kinetic busy compositions |
| `aleksi_briclot` | **Aleksi Briclot** | Sleek cinematic concept-art style |
| `magali_villeneuve` | **Magali Villeneuve** | Ultra-detailed elegant high-fantasy realism |
| `drew_tucker` | **Drew Tucker** | Abstract experimental texture-heavy fantasy |
| `dan_frazier` | **Dan Frazier** | Clean iconic artifact look, graphic design sensibility |
| `chippy` | **Chippy** | Stylized bold shapes, cartoon satirical edge |

## Quick Start

### Step 1: Fetch Artist Card Art

```bash
cd /path/to/synth-ai
uv run python demos/mtg_artist_style/fetch_artist_cards.py
```

Downloads card art from Scryfall for all 6 artists (~8 cards each).

### Step 2: Run Full Demo

```bash
# Run for Seb McKinnon (default)
uv run python demos/mtg_artist_style/run_demo.py

# Run for a specific artist
uv run python demos/mtg_artist_style/run_demo.py --artist rebecca_guay

# Use local backend
uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon --local
```

This runs:
1. **Verifier Optimization** (Graph Evolve) - creates a verifier that penalizes artist name mentions
2. **Prompt Optimization** (GEPA) - optimizes prompts to match style without naming the artist

### Step 3: View Results

```
artifacts/seb_mckinnon/
├── verifier_opt.json    # Optimized verifier graph
└── prompt_opt.json      # Optimized prompt (should NOT contain artist name!)
```

## How It Works

### Two-Stage Optimization

1. **Verifier Optimization (Graph Evolve)**
   - Trains a verifier that:
     - Compares generated images to reference art using contrastive VLM
     - **Returns score 0** if prompt contains artist name or variations
   - Creates calibration examples with good/bad prompts

2. **Prompt Optimization (GEPA)**
   - Uses the optimized verifier to score candidate prompts
   - Evolves prompts that capture the style through description
   - Cannot "cheat" by just naming the artist

### Artist Name Detection

The verifier checks for forbidden patterns including:
- Full name: "Seb McKinnon"
- Last name only: "McKinnon"
- Variations: "seb_mckinnon", "seb-mckinnon", "sebmckinnon"

```python
# ✅ VALID - describes style without naming artist
"Generate a moody gothic fantasy image with ethereal atmosphere and painterly surrealism"

# ❌ INVALID - mentions artist name (score = 0)
"Generate an image in the style of Seb McKinnon"
```

## Running Individual Steps

### Verifier Optimization Only

```bash
uv run python demos/mtg_artist_style/run_verifier_opt.py --artist seb_mckinnon
uv run python demos/mtg_artist_style/run_verifier_opt.py --artist seb_mckinnon --local
```

### Prompt Optimization Only (requires verifier artifact)

```bash
uv run python demos/mtg_artist_style/run_prompt_opt.py --artist seb_mckinnon
uv run python demos/mtg_artist_style/run_prompt_opt.py --artist seb_mckinnon --local
```

### Skip Steps

```bash
# Skip verifier opt (use existing artifact)
uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon --skip-verifier-opt

# Only run verifier opt
uv run python demos/mtg_artist_style/run_demo.py --artist seb_mckinnon --skip-prompt-opt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Your Synth API key (optional - will mint demo key) |
| `LOCAL_BACKEND` | Alternative to `--local` flag |
| `BACKEND_BASE_URL` | Custom backend URL (default: `https://api.usesynth.ai`) |

## Model Configuration

- **Verifier Model**: `gpt-4.1-nano` (contrastive VLM evaluation)
- **Policy Model**: `gemini-2.5-flash-image` (image generation)
- **Proposer Model**: `gpt-4.1` (Graph Evolve prompt proposals)

## Adding More Artists

Edit `ARTISTS` dict in `fetch_artist_cards.py`:

```python
ARTISTS = {
    "your_artist_key": {
        "name": "Artist Full Name",
        "query": 'a:"Artist Full Name"',  # Scryfall query
        "style_description": "Description of their distinctive style",
        "cards_to_fetch": 8,
    },
}
```

Then run `fetch_artist_cards.py` again.

## Dataset

The dataset (`card_descriptions.json`) contains 360 cards with:
- **card_name** - Card title  
- **artist** / **artist_key** - Artist attribution
- **type_line** - Card type (Creature, Land, etc.)
- **oracle_text** - Card game rules text
- **art_description** - Style-neutral description of visual content
- **image_path** - Path to reference image

Generate descriptions with:
```bash
uv run python demos/mtg_artist_style/generate_descriptions.py
```

## File Structure

```
demos/mtg_artist_style/
├── README.md
├── fetch_artist_cards.py      # Download card art from Scryfall
├── generate_descriptions.py   # Generate style-neutral art descriptions
├── run_demo.py                # Main entry point (runs both stages)
├── run_verifier_opt.py        # Stage 1: Graph Evolve verifier optimization
├── run_prompt_opt.py          # Stage 2: GEPA prompt optimization
├── artist_metadata.json       # Generated: artist/card metadata
├── card_descriptions.json     # Generated: art descriptions dataset
├── gold_images/               # Downloaded card art (360 images)
│   ├── seb_mckinnon/
│   ├── rebecca_guay/
│   └── ...
└── artifacts/                 # Optimization results
    ├── seb_mckinnon/
    │   ├── verifier_opt.json
    │   └── prompt_opt.json
    └── ...
```

## Scryfall API

Uses the [Scryfall API](https://scryfall.com/docs/api) to fetch card images.
Rate limit: 10 requests/second (we use 150ms delay).

Query examples:
- `a:"John Avon"` - cards by John Avon
- `a:"Rebecca Guay" has:image unique:art` - unique art with images
