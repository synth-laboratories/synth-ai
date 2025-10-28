# Qwen VL Examples for Crafter

**📖 MASTER GUIDE**: See `../../VLM_COMPLETE_GUIDE.md` for full documentation

Vision-language model examples for Crafter agents with image observations.

**Status**: ✅ Production Ready (October 27, 2025)

## Documentation

| Document | Purpose |
|----------|---------|
| `../../VLM_COMPLETE_GUIDE.md` | Complete VLM documentation |
| `VLM_PIPELINE_COMPLETE.md` | Pipeline success summary |
| `QUICKSTART.md` | Quick start guide |
| `collect_data_via_cli.md` | CLI-based data collection |
| `BUGS_AND_FIXES.md` | Historical issues and fixes |
| `monorepo/VISION_SFT_COLLATOR_REFERENCE.md` | Collator technical details |

## 🚀 Quick Start (Recommended)

**Use synth-ai CLI for complete pipeline:**

```bash
# Run complete pipeline: collect → filter → train
bash examples/qwen_vl/run_vision_sft_pipeline.sh
```

This will:
1. Collect 100 episodes with gpt-5-nano (vision enabled)
2. Filter traces and export to SFT JSONL format
3. Optionally start SFT training

**Or step-by-step:**

```bash
# 1. Collect traces
uvx synth-ai eval --config examples/qwen_vl/configs/eval_gpt5nano_vision.toml

# 2. Filter and export
uvx synth-ai filter --config examples/qwen_vl/configs/filter_vision_sft.toml

# 3. Train SFT
cd /path/to/monorepo
uvx synth-ai train --type sft --config configs/vision_sft/crafter_qwen3vl_8b_gpt5nano.toml
```

📖 **Full guide:** See `collect_data_via_cli.md` for detailed CLI usage.

---

## Examples (Direct Python Scripts)

### 1. **crafter_qwen_vl_agent.py**
Run Crafter agent using Qwen-VL models via synth-ai's hosted inference.

**Models supported:**
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct` (or any Qwen3 VL variant)

**Usage:**
```bash
# Run with Qwen3-VL-4B
uv run python examples/qwen_vl/crafter_qwen_vl_agent.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --seeds 10 \
  --steps 20

# Run with Qwen3-VL-8B  
uv run python examples/qwen_vl/crafter_qwen_vl_agent.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --seeds 10 \
  --steps 20
```

**Requires:** Synth-AI API key (`SYNTH_API_KEY` environment variable)

---

### 2. **crafter_gpt5nano_agent.py**
Run Crafter agent using OpenAI's gpt-5-nano vision model.

**Usage:**
```bash
# Run with gpt-5-nano
uv run python examples/qwen_vl/crafter_gpt5nano_agent.py \
  --model gpt-5-nano \
  --seeds 10 \
  --steps 20

# Run with gpt-4o-mini for comparison
uv run python examples/qwen_vl/crafter_gpt5nano_agent.py \
  --model gpt-4o-mini-2024-07-18 \
  --seeds 10 \
  --steps 20
```

**Requires:** OpenAI API key (`OPENAI_API_KEY` environment variable)

---

### 3. **collect_vision_traces.py**
Collect vision traces for SFT dataset creation. Supports both Qwen-VL (synth) and OpenAI models.

**Usage:**
```bash
# Collect traces with gpt-5-nano
uv run python examples/qwen_vl/collect_vision_traces.py \
  --model gpt-5-nano \
  --provider openai \
  --episodes 100 \
  --max-steps 50 \
  --output-dir traces/gpt5nano_vision

# Collect traces with Qwen3-VL via synth
uv run python examples/qwen_vl/collect_vision_traces.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --provider synth \
  --episodes 100 \
  --max-steps 50 \
  --output-dir traces/qwen3vl_vision
```

**Output:** SQLite database with multimodal traces ready for SFT export.

---

## Vision Detection

CrafterPolicy automatically detects vision capability from model names:
- ✅ `gpt-5*` → Vision enabled
- ✅ `gpt-4o*` → Vision enabled  
- ✅ `*qwen-vl*` → Vision enabled
- ✅ `*qwen3-vl*` → Vision enabled

Or set explicitly: `policy.use_vision = True`

## Image Format

Crafter environment provides observations as:
- **64x64 PNG images**
- **Base64-encoded data URLs**
- Format: `"data:image/png;base64,iVBORw0KGgo..."`

## Next Steps

1. Run demo agents to verify vision inference works
2. Collect training traces with `collect_vision_traces.py`
3. Export to SFT JSONL format (see `vision_sft_rl.txt`)
4. Train VLM with LoRA (see monorepo SFT configs)
5. Fine-tune with RL/GRPO
