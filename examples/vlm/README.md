# Crafter VLM Pipeline

This folder captures the reference workflow for fine-tuning Crafter policies with
multimodal (text + image) prompts. It stitches together the new image-aware tracing
plumbing with lightweight utilities for dataset curation and training.

## Quick Start

1. **Verify image capture**  
   ```
   uv run python examples/vlm/crafter_image_only_agent.py --seed 7 --steps 5
   ```
   This writes PNG frames to `examples/vlm/output/frames/` and produces a JSONL preview
   of OpenAI-style image-only user messages.

2. **Collect traced rollouts**  
   Use the Crafter task app (or your existing pipeline) with tracing enabled. The new
   tracing schema automatically records `observation_image_base64` and stores image parts
   in LM call records.

3. **Export multimodal SFT rows**  
   ```
   uv run python examples/warming_up_to_rl/export_trace_sft.py \
       --db traces/v3/synth_ai.db \
       --output examples/vlm/output/crafter_traces_full.jsonl
   ```
   The exporter now emits `metadata.has_image`, `metadata.user_has_image`, and
   `metadata.assistant_has_image` flags per turn.

4. **Filter to image-rich turns**  
   ```
   uv run python examples/vlm/filter_image_rows.py \
       --input examples/vlm/output/crafter_traces_full.jsonl \
       --output examples/vlm/output/crafter_vlm_dataset.jsonl
   ```

5. **(Optional) Split validation or augment**, then upload using the standard CLI:
   ```
   uv run python examples/warming_up_to_rl/run_fft_and_save.py \
       --toml examples/vlm/configs/crafter_vlm_gpt4o.toml \
       --data examples/vlm/output/crafter_vlm_dataset.jsonl
   ```

## Config & Utilities

| File | Purpose |
| --- | --- |
| `configs/crafter_vlm_gpt4o.toml` | Sample Synth job targeting an image-capable model (`openai/gpt-4o-mini`). Set `job.data` or pass `--data` explicitly. |
| `crafter_image_only_agent.py` | Captures frames and builds image-only prompts for sanity checks. |
| `filter_image_rows.py` | Extracts rows with image parts from exported JSONL datasets. |

## Notes & Next Steps

- The training config assumes full-finetuning (`mode = "sft_offline"`). Adjust the
  model id, hardware, or hyperparameters to match available infrastructure.
- Dataset rows emitted by `export_trace_sft.py` already contain OpenAI multimodal
  content parts like:
  ```json
  {
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." }}
    ]
  }
  ```
- See `PROPOSAL.md` for a deeper dive into outstanding work (longer rollouts,
  richer multimodal augmentations, evaluation ideas).
