# Fine-tuning Workflow (FFT & QLoRA)

Synth supports two SFT styles today:

1. **Full fine-tune (FFT)** – existing Crafter example (`examples/warming_up_to_rl/run_fft_and_save.py`).
2. **QLoRA** – enable `training.use_qlora = true` to fit larger models on constrained hardware. The Crafter example will migrate to QLoRA as part of the roadmap.

## Using the legacy script

```bash
BACKEND_BASE_URL=https://<backend>/api \
SYNTH_API_KEY=<key> \
uv run python examples/warming_up_to_rl/run_fft_and_save.py \
  --toml examples/warming_up_to_rl/configs/crafter_fft.toml \
  --data /absolute/path/to/dataset.jsonl \
  --poll-seconds 3600
```

Highlights (`run_fft_and_save.py`):
- Uploads training and optional validation JSONL to `/learning/files` (`post_multipart`).
- Submits the job payload as defined by the TOML (model, hyperparameters, compute settings).
- Polls the backend until completion and writes the resulting model ID to `ft_model_id.txt`.

## Using `uvx synth-ai train`

The new CLI wraps the same flow with interactive prompts (`synth_ai/api/train/cli.py:95`).

```bash
uvx synth-ai train --type sft --config examples/warming_up_to_rl/configs/crafter_fft.toml
```

What happens behind the scenes:
1. Select or confirm a `.env` file that defines `SYNTH_API_KEY` (and optional defaults like `BACKEND_BASE_URL`).
2. Validate the dataset path from the TOML or prompt for `--dataset`.
3. Upload training/validation files and remember their IDs.
4. Build the payload via `build_sft_payload` (`synth_ai/api/train/builders.py:72`).
5. Submit the job, optionally polling status (`--no-poll` disables).

### Hyperparameters & QLoRA

- Set `hyperparameters.train_kind = "fft"` (default) or tailor batch sizes, LR, sequence length.
- Toggle `training.use_qlora = true` to enable QLoRA-friendly configuration in the payload metadata.
- Provide validation settings under `[training.validation]` to compute metrics during training (integrated into the payload metadata).

### Outputs

- Successful runs return the fine-tuned model ID (printed and persisted when the legacy script is used).
- Use that ID in evaluation TOMLs (`model = "ft:<id>"`) or RL configs (`[model].source = "ft:<id>"`).

## Roadmap

- **Crafter QLoRA**: migrate the Crafter FFT example to a QLoRA configuration with updated hyperparameters and dataset instructions.
- **Documentation sync**: once QLoRA configs are live, cross-link them from this page and the examples roadmap.

