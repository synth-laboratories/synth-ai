# Example Refresh Roadmap

This roadmap tracks the documentation-driven updates we plan to make across the example directories.

## Evals (`examples/evals`)
- Rebase evaluation scripts on top of `examples/warming_up_to_rl/run_eval.py`.
- Share TOML schemas and policy helpers with the Crafter example.
- Ensure every eval README links to `docs/workflows/evaluation.md` for canonical instructions.

## Finetuning (`examples/finetuning`)
- Introduce a Crafter QLoRA config and dataset recipe matching the FFT guide.
- Update readmes to reference `uvx synth-ai train --type sft` instead of bespoke scripts.
- Provide dataset verification scripts that leverage the tracing exporter.

## RL (`examples/rl`)
- Move the primary RL walkthrough to the Math task app (FFT-first).
- Supply TOMLs that mirror the train CLI defaults.
- Document evaluation checkpoints so users can compare FFT vs RL improvements.

Progress on these items will roll back into the workflow docs to keep instructions in sync with the codebase.

