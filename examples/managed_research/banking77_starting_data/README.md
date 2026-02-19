# Banking77 Starting Data Bundle

This directory is uploaded through `POST /smr/projects/{project_id}/starting-data/upload-urls`
(via `SmrControlClient`) before triggering a managed research run.

Contents:

- `input_spec.json`: Managed-research `execution.input_spec` payload for an eval job.
- `banking77_eval_output_seed20.json`: Sample baseline eval output over seeds `0..19`.

Recommended dataset ref for upload:

- `starting-data/banking77`
