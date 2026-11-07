# Changelog

## Unreleased

Runs no longer take a project. `launch` and `run` drop `project_id`, the CLI drops
`--project-id`, and `SYNTH_PROJECT_ID` is no longer read. `create_project`, `project` and
`projects` are gone, and files now live at `/v1/files`, so `upload_file`, `upload_text`,
`prepare_upload`, `files` and `file` take no project. Named methods take a published task:
`method={"release": ..., "task": ...}` in Python and `--method-task` in the CLI.

## 0.26.0

Public API v1: generated models and operations, opaque resource IDs, decimal-string
money, typed Problem Details, cursor pagination, resumable event streams, OAuth
PKCE, and matching local/hosted MCP tools. Async and synchronous clients share
one transport. API-key management requires a signed-in session.

Adds owner resource intents, provider presets through profile overrides, and CLI
`watch` with terminal snapshots and bounded streaming.

Adds execution method selection for Status Quo, Fork & Flush and Forgetting through
`method=` and `--method`. The two research methods require a published project
pairing. Adds file upload and run attachment through `upload_file`, `upload_text` and
`synth swarms upload`.

Adds `run.selection` for research run results and the selected snapshot output. Removes
`method.forge_profile_digest` from run snapshots.

Earlier releases and their changelog remain at tag `legacy/0.24.0`.
