---
name: swarms-research
description: Work with Cloud Swarms Track B notebook research primitives — experiments, append-only research logs with corrections, interactive versioned artifacts, and consolidated run reviews.
---

# Cloud Swarms Research Workflows (Track B)

Track B primitives organize iterative research inside a `notebook_id` across one or more runs with immutable provenance.

## 1. Experiments

Define structured hypotheses and multi-arm specifications tied to a run:
- Create: `swarms_experiment_create` (`notebook_id`, `body: {run_id, title, specification, arms, hypothesis?, pinned_inputs?, stopping_criteria?}`, `idempotency_key`).
- Inspect: `swarms_experiments_list` (`notebook_id`, optional `run_id`) and `swarms_experiment_get` (`experiment_id`) for `revision`, `provenance`, and `trials`.
- Revise: `swarms_experiment_revise` (`experiment_id`, `body: {title?, specification?, hypothesis?, arms?, stopping_criteria?}`, `idempotency_key`). Creates a new immutable revision without mutating prior history.

## 2. Research Log (Append-Only + Corrections)

Maintain an auditable lab notebook per `notebook_id`:
- Initialize once: `swarms_research_log_init` (`notebook_id`, `body: {title, summary?}`, `idempotency_key`).
- Append entries: `swarms_research_log_append` (`notebook_id`, `body: {category, text, run_id?, author_kind?, author_label?, sources?, artifact_ids?}`, `idempotency_key`).
- Search: `swarms_research_log_search` (`notebook_id`, optional filters `query`, `category`, `author_kind`, `run_id`).
- Correct prior entries: never overwrite history — call `swarms_research_log_correct` (`notebook_id`, `entry_id`, `body: {text, run_id?, sources?, artifact_ids?}`, `idempotency_key`) to append a linked correction referencing `entry_id`.
- Export: `swarms_research_log_export` (`notebook_id`, optional `format`, `run_id`).

## 3. Interactive Artifacts

Store versioned reports, tables, plots, or structured deliverables linked to a notebook and run:
- Create: `swarms_artifact_create` (`body: {notebook_id, title, kind, content, summary, run_id?, origin?}`, `idempotency_key`).
- List & inspect: `swarms_artifacts_list` (`notebook_id`, optional `run_id`) and `swarms_artifact_get` (`artifact_id`) for `current_revision` and full `revisions` history.
- Publish a new revision: `swarms_artifact_update` (`artifact_id`, `body: {content, summary, author_kind?}`, `idempotency_key`).
- Read content with integrity pinning: `swarms_artifact_content` (`artifact_id`, optional `revision_id`, `digest_sha256`).
- Sharing: `swarms_artifact_share` (`artifact_id`, `body: {scope}`, `idempotency_key`) and `swarms_artifact_share_revoke` (`artifact_id`).

## 4. Consolidated Run Review

Call `swarms_run_review` (`run_id`) to retrieve a unified view of `pacing`, `experiments`, `research_log`, and `artifacts` for a run.
