# GRPO Enron Task App

This example mirrors the layout of the Crafter task app while targeting the
Enron email question answering environment found under
`synth_ai.environments.examples.enron`. It exposes the usual task-app config
entry point (`grpo_enron.py`) together with a compatibility wrapper
(`grpo_enron_task_app.py`) for running the FastAPI service directly or via
Modal.

The configuration builds a lightweight dataset from the Enron sample QA
corpus, publishes minimal task metadata, and provides a no-op rollout bridge
that yields the initial observation for the selected instance. The intent is
to serve as a starting point for richer integrations (rubrics, hosted env
service, policy orchestration) matching the Crafter example.
