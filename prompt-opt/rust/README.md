# mipro_v2

General-purpose batch prompt/policy optimization.

This crate is standalone and has **no dependency on Horizons**.

## What It Does

- Splits a dataset into train and holdout sets deterministically (seeded).
- Generates prompt/policy variants via a `VariantSampler`.
- Evaluates candidates on the holdout set via an `Evaluator` (LLM + metric).
- Picks the best candidate and iterates with early stopping.

## Core Types

- `Policy`: prompt template (supports `{input}` substitution).
- `Dataset`: collection of examples (`input`, `expected`, `metadata`).
- `Optimizer`: batch search loop with `MiproConfig`.

