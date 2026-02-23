# rlm-rs

Standalone Reward Language Model (RLM) verifier for evaluating outputs against weighted signals.

This crate is general-purpose and has **no dependency on Horizons**.

## Core Ideas

- Define a set of `RewardSignal`s with weights.
- Verify a `VerificationCase` to produce a `RewardOutcome` in `[0,1]`.
- Generate an `EvalReport` and render it as Markdown or JSON.

## Crate Name

The Cargo package is `rlm-rs`. In Rust code, hyphens become underscores: `rlm_rs`.
