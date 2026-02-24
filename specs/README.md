# Synth SDK Specifications

This folder is the SDK-specific source of truth for the canonical Synth client surface and its HTTP contract.

## Files

- `sdk_logic.md`: canonical object model, namespaces, sync/async behavior, and invariants.
- `http_openai.md`: transport contract for HTTP endpoints, including OpenAI-compatible chat completions.
- `api_stability_lifecycle.md`: SDK/API stability levels, deprecation windows, and release lifecycle policy.

## Scope

These specs describe the current canonical SDK surface in `synth_ai`:

- `SynthClient` and `AsyncSynthClient` front-door clients
- First-class `containers`, `pools`, and `tunnels` abstractions
- Optimization, inference, graphs, and verifiers API areas

For machine-readable endpoint schemas, see:

- `openapi/synth-api-v1.yaml`
