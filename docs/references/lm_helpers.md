# LM Helper Reference

Synth ships a suite of language-model utilities under `synth_ai/lm`. They provide consistent APIs for invoking providers, instrumenting calls, and adapting outputs for task apps.

## Modules at a glance

- `core/` – shared abstractions for chat completion engines, tool-calling helpers, and response normalization.
- `unified_interface.py` – primary entry point that exposes a provider-agnostic interface for chat models.
- `provider_support/` – adapters and capability flags for OpenAI, Anthropic, Groq, and Synth-hosted vLLM endpoints.
- `tools/` – definitions of tool schemas (e.g., the `interact` function used by task apps).
- `structured_outputs/` – helpers for parsing and validating structured model responses.
- `caching/` – disk/memory caches for deterministic runs and debugging.
- `cost/` – accounting for per-model pricing (useful when scoring runs).
- `LM_V2_TRACING_README.md` – guidance for instrumenting model calls with tracing v3.

## Task app integration

Task apps rely on the shared `interact` tool schema defined in `synth_ai/task/proxy.py`. When you call `prepare_for_openai` or `prepare_for_groq`, the payload is sanitized to include the tool automatically, ensuring LLM responses conform to the expected format.

## Structured outputs & reasoning

- Use `structured_outputs` to parse JSON fragments returned by policy models.
- Combine with tracing v3 to capture reasoning strings alongside actions for later analysis.

## Extending provider support

When adding a new provider:
1. Implement capability detection and request shaping in `provider_support/`.
2. Update `unified_interface.py` so higher-level code can route calls without change.
3. Ensure proxies in your task app call `prepare_for_<provider>` to maintain tool schemas and system hints.

