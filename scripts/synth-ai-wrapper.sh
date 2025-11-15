#!/bin/bash
# Wrapper script that sets RUST_LOG to suppress noisy codex_otel logs
# Usage: ./synth-ai-wrapper.sh <command> [args...]

export RUST_LOG="${RUST_LOG:-codex_otel::otel_event_manager=warn}"

# Run the actual synth-ai command
exec synth-ai "$@"







