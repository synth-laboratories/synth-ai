import os

# Base URL for all backends
BACKEND_URL_BASE = os.getenv("SYNTH_BACKEND_OVERRIDE") or "https://www.api.usesynth.ai"

# Synth Research API base (supports OpenAI, Anthropic, and custom formats)
# Real routes: /api/synth-research/chat/completions, /api/synth-research/messages
# V1 routes: /api/synth-research/v1/chat/completions, /api/synth-research/v1/messages
BACKEND_URL_SYNTH_RESEARCH_BASE = BACKEND_URL_BASE + "/api/synth-research"

# Provider-specific URLs (for SDKs that expect standard paths)
BACKEND_URL_SYNTH_RESEARCH_OPENAI = BACKEND_URL_SYNTH_RESEARCH_BASE + "/v1"  # For OpenAI SDKs (appends /chat/completions)
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = BACKEND_URL_SYNTH_RESEARCH_BASE  # For Anthropic SDKs (appends /v1/messages)




FRONTEND_URL_BASE = os.getenv("SYNTH_FRONTEND_OVERRIDE") or "https://www.usesynth.ai"
