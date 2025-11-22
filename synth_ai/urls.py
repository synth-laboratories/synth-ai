import os


BACKEND_BASE = os.getenv("SYNTH_BACKEND_OVERRIDE") or "https://agent-learning.onrender.com"
BACKEND_API = BACKEND_BASE + "/api"
BACKEND_PO = BACKEND_API + "/prompt-learning"
BACKEND_URL_SYNTH_RESEARCH_BASE = BACKEND_BASE + "/api/synth-research"

BACKEND_URL_SYNTH_RESEARCH_OPENAI = BACKEND_URL_SYNTH_RESEARCH_BASE + "/v1"  # For OpenAI SDKs (appends /chat/completions)
BACKEND_URL_SYNTH_RESEARCH_ANTHROPIC = BACKEND_URL_SYNTH_RESEARCH_BASE  # For Anthropic SDKs (appends /v1/messages)


FRONTEND_BASE = os.getenv("SYNTH_FRONTEND_OVERRIDE") or "https://www.usesynth.ai"
