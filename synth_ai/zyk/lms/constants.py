OPENAI_REASONING_MODELS = ["o4", "o4-mini", "o3","o3-mini", "o1-mini", "o1"]
CLAUDE_REASONING_MODELS = ["claude-3-7-sonnet-latest"]
GEMINI_REASONING_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Gemini models that support thinking
GEMINI_REASONING_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
GEMINI_THINKING_BUDGETS = {
    "high": 10000,
    "medium": 5000,
    "low": 2500,
}

# Anthropic Sonnet 3.7 budgets
SONNET_37_BUDGETS = {
    "high": 8192,
    "medium": 4096,
    "low": 2048,
}

REASONING_MODELS = OPENAI_REASONING_MODELS + CLAUDE_REASONING_MODELS + GEMINI_REASONING_MODELS

SPECIAL_BASE_TEMPS = {model: 1 for model in REASONING_MODELS}