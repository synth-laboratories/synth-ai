"""
Constants for LM module.

This module defines constants used across the LM (Language Model) module,
including model names, reasoning models, thinking budgets, and temperature settings.
"""

# Reasoning model names by provider
OPENAI_REASONING_MODELS = ["o4", "o4-mini", "o3", "o3-mini", "o1-mini", "o1"]
CLAUDE_REASONING_MODELS = ["claude-3-7-sonnet-latest"]
GEMINI_REASONING_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Gemini models that support thinking
GEMINI_REASONING_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
GEMINI_THINKING_BUDGETS = {
    "high": 10000,    # High thinking budget for complex reasoning
    "medium": 5000,   # Medium thinking budget for standard reasoning
    "low": 2500,      # Low thinking budget for simple reasoning
}

# Anthropic Sonnet 3.7 budgets
SONNET_37_BUDGETS = {
    "high": 8192,     # High budget for complex tasks
    "medium": 4096,   # Medium budget for standard tasks
    "low": 2048,      # Low budget for simple tasks
}

# Combined list of all reasoning models
REASONING_MODELS = OPENAI_REASONING_MODELS + CLAUDE_REASONING_MODELS + GEMINI_REASONING_MODELS

# Special base temperatures for reasoning models (all set to 1.0)
SPECIAL_BASE_TEMPS = {model: 1 for model in REASONING_MODELS}
