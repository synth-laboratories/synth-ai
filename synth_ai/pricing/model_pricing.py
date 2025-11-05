"""Static pricing table for supported models.

This module provides per-token pricing used by the SDK status commands.
Rates are expressed in USD per token and split into input/output prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TokenRates:
    input_usd: float
    output_usd: float


# Default per-token prices (USD), sourced Nov 3, 2025 — update as contracts change
MODEL_PRICES: Dict[str, Dict[str, TokenRates]] = {
    # OpenAI official pricing
    "openai": {
        # GPT-5 family
        "gpt-5":       TokenRates(input_usd=0.00000125, output_usd=0.00001000),  # $1.25 / $10 per 1M
        "gpt-5-mini":  TokenRates(input_usd=0.00000025, output_usd=0.00000200),  # $0.25 / $2.00 per 1M
        "gpt-5-nano":  TokenRates(input_usd=0.00000005, output_usd=0.00000040),  # $0.05 / $0.40 per 1M

        "gpt-4o-mini": TokenRates(input_usd=0.00000015, output_usd=0.00000060),  # $0.15 / $0.60 per 1M
        "gpt-4o":      TokenRates(input_usd=0.00000250, output_usd=0.00001000),  # $2.50 / $10.00 per 1M
    },
    # Groq OSS via OpenAI-compatible path (latest Groq docs)
    "groq": {
        "openai/gpt-oss-20b":  TokenRates(input_usd=0.000000075, output_usd=0.000000300),  # $0.075 / $0.30 per 1M
        
        "openai/gpt-oss-120b": TokenRates(input_usd=0.000000150, output_usd=0.000000600),  # $0.15 / $0.60 per 1M

        # Additional Groq on-demand models
        "moonshotai/kimi-k2-0905":        TokenRates(input_usd=0.000001000, output_usd=0.000003000),  # $1.00 / $3.00 per 1M
        
        "meta/llama-guard-4-12b":         TokenRates(input_usd=0.000000200, output_usd=0.000000200),  # $0.20 / $0.20 per 1M
        "qwen/qwen3-32b":                 TokenRates(input_usd=0.000000290, output_usd=0.000000590),  # $0.29 / $0.59 per 1M
        "meta/llama-3.3-70b-versatile":   TokenRates(input_usd=0.000000590, output_usd=0.000000790),  # $0.59 / $0.79 per 1M
        "meta/llama-3.1-8b-instant":      TokenRates(input_usd=0.000000050, output_usd=0.000000080),  # $0.05 / $0.08 per 1M
    },
    # Google Gemini pricing — per-token USD (per 1M ÷ 1e6), Nov 3, 2025
    "google": {
        # Gemini 2.5 Pro (two tiers by prompt size)
        "gemini-2.5-pro":        TokenRates(input_usd=0.00000125, output_usd=0.00001000),  # <=200k tokens
        "gemini-2.5-pro-gt200k": TokenRates(input_usd=0.00000250, output_usd=0.00001500),  # >200k tokens

        # Gemini 2.5 Flash (hybrid reasoning)
        "gemini-2.5-flash":      TokenRates(input_usd=0.00000030, output_usd=0.00000250),

        # Gemini 2.5 Flash-Lite (cheapest)
        "gemini-2.5-flash-lite": TokenRates(input_usd=0.00000010, output_usd=0.00000040),
    },
}

