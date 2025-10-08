"""Inference module for OpenAI-compatible API clients."""

from .openai_client import OpenAIClient, create_inference_client

__all__ = ["OpenAIClient", "create_inference_client"]
