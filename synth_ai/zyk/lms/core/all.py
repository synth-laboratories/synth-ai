from synth_ai.zyk.lms.vendors.core.anthropic_api import AnthropicAPI
from synth_ai.zyk.lms.vendors.core.gemini_api import GeminiAPI
from synth_ai.zyk.lms.vendors.core.openai_api import (
    OpenAIPrivate,
    OpenAIStructuredOutputClient,
)
from synth_ai.zyk.lms.vendors.supported.deepseek import DeepSeekAPI
from synth_ai.zyk.lms.vendors.supported.together import TogetherAPI
from synth_ai.zyk.lms.vendors.supported.groq import GroqAPI
from synth_ai.zyk.lms.vendors.core.mistral_api import MistralAPI
from synth_ai.zyk.lms.vendors.supported.custom_endpoint import CustomEndpointAPI
from synth_ai.zyk.lms.vendors.supported.openrouter import OpenRouterAPI


class OpenAIClient(OpenAIPrivate):
    def __init__(self, synth_logging: bool = True):
        super().__init__(
            synth_logging=synth_logging,
        )


class AnthropicClient(AnthropicAPI):
    def __init__(self):
        super().__init__()


class GeminiClient(GeminiAPI):
    def __init__(self):
        super().__init__()


class DeepSeekClient(DeepSeekAPI):
    def __init__(self):
        super().__init__()


class TogetherClient(TogetherAPI):
    def __init__(self):
        super().__init__()


class GroqClient(GroqAPI):
    def __init__(self):
        super().__init__()


class MistralClient(MistralAPI):
    def __init__(self):
        super().__init__()


class CustomEndpointClient(CustomEndpointAPI):
    def __init__(self, endpoint_url: str):
        super().__init__(endpoint_url=endpoint_url)


class OpenRouterClient(OpenRouterAPI):
    def __init__(self):
        super().__init__()
