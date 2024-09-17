
from zyk.src.lms.vendors.core.anthropic_api import AnthropicAPI
from zyk.src.lms.vendors.core.gemini_api import GeminiAPI
from zyk.src.lms.vendors.core.openai_api import (
    OpenAIPrivate,
    OpenAIStructuredOutputClient
)
from zyk.src.lms.vendors.supported.deepseek import DeepSeekAPI
from zyk.src.lms.vendors.supported.together import TogetherAPI


class OpenAIClient(OpenAIPrivate):
    def __init__(self):
        super().__init__()

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

    