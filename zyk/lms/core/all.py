from zyk.lms.vendors.core.anthropic_api import AnthropicAPI
from zyk.lms.vendors.core.gemini_api import GeminiAPI
from zyk.lms.vendors.core.openai_api import OpenAIPrivate, OpenAIStructuredOutputClient
from zyk.lms.vendors.supported.deepseek import DeepSeekAPI
from zyk.lms.vendors.supported.together import TogetherAPI


class OpenAIClient(OpenAIPrivate):
    def __init__(self, synth_logging: bool = False):
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
