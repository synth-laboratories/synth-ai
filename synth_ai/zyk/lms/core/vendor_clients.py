import re
from typing import Any, List, Pattern

from synth_ai.zyk.lms.core.all import (
    AnthropicClient,
    DeepSeekClient,
    GeminiClient,
    GroqAPI,
    MistralAPI,
    # OpenAIClient,
    OpenAIStructuredOutputClient,
    TogetherClient,
)

openai_naming_regexes: List[Pattern] = [
    re.compile(r"^(ft:)?(o[1,3](-.*)?|gpt-.*)$"),
]
openai_formatting_model_regexes: List[Pattern] = [
    re.compile(r"^(ft:)?gpt-4o(-.*)?$"),
]
anthropic_naming_regexes: List[Pattern] = [
    re.compile(r"^claude-.*$"),
]
gemini_naming_regexes: List[Pattern] = [
    re.compile(r"^gemini-.*$"),
    re.compile(r"^gemma[2-9].*$"),
]
deepseek_naming_regexes: List[Pattern] = [
    re.compile(r"^deepseek-.*$"),
]
together_naming_regexes: List[Pattern] = [
    re.compile(r"^.*\/.*$"),
]

groq_naming_regexes: List[Pattern] = [
    re.compile(r"^llama-3.3-70b-versatile$"),
    re.compile(r"^llama-3.1-8b-instant$"),
    re.compile(r"^qwen-2.5-32b$"),
    re.compile(r"^deepseek-r1-distill-qwen-32b$"),
    re.compile(r"^deepseek-r1-distill-llama-70b-specdec$"),
    re.compile(r"^deepseek-r1-distill-llama-70b$"),
    re.compile(r"^llama-3.3-70b-specdec$"),
    re.compile(r"^llama-3.2-1b-preview$"),
    re.compile(r"^llama-3.2-3b-preview$"),
    re.compile(r"^llama-3.2-11b-vision-preview$"),
    re.compile(r"^llama-3.2-90b-vision-preview$"),
]

mistral_naming_regexes: List[Pattern] = [
    re.compile(r"^mistral-.*$"),
]


def get_client(
    model_name: str,
    with_formatting: bool = False,
    synth_logging: bool = True,
) -> Any:
    # print("With formatting", with_formatting)
    if any(regex.match(model_name) for regex in openai_naming_regexes):
        # print("Returning OpenAIStructuredOutputClient")
        return OpenAIStructuredOutputClient(
            synth_logging=synth_logging,
        )
    elif any(regex.match(model_name) for regex in anthropic_naming_regexes):
        if with_formatting:
            client = AnthropicClient()
            client._hit_api_async_structured_output = OpenAIStructuredOutputClient(
                synth_logging=synth_logging
            )._hit_api_async
            return client
        else:
            return AnthropicClient()
    elif any(regex.match(model_name) for regex in gemini_naming_regexes):
        return GeminiClient()
    elif any(regex.match(model_name) for regex in deepseek_naming_regexes):
        return DeepSeekClient()
    elif any(regex.match(model_name) for regex in together_naming_regexes):
        return TogetherClient()
    elif any(regex.match(model_name) for regex in groq_naming_regexes):
        return GroqAPI()
    elif any(regex.match(model_name) for regex in mistral_naming_regexes):
        return MistralAPI()
    else:
        raise ValueError(f"Invalid model name: {model_name}")
