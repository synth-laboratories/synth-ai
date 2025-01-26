import re
from typing import Any, List, Pattern

from zyk.lms.core.all import (
    AnthropicClient,
    DeepSeekClient,
    GeminiClient,
    OpenAIClient,
    OpenAIStructuredOutputClient,
    TogetherClient,
)

openai_naming_regexes: List[Pattern] = [
    re.compile(r"^(ft:)?(o1(-.*)?|gpt-.*)$"),
]
openai_formatting_model_regexes: List[Pattern] = [
    re.compile(r"^(ft:)?gpt-4o(-.*)?$"),
]
anthropic_naming_regexes: List[Pattern] = [
    re.compile(r"^claude-.*$"),
]
gemini_naming_regexes: List[Pattern] = [
    re.compile(r"^gemini-.*$"),
]
deepseek_naming_regexes: List[Pattern] = [
    re.compile(r"^deepseek-.*$"),
]
together_naming_regexes: List[Pattern] = [
    re.compile(r"^.*\/.*$"),
]


def get_client(
    model_name: str,
    with_formatting: bool = False,
    synth_logging: bool = False,
) -> Any:
    # print("With formatting", with_formatting)
    if any(regex.match(model_name) for regex in openai_naming_regexes):
        # print("Openai model regexes", openai_naming_regexes)
        # print("Model name", model_name)
        model_match = any(
            regex.match(model_name) for regex in openai_formatting_model_regexes
        )
        # print("Model match", model_match)
        use_structured = with_formatting and model_match
        # print("use_structured", use_structured)
        if use_structured:
            return OpenAIStructuredOutputClient(
                synth_logging=synth_logging,
            )
        else:
            return OpenAIClient(
                synth_logging=synth_logging,
            )
    elif any(regex.match(model_name) for regex in anthropic_naming_regexes):
        if with_formatting:
            # Instead of raising error, return a client that will fallback to OpenAI
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
    else:
        raise ValueError(f"Invalid model name: {model_name}")
