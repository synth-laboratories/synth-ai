import re
from typing import Any, List, Pattern

from zyk.src.lms.core.all import (
    AnthropicClient,
    DeepSeekClient,
    GeminiClient,
    OpenAIClient,
    OpenAIStructuredOutputClient,
    TogetherClient,
)

openai_naming_regexes: List[Pattern] = [
    re.compile(r'^o1-.*$'),
    re.compile(r'^gpt-.*$'),
]
openai_formatting_model_regexes: List[Pattern] = [
    re.compile(r'^gpt-4o-.*$'), 
]
anthropic_naming_regexes: List[Pattern] = [
    re.compile(r'^claude-.*$'),
]
gemini_naming_regexes: List[Pattern] = [
    re.compile(r'^gemini-.*$'),
]
deepseek_naming_regexes: List[Pattern] = [
    re.compile(r'^deepseek-.*$'),
]
together_naming_regexes: List[Pattern] = [
    re.compile(r'^.*\/.*$'),
]

def get_client(
    model_name: str,
    with_formatting: bool = False
) -> Any:
    if any(regex.match(model_name) for regex in openai_naming_regexes):
        use_structured = with_formatting and any(regex.match(model_name) for regex in openai_formatting_model_regexes)
        if use_structured:
            return OpenAIStructuredOutputClient()
        else:
            return OpenAIClient()
    elif any(regex.match(model_name) for regex in anthropic_naming_regexes):
        if with_formatting:
            raise NotImplementedError("Structured outputs not supported for Anthropic")
            #eturn AnthropicStructuredOutputClient(used_for_structured_outputs=False)
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
