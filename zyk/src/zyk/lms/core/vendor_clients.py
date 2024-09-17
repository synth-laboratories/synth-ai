from typing import Any, List, Pattern
import re
from zyk.src.zyk.lms.core.all import OpenAIClient, OpenAIStructuredOutputClient, AnthropicClient

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
    else:
        raise ValueError(f"Invalid model name: {model_name}")
