import ast
import json
import logging
import re
from typing import Dict, List, Type, Union

from pydantic import BaseModel

from zyk.lms.vendors.base import VendorBase
from zyk.lms.vendors.core.openai_api import OpenAIStructuredOutputClient


def pull_out_structured_output(
    response_raw: str, response_model: Type[BaseModel]
) -> BaseModel:
    logger = logging.getLogger(__name__)
    logger.debug(f"Raw response received: {response_raw}")

    assert isinstance(
        response_raw, str
    ), f"Response raw is not a string: {type(response_raw)}"

    # Use regex to extract JSON content within ```json ... ```
    json_pattern = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)
    match = json_pattern.search(response_raw)
    if match:
        response_prepared = match.group(1).strip()
    else:
        # Fallback to existing parsing if no code fencing is found
        if "```" in response_raw:
            response_prepared = response_raw.split("```")[1].strip()
        else:
            response_prepared = response_raw.strip()

    # Replace "null" with '"None"' if needed (ensure this aligns with your data)
    response_prepared = response_prepared.replace("null", '"None"')

    logger.debug(f"Prepared response for JSON parsing: {response_prepared}")

    try:
        response = json.loads(response_prepared)
        final = response_model(**response)
    except json.JSONDecodeError as e:
        # Attempt to parse using ast.literal_eval as a fallback
        response_prepared = response_prepared.replace("\n", "").replace("\\n", "")
        response_prepared = response_prepared.replace('\\"', '"')
        try:
            response = ast.literal_eval(response_prepared)
            final = response_model(**response)
        except Exception as inner_e:
            raise ValueError(
                f"Failed to parse response as {response_model}: {inner_e} - {response_prepared}"
            )
    except Exception as e:
        raise ValueError(
            f"Failed to parse response as {response_model}: {e} - {response_prepared}"
        )
    return final


def fix_errant_stringified_json_sync(
    response_raw: str,
    response_model: Type[BaseModel],
    models: List[str] = ["gpt-4o-mini", "gpt-4o"],
) -> BaseModel:
    try:
        return pull_out_structured_output(response_raw, response_model)
    except ValueError as e:
        mini_client = OpenAIStructuredOutputClient()
        messages = [
            {
                "role": "system",
                "content": "An AI attempted to generate stringified json that could be extracted into the provided Pydantic model. This json cannot be extracted, and an error message is provided to elucidate why. Please review the information and return a corrected response. Do not materially change the content of the response, just formatting where necessary.",
            },
            {
                "role": "user",
                "content": f"# Errant Attempt\n{response_raw}\n# Response Model\n {response_model.model_json_schema()}\n# Error Message\n {str(e)}.",
            },
        ]
        for model in models:
            if model == "gpt-4o":
                print(
                    "Warning - using gpt-4o for structured output correction - this could add up over time (latency, cost)"
                )
            try:
                fixed_response = mini_client._hit_api_sync(
                    model, messages=messages, lm_config={"temperature": 0.0}
                )
                return pull_out_structured_output(fixed_response, response_model)
            except Exception as e:
                pass
        raise ValueError("Failed to fix response using any model")


async def fix_errant_stringified_json_async(
    response_raw: str,
    response_model: Type[BaseModel],
    models: List[str] = ["gpt-4o-mini", "gpt-4o"],
) -> BaseModel:
    try:
        return pull_out_structured_output(response_raw, response_model)
    except ValueError as e:
        mini_client = OpenAIStructuredOutputClient()
        messages = [
            {
                "role": "system",
                "content": "An AI attempted to generate stringified json that could be extracted into the provided Pydantic model. This json cannot be extracted, and an error message is provided to elucidate why. Please review the information and return a corrected response. Do not materially change the content of the response, just formatting where necessary.",
            },
            {
                "role": "user",
                "content": f"# Errant Attempt\n{response_raw}\n# Response Model\n {response_model.model_json_schema()}\n# Error Message\n {str(e)}.",
            },
        ]
        for model in models:
            if model == "gpt-4o":
                print(
                    "Warning - using gpt-4o for structured output correction - this could add up over time (latency, cost)"
                )
            try:
                fixed_response = await mini_client._hit_api_async(
                    model, messages=messages, lm_config={"temperature": 0.0}
                )
                return pull_out_structured_output(fixed_response, response_model)
            except Exception as e:
                pass
        raise ValueError("Failed to fix response using any model")


async def fix_errant_forced_async(
    messages: List[Dict],
    response_raw: str,
    response_model: Type[BaseModel],
    model: str,
) -> BaseModel:
    try:
        return pull_out_structured_output(response_raw, response_model)
    except ValueError as e:
        client = OpenAIStructuredOutputClient()
        messages = [
            {
                "role": "system",
                "content": "An AI attempted to generate stringified json that could be extracted into the provided Pydantic model. This json cannot be extracted, and an error message is provided to elucidate why. Please review the information and return a corrected response. Do not materially change the content of the response, just formatting where necessary.",
            },
            {
                "role": "user",
                "content": f"<previous messages>\n<system_message>{messages[0]['content']}</system_message>\n<user_message>{messages[1]['content']}</user_message></previous messages> # Errant Attempt\n{response_raw}\n# Response Model\n {response_model.model_json_schema()}\n# Error Message\n {str(e)}.",
            },
        ]
        # print("Broken response:")
        # print(response_raw)
        fixed_response = await client._hit_api_async_structured_output(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=0.0,
        )
        # print("Fixed response:")
        # print(fixed_response)
        return fixed_response


def fix_errant_forced_sync(
    response_raw: str,
    response_model: Type[BaseModel],
    model: str,
) -> BaseModel:
    client = OpenAIStructuredOutputClient()
    messages = [
        {
            "role": "system",
            "content": "An AI attempted to generate a response that could be extracted into the provided Pydantic model. This response cannot be extracted. Please review the information and return a corrected response.",
        },
        {
            "role": "user",
            "content": f"# Errant Attempt\n{response_raw}\n# Response Model\n {response_model.model_json_schema()}",
        },
    ]
    # print("Broken response:")
    # print(response_raw)
    fixed_response = client._hit_api_sync_structured_output(
        model=model, messages=messages, response_model=response_model, temperature=0.0
    )
    # print("Fixed response:")
    # print(fixed_response)
    return fixed_response
