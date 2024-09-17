from typing import Type
from pydantic import BaseModel
import json
import ast

def pull_out_structured_output(
    response_raw: str, response_model: Type[BaseModel]
) -> BaseModel:
    assert isinstance(response_raw, str), f"Response raw is not a string: {type(response_raw)}"
    if "```" in response_raw and "json" in response_raw:
        response_prepared = response_raw.split("```json")[1].split("```")[0].strip()
    elif "```" in response_raw:
        response_prepared = response_raw.split("```")[1].strip()
    else:
        response_prepared = response_raw.strip()
    # TODO: review???? seems dangerous
    response_prepared = response_prepared.replace("null", '"None"')
    try:
        response = json.loads(response_prepared)
        final = response_model(**response)
    except Exception as e:
        response_prepared = response_prepared.strip()
        try:
            response = json.loads(response_prepared)
            final = response_model(**response)
        except json.JSONDecodeError:
            response_prepared = response_prepared.replace('\n', '').replace('\\n', '')
            response_prepared = response_prepared.replace('\\"', '"')
            try:
                
                response = ast.literal_eval(response_prepared)
                final = response_model(**response)
            except Exception as inner_e:
                raise ValueError(
                    f"""Failed to parse response as {response_model}: {inner_e} - {response_prepared}"""
                )
    return final