from typing import List

import openai
from pydantic import BaseModel

from synth_ai.zyk import LM

class Person(BaseModel):
    name: str
    age: int
    hobbies: List[str]


TEST_PROMPT = "Extract information about a person from this text: John is 30 years old and enjoys reading, hiking, and photography."


def test_openai_structured_lm():
    lm = LM(
        model_name="gpt-4o-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="",
        user_message=TEST_PROMPT,
        response_model=Person,
    )

    assert isinstance(response.structured_output, Person)
    assert response.structured_output.name == "John"
    assert response.structured_output.age == 30
    assert set(response.structured_output.hobbies) == {
        "reading",
        "hiking",
        "photography",
    }

def test_anthropic_structured_lm():
    lm = LM(
        model_name="claude-3-haiku-20240307",
        formatting_model_name="claude-3-haiku-20240307",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that extracts structured information.",
        user_message=TEST_PROMPT,
        response_model=Person,
    )

    assert isinstance(response.structured_output, Person)
    assert response.structured_output.name == "John"
    assert response.structured_output.age == 30
    assert set(response.structured_output.hobbies) == {
        "reading",
        "hiking",
        "photography",
    }


# def test_gemini_structured():
#     client = GeminiAPI(
#         used_for_structured_outputs=True,
#         exceptions_to_retry=[],
#     )

#     response = client._hit_api_sync(
#         model="gemini-2.0-flash",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant that extracts structured information.",
#             },
#             {"role": "user", "content": TEST_PROMPT},
#         ],
#         response_model=Person,
#         #temperature=0,
#     )

#     assert isinstance(response.structured_output, Person)
#     assert response.structured_output.name == "John"
#     assert response.structured_output.age == 30
#     assert set(response.structured_output.hobbies) == {
#         "reading",
#         "hiking",
#         "photography",
#     }


def test_gemini_structured_lm():
    lm = LM(
        model_name="gemini-2.0-flash",
        formatting_model_name="gemini-2.0-flash",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that extracts structured information.",
        user_message=TEST_PROMPT,
        response_model=Person,
    )

    assert isinstance(response.structured_output, Person)
    assert response.structured_output.name == "John"
    assert response.structured_output.age == 30
    assert set(response.structured_output.hobbies) == {
        "reading",
        "hiking",
        "photography",
    }

def test_mistral_structured_lm():
    lm = LM(
        model_name="mistral-small-latest",
        formatting_model_name="mistral-small-latest",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that extracts structured information.",
        user_message=TEST_PROMPT,
        response_model=Person,
    )

    assert isinstance(response.structured_output, Person)
    assert response.structured_output.name == "John"
    assert response.structured_output.age == 30
    assert set(response.structured_output.hobbies) == {
        "reading",
        "hiking",
        "photography",
    }


if __name__ == "__main__":
    test_openai_structured_lm()
