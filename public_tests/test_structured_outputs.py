import asyncio
import unittest
from typing import List

from pydantic import BaseModel, Field

from synth_ai.zyk.lms.core.main import LM
from synth_ai.zyk import BaseLMResponse


# Define example structured output models
class SimpleResponse(BaseModel):
    message: str
    confidence_between_zero_one: float = Field(
        ..., description="Confidence level between 0 and 1"
    )


class ComplexResponse(BaseModel):
    title: str
    tags: List[str]
    content: str


class NestedResponse(BaseModel):
    main_category: str
    subcategories: List[str]
    details: SimpleResponse


class TestLMStructuredOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize LMs for both forced_json and stringified_json modes
        cls.lm_forced_json = LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
            max_retries="Few",
            structured_output_mode="forced_json",
        )
        cls.lm_stringified_json = LM(
            model_name="gemma3-27b-it",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
            max_retries="Few",
            structured_output_mode="stringified_json",
        )

    def test_sync_simple_response(self):
        for lm in [self.lm_forced_json, self.lm_stringified_json]:
            with self.subTest(
                mode=lm.structured_output_handler.handler.structured_output_mode
            ):
                result = lm.respond_sync(
                    system_message="You are a helpful assistant.",
                    user_message="Give me a short greeting and your confidence level.",
                    response_model=SimpleResponse,
                )
                self.assertIsInstance(result.structured_output, SimpleResponse)
                self.assertIsInstance(result.structured_output.message, str)
                self.assertIsInstance(
                    result.structured_output.confidence_between_zero_one, float
                )
                self.assertGreaterEqual(
                    result.structured_output.confidence_between_zero_one, 0
                )
                self.assertLessEqual(
                    result.structured_output.confidence_between_zero_one, 1
                )

    def test_sync_complex_response(self):
        for lm in [self.lm_forced_json, self.lm_stringified_json]:
            with self.subTest(
                mode=lm.structured_output_handler.handler.structured_output_mode
            ):
                result = lm.respond_sync(
                    system_message="You are a content creator.",
                    user_message="Create a short blog post about AI.",
                    response_model=ComplexResponse,
                )
                self.assertIsInstance(result.structured_output, ComplexResponse)
                self.assertIsInstance(result.structured_output.title, str)
                self.assertIsInstance(result.structured_output.tags, list)
                self.assertIsInstance(result.structured_output.content, str)

    async def async_nested_response(self, lm):
        result = await lm.respond_async(
            system_message="You are a categorization expert.",
            user_message="Categorize 'Python' and provide a brief description.",
            response_model=NestedResponse,
        )
        print("Result:")
        assert not isinstance(result.structured_output, BaseLMResponse), "Structured output must be a Pydantic model or None - got BaseLMResponse"
        self.assertIsInstance(result.structured_output, NestedResponse)
        self.assertIsInstance(result.structured_output.main_category, str)
        self.assertIsInstance(result.structured_output.subcategories, list)
        self.assertIsInstance(result.structured_output.details, SimpleResponse)

    def test_async_nested_response(self):
        for lm in [self.lm_forced_json, self.lm_stringified_json]:  #
            with self.subTest(
                mode=lm.structured_output_handler.handler.structured_output_mode
            ):
                asyncio.run(self.async_nested_response(lm))


if __name__ == "__main__":
    unittest.main()
