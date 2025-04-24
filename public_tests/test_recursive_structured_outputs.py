import asyncio
import unittest
from typing import List

from pydantic import BaseModel

from synth_ai.zyk.lms.core.main import LM


# Define example structured output models
class SimpleResponse(BaseModel):
    message: str
    confidence: float


class ComplexResponse(BaseModel):
    title: str
    tags: List[str]
    content: str


class NestedResponse(BaseModel):
    main_category: str
    subcategories: List[str]
    details: SimpleResponse


# Define nested structured output models
class Address(BaseModel):
    street: str
    city: str
    country: str


class PersonalInfo(BaseModel):
    name: str
    age: int
    address: Address


class WorkInfo(BaseModel):
    company: str
    position: str
    years_experience: int


class NestedPersonResponse(BaseModel):
    personal: PersonalInfo
    work: WorkInfo
    skills: List[str]


class ProjectDetails(BaseModel):
    name: str
    description: str
    technologies: List[str]


class NestedPortfolioResponse(BaseModel):
    developer: PersonalInfo
    projects: List[ProjectDetails]
    total_experience: int


class NestedCompanyResponse(BaseModel):
    name: str
    founded: int
    headquarters: Address
    employees: List[PersonalInfo]
    main_products: List[str]


class TestLMStructuredOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the LM once for all tests
        cls.lm = LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
            max_retries="Few",
            structured_output_mode="forced_json",
        )

    def test_sync_simple_response(self):
        result = self.lm.respond_sync(
            system_message="You are a helpful assistant.",
            user_message="Give me a short greeting and your confidence level.",
            response_model=SimpleResponse,
        )
        self.assertIsInstance(result.structured_output, SimpleResponse)
        self.assertIsInstance(result.structured_output.message, str)
        self.assertIsInstance(result.structured_output.confidence, float)
        self.assertGreaterEqual(result.structured_output.confidence, 0)
        self.assertLessEqual(result.structured_output.confidence, 1)

    def test_sync_complex_response(self):
        result = self.lm.respond_sync(
            system_message="You are a content creator.",
            user_message="Create a short blog post about AI.",
            response_model=ComplexResponse,
        )
        self.assertIsInstance(result.structured_output, ComplexResponse)
        self.assertIsInstance(result.structured_output.title, str)
        self.assertIsInstance(result.structured_output.tags, list)
        self.assertIsInstance(result.structured_output.content, str)

    async def async_nested_response(self):
        result = await self.lm.respond_async(
            system_message="You are a categorization expert.",
            user_message="Categorize 'Python' and provide a brief description.",
            response_model=NestedResponse,
        )
        self.assertIsInstance(result.structured_output, NestedResponse)
        self.assertIsInstance(result.structured_output.main_category, str)
        self.assertIsInstance(result.structured_output.subcategories, list)
        self.assertIsInstance(result.structured_output.details, SimpleResponse)

    def test_async_nested_response(self):
        asyncio.run(self.async_nested_response())


class TestLMNestedStructuredOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the LM once for all tests
        cls.lm = LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
            max_retries="Few",
            structured_output_mode="forced_json",
        )

    def test_sync_nested_person_response(self):
        result = self.lm.respond_sync(
            system_message="You are an HR assistant.",
            user_message="Provide detailed information about a fictional employee named John Doe.",
            response_model=NestedPersonResponse,
        )
        self.assertIsInstance(result.structured_output, NestedPersonResponse)
        self.assertIsInstance(result.structured_output.personal, PersonalInfo)
        self.assertIsInstance(result.structured_output.personal.address, Address)
        self.assertIsInstance(result.structured_output.work, WorkInfo)
        self.assertIsInstance(result.structured_output.skills, list)

    def test_sync_nested_portfolio_response(self):
        result = self.lm.respond_sync(
            system_message="You are a portfolio manager.",
            user_message="Create a portfolio for a fictional software developer with multiple projects.",
            response_model=NestedPortfolioResponse,
        )
        self.assertIsInstance(result.structured_output, NestedPortfolioResponse)
        self.assertIsInstance(result.structured_output.developer, PersonalInfo)
        self.assertIsInstance(result.structured_output.developer.address, Address)
        self.assertIsInstance(result.structured_output.projects, list)
        for project in result.structured_output.projects:
            self.assertIsInstance(project, ProjectDetails)
        self.assertIsInstance(result.structured_output.total_experience, int)

    async def async_nested_company_response(self):
        result = await self.lm.respond_async(
            system_message="You are a company information specialist.",
            user_message="Provide detailed information about a fictional tech company.",
            response_model=NestedCompanyResponse,
        )
        self.assertIsInstance(result.structured_output, NestedCompanyResponse)
        self.assertIsInstance(result.structured_output.headquarters, Address)
        self.assertIsInstance(result.structured_output.employees, list)
        for employee in result.structured_output.employees:
            self.assertIsInstance(employee, PersonalInfo)
            self.assertIsInstance(employee.address, Address)
        self.assertIsInstance(result.structured_output.main_products, list)

    def test_async_nested_company_response(self):
        asyncio.run(self.async_nested_company_response())


if __name__ == "__main__":
    unittest.main()
