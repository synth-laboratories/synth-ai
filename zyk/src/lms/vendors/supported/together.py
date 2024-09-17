import os

from together import AsyncTogether, Together

from zyk.src.lms.vendors.openai_standard import OpenAIStandard


class TogetherAPI(OpenAIStandard):
    def __init__(self):
        self.sync_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))