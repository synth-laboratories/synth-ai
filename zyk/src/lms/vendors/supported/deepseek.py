import os

from openai import AsyncOpenAI, OpenAI

from zyk.src.lms.vendors.openai_standard import OpenAIStandard


class DeepSeekAPI(OpenAIStandard):
    def __init__(self):
        self.sync_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.async_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )