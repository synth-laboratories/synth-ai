import os

from dotenv import load_dotenv
from groq import AsyncGroq, Groq

from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard

load_dotenv()


class GroqAPI(OpenAIStandard):
    def __init__(self):
        super().__init__(
            sync_client=Groq(api_key=os.getenv("GROQ_API_KEY")),
            async_client=AsyncGroq(api_key=os.getenv("GROQ_API_KEY")),
        )
