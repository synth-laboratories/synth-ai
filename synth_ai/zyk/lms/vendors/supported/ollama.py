from openai import OpenAI, AsyncOpenAI
from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard


class OllamaAPI(OpenAIStandard):
    def __init__(self):
        self.sync_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        self.async_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )