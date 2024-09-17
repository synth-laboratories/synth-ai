from together import Together, AsyncTogether
import os
from zyk.src.zyk.lms.vendors.base import VendorBase
from zyk.src.zyk.lms.vendors.openai_standard import OpenAIStandard

class TogetherAPI(OpenAIStandard):
    def __init__(self):
        self.sync_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))