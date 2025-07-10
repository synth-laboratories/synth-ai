# jaZYK

Simple LM api wrappers for prod

Supports:
- OpenAI
- Anthropic
- Gemini
- Grok (xAI)
- DeepSeek
- Together

Key Features:
- Structured Output logic (with retries)
- Caching (ephemeral in-memory and/or on-disk sqlite)
- Supports images for Anthropic and OpenAI

Add a PR if you want to add another provider!

## Usage
```
from synth_ai.zyk import LM
lm = LM(model_name="gpt-4o-mini", temperature=0.0)
class HelpfulResponse(BaseModel):
    greeting: str
    name: str
print(lm.respond_sync(system_message="You are a helpful assistant", user_message="Hello, how are you?", response_model=HelpfulResponse))
```