# jaZYK

Simple LM api wrappers for prod

Supports:
- OpenAI
- Anthropic
- Gemini
- DeepSeek
- Together
- OpenRouter

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

### OpenRouter Example
```
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_api_key_here"

# Optional: Set app information for OpenRouter
export OPENROUTER_APP_URL="https://your-app-url.com"
export OPENROUTER_APP_TITLE="Your App Name"

# Use OpenRouter with any available model
from synth_ai.zyk import LM
lm = LM(model_name="openrouter/anthropic/claude-3.5-sonnet", temperature=0.7)
response = lm.respond_sync(
    system_message="You are a helpful assistant", 
    user_message="What's the weather like?"
)
print(response)
```