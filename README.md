# jaZYK

Simple LM api wrappers for production

Supports:
- OpenAI
- Anthropic
- Gemini
- DeepSeek
- Together

Add a PR if you want to add another provider!

## Installation
```
uv add zyk
```
or
```
pip install zyk
```

## Usage
```
from zyk import LM
lm = LM(model_name="gpt-4o-mini", temperature=0.0)
class HelpfulResponse(BaseModel):
    greeting: str
    name: str
print(lm.respond_sync(system_message="You are a helpful assistant", user_message="Hello, how are you?", response_model=HelpfulResponse))
```
