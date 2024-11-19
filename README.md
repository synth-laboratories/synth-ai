# jaZYK

Simple LM api wrappers for production

Supports:
- OpenAI
- Anthropic
- Gemini
- DeepSeek
- Together

Key Features:
- Structured Output logic (with retries)
- Caching (ephemeral in-memory and/or on-disk sqlite)
- Supports images for Anthropic and OpenAI

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

## Name

ZYK is short for "Z you know"

python -m build                   
twine check dist/*
twine upload dist/*