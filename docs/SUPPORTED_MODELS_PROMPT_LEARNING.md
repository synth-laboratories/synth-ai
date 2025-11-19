# Supported Models for Prompt Learning

## Overview
Prompt learning supports three types of models:
1. **Policy Models** - The model that executes the task (used by both GEPA and MIPRO)
2. **Mutation Models** (GEPA only) - The model used to generate prompt mutations
3. **Meta Models** (MIPRO only) - The model used to generate instruction proposals

All three types use the **same validation rules** - they must be from supported providers (OpenAI, Groq, or Google) and must be in the supported model list.

---

## Supported Providers

### 1. OpenAI
**Supported Models:**
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`

**Explicitly REJECTED:**
- `gpt-5-pro` (too expensive: $15/$120 per 1M tokens)

**Examples from configs:**
- Policy: `gpt-4.1-mini` (used in MIPRO configs)
- Meta model: `gpt-4o-mini`, `gpt-4.1-mini` (most common in MIPRO configs)

### 2. Groq
**Supported Models:**
- `gpt-oss-Xb` pattern (e.g., `gpt-oss-20b`, `openai/gpt-oss-120b`)
- `llama-3.3-70b` and variants (e.g., `llama-3.3-70b-versatile`)
- `qwen-32b`, `qwen3-32b`, `groq/qwen3-32b`

**Examples from configs:**
- Policy: `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`
- Mutation: `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, `llama3-groq-70b-8192-tool-use-preview`

### 3. Google/Gemini
**Supported Models:**
- `gemini-2.5-pro`
- `gemini-2.5-pro-gt200k`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

**Examples from configs:**
- Policy: `gemini-2.5-flash-lite` (used in Gemini MIPRO config)

---

## Usage by Algorithm

### GEPA (Genetic Evolutionary Prompt Optimization)

**Policy Model** (`prompt_learning.policy.model`):
- Can be any supported model from OpenAI, Groq, or Google
- Examples: `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `gemini-2.5-flash-lite`

**Mutation Model** (`prompt_learning.gepa.mutation.llm_model`):
- Used to generate prompt mutations/variations
- Must be a supported model from OpenAI, Groq, or Google
- Common choices:
  - `openai/gpt-oss-120b` (Groq provider) - most common
  - `openai/gpt-oss-20b` (Groq provider)
  - `llama-3.3-70b-versatile` (Groq provider)
  - `llama3-groq-70b-8192-tool-use-preview` (Groq provider)

### MIPRO (Meta-learning for Instruction PROposal)

**Policy Model** (`prompt_learning.policy.model`):
- Can be any supported model from OpenAI, Groq, or Google
- Examples: `openai/gpt-oss-20b`, `gpt-4.1-mini`, `gemini-2.5-flash-lite`

**Meta Model** (`prompt_learning.mipro.meta_model`):
- Used to generate instruction proposals
- Must be a supported model from OpenAI, Groq, or Google
- Common choices:
  - `gpt-4o-mini` (OpenAI) - most common default
  - `gpt-4.1-mini` (OpenAI) - used in some configs
  - `gpt-4o` (OpenAI) - used in some configs

---

## Model Restrictions

### Nano Models
**Nano models (e.g., `gpt-4.1-nano`, `gpt-5-nano`) are:**
- ✅ **ALLOWED** for policy models (`prompt_learning.policy.model`)
- ❌ **REJECTED** for mutation models (`prompt_learning.gepa.mutation.llm_model`)
- ❌ **REJECTED** for meta models (`prompt_learning.mipro.meta_model`)

**Reason:** Nano models are too small for generating high-quality prompt variations and instruction proposals. They are fine for executing tasks (policy), but not for creative generation tasks (mutation/proposal).

### gpt-5-pro
**Explicitly REJECTED** for all model types (too expensive: $15/$120 per 1M tokens)

---

## Model Validation

All models are validated **before sending requests to the backend**:
- Unsupported models are rejected with clear error messages
- Provider must match the model (e.g., OpenAI model with OpenAI provider)
- Models can be specified with or without provider prefix (e.g., `gpt-4o-mini` or `openai/gpt-4o-mini`)

**Validation happens in:**
- SDK: `synth_ai/api/train/validators.py` - `validate_prompt_learning_config()`
- Called before any backend requests in `build_prompt_learning_payload()`

---

## Common Patterns from Configs

### GEPA Mutation Models:
```toml
[prompt_learning.gepa.mutation]
llm_model = "openai/gpt-oss-120b"  # Most common
llm_provider = "groq"
# OR
llm_model = "llama-3.3-70b-versatile"
llm_provider = "groq"
```

### MIPRO Meta Models:
```toml
[prompt_learning.mipro]
meta_model = "gpt-4o-mini"  # Most common default
meta_model_provider = "openai"
# OR
meta_model = "gpt-4.1-mini"
meta_model_provider = "openai"
```

---

## Notes

1. **Same validation for all model types**: Policy, mutation, and meta models all use the same validation rules
2. **Provider prefix optional**: Models can be specified as `gpt-4o-mini` or `openai/gpt-4o-mini`
3. **Case-insensitive**: Model names are validated case-insensitively
4. **gpt-5-pro explicitly rejected**: Too expensive for prompt learning workloads
5. **No other providers**: Only OpenAI, Groq, and Google are supported

