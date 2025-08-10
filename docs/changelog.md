# Changelog

## 0.2.4.dev1 - 2025-08-10

### First-Class LLM API Events with LLMCallRecord

#### Overview
Introduced comprehensive system for capturing and storing detailed LLM API interactions through new `LLMCallRecord` abstraction. Provides normalized, provider-agnostic storage of LLM interactions suitable for analytics, debugging, and fine-tuning data extraction.

#### New Features

**LLMCallRecord Abstraction** (`synth_ai/tracing_v3/lm_call_record_abstractions.py`)
- Unified schema normalizing OpenAI, Anthropic, and local model API responses
- Comprehensive capture of request/response details including:
  - Multi-modal content (text, images, audio)
  - Tool/function calls and execution results
  - Detailed token usage including reasoning tokens and cache metrics
  - Streaming chunk reconstruction
  - Request parameters and provider-specific metadata

**Helper Functions** (`synth_ai/tracing_v3/llm_call_record_helpers.py`)
- `create_llm_call_record_from_response()`: Converts vendor responses to normalized format
- `compute_aggregates_from_call_records()`: Aggregates statistics across multiple API calls
- `create_llm_call_record_from_streaming()`: Reconstructs complete records from streaming chunks

**Enhanced Tracing**
- `LMCAISEvent` now includes `call_records: List[LLMCallRecord]` field
- Database schema updated with proper `call_records` column (JSON storage)
- Clean migration path maintaining backward compatibility

**Training Data Extraction**
- New `extract_openai_format_from_call_records()` method in filter scripts
- Direct extraction from structured call_records for fine-tuning datasets
- Preserves full conversation context including tool interactions

#### Architectural Improvements

**Markov Blanket Clarification**
- Renamed `SessionEventMessage` to `SessionEventMarkovBlanketMessage`
- Enhanced documentation explaining system boundaries and information flow
- Clear distinction between:
  - Chat messages (stored in LLMCallRecord)
  - Inter-system messages (SessionEventMarkovBlanketMessage)
- Aligns with Free Energy Principle and Active Inference frameworks

---

## 0.2.2.dev2 - 2025-08-08

### Highlights

- Fine-tuning (SFT) endpoints are now available and documented end-to-end (files → jobs → status)
- Added interactive demo launcher (`uvx synth-ai demo`) with finetuning flow for Qwen 4B (Crafter)
- Demo script streams live polling output during training (status updates visible while running)

### CLI & Demos

- `uvx synth-ai demo` — lists available demos and lets you run them interactively:
  - `examples/finetuning/synth_qwen/run_demo.sh` — rollouts → v3 trace filtering → SFT kickoff, with live polling
  - `examples/evals/run_demo.sh` — quick eval rollouts and trace filtering for dataset prep
- Improved demo UX: training status lines (e.g., `⏳ poll N/20 – status = running`) now stream live in the terminal

### Documentation

- Examples → Walkthrough — synth_qwen (Crafter) with: generate → filter → finetune → run
- CLI Reference section for `uvx synth-ai serve`, `uvx synth-ai traces`, and demo launcher
- Turso v3 tracing guide and filtering guide for SFT JSONL generation

---

## 0.2.2.dev1 - 2025-08-07

### Highlights

- New backend balance APIs and CLI for quick account visibility (USD balance + token/GPU spend windows)
- New CLI utilities and manual: compact, one-off commands with `uvx synth-ai <cmd>` and `man`
- Traces inventory view showing per-DB and per-system counts, plus on-disk size (GB)
- Inference and SFT API routes consolidated and documented for local and Modal deployments

### CLI

- Added `balance`: prints minimal balance in USD and a compact spend table for the last 24h and 7d
  - Flags: `--base-url`, `--api-key`, `--usage`; sources `.env` automatically; guards against Modal URLs for account endpoints
- Added `traces`: lists local trace DBs under `./synth_ai.db/dbs`, shows traces, experiments, last activity, and size (GB), plus aggregated per-system counts
- Added `man`: human-friendly command reference with options, env vars, and examples
- Standardized one-off usage: `uvx synth-ai <command>` (removed legacy interactive `watch`)
- Improved `.env` loading and API key resolution (`SYNTH_BACKEND_API_KEY` → `SYNTH_API_KEY` → `DEFAULT_DEV_API_KEY`)
- Existing commands remain available: `experiments`, `experiment <id>`, `usage [--model]`, `status`, `calc`, and `env` (list/register/unregister)

### Demo

- Local end-to-end demo: start backend (`uv run uvicorn app.services.main:app --reload --port 8000`), set `SYNTH_BACKEND_API_KEY`, then:
  - `uvx . balance` → shows USD balance + 24h/7d spend
  - `uvx . calc 'gpt-4o' 1000000 500000` → cost calculation
  - `uvx . usage --model gpt-4o` → recent model-specific usage
- Modal demo: deploy Modal app (`modal deploy app.modal.app`) to get URLs → `uvx . balance`
- Added `--base-url <modal_url>` handling for hosted backends

### Backend & APIs

- Exposed `/balance`, `/balance/usage`, and `/balance/account` routes for cost tracking
- Fixed Modal deployments (renamed `app.modal.app` → `app_modal.py`, handling import issues with shared.py)
- Routing now strips `/sft/` prefix in handler delegations; simplified path construction
- Improved error handling with structured responses and clear messages for API key issues
- Consolidated `/status` endpoint returns `{"status": "healthy"}` with new balance info options

### Infrastructure

- Fixed modal deployments with proper relative imports between `app_modal.py` and `shared.py`
- Ensured `httpx` availability in Modal containers
- SFT handlers now properly strip route prefix for internal dispatch
- Database path standardization: `./synth_ai.db/dbs/` for tracing DBs
- Improved environment variable cascading with `.env` support

### Documentation

- New CLI section in docs with `balance`, `traces`, and `man` commands
- Updated deployment guide for Modal with corrected file names and imports
- Added troubleshooting section for common API key and URL configuration issues

---

## Previous Releases

See git history for earlier versions.