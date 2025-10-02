# ruff: noqa
'''
Synth OSS Integration Module

This module provides integration with Synth's open-source inference and training APIs
from the monorepo learning_v2 service. All APIs are OpenAI-compatible.

Learning V2 APIs available for integration via lm/:
"""

# API Configuration
SYNTH_BACKEND_URL = ""

# Learning V2 Modal Service URLs
LEARNING_V2_URLS = {
    "dev": "https://synth-laboratories-dev--learning-v2-service-fastapi-app.modal.run",
    "prod": "https://synth-laboratories-prod--learning-v2-service-fastapi-app.modal.run", 
    "main": "https://synth-laboratories--learning-v2-service-fastapi-app.modal.run"
}

# ============================================================================
# HEALTH & STATUS APIS
# ============================================================================

HEALTH_APIS = {
    "basic_health": {
        "method": "GET",
        "endpoint": "/health",
        "description": "Basic health check",
        "response": {"status": "healthy"}
    },
    "detailed_health": {
        "method": "GET", 
        "endpoint": "/learning/health",
        "description": "Detailed health check including GPU function availability",
        "response": {"status": "healthy", "components": {...}}
    }
}

# ============================================================================
# FILE MANAGEMENT APIS
# ============================================================================

FILE_MANAGEMENT_APIS = {
    "upload_file": {
        "method": "POST",
        "endpoint": "/files",
        "description": "Upload a file for fine-tuning (JSONL format)",
        "request": "multipart/form-data with 'file' and 'purpose'='fine-tune'",
        "response": {
            "id": "file-abc123",
            "object": "file", 
            "bytes": 1234,
            "created_at": 1638360000,
            "filename": "data.jsonl",
            "purpose": "fine-tune"
        }
    },
    "list_files": {
        "method": "GET",
        "endpoint": "/files",
        "description": "List all uploaded files",
        "params": {"limit": "optional"},
        "response": {"object": "list", "data": ["file_objects"]}
    },
    "get_file": {
        "method": "GET",
        "endpoint": "/files/{file_id}",
        "description": "Get file metadata by ID",
        "response": "Single file object with metadata"
    },
    "delete_file": {
        "method": "DELETE",
        "endpoint": "/files/{file_id}",
        "description": "Delete a file",
        "response": {"id": "file-abc123", "object": "file", "deleted": True}
    },
    "get_file_content": {
        "method": "GET",
        "endpoint": "/files/{file_id}/content",
        "description": "Download raw file content",
        "response": "Raw file content stream"
    }
}

# ============================================================================
# TRAINING/FINE-TUNING APIS  
# ============================================================================

TRAINING_APIS = {
    "create_training_job": {
        "method": "POST",
        "endpoint": "/fine_tuning/jobs",
        "description": "Create a fine-tuning job",
        "request": {
            "model": "Qwen/Qwen3-0.5B",
            "training_file": "file-abc123", 
            "training_type": "sft",  # or "dpo"
            "hyperparameters": {...},
            "suffix": "optional"
        },
        "response": {
            "object": "fine_tuning.job",
            "id": "ftjob-xyz789",
            "model": "...",
            "status": "validating_files",
            "training_file": "file-abc123",
            "hyperparameters": {...}
        }
    },
    "list_training_jobs": {
        "method": "GET",
        "endpoint": "/fine_tuning/jobs", 
        "description": "List all training jobs",
        "response": {"object": "list", "data": ["job_objects"]}
    },
    "get_training_job": {
        "method": "GET",
        "endpoint": "/fine_tuning/jobs/{job_id}",
        "description": "Get training job status",
        "response": {
            "object": "fine_tuning.job",
            "id": "ftjob-xyz789",
            "status": "running",  # or "completed", "failed", "cancelled"
            "fine_tuned_model": "ft:model:suffix"  # when completed
        }
    },
    "cancel_training_job": {
        "method": "POST",
        "endpoint": "/fine_tuning/jobs/{job_id}/cancel",
        "description": "Cancel a running training job",
        "response": {"object": "fine_tuning.job", "id": "...", "status": "cancelled"}
    },
    "get_training_events": {
        "method": "GET", 
        "endpoint": "/fine_tuning/jobs/{job_id}/events",
        "description": "Get training logs/events",
        "response": {
            "object": "list",
            "data": [{
                "object": "fine_tuning.job.event",
                "level": "info",
                "message": "Training started",
                "created_at": 1638360000
            }]
        }
    }
}

# ============================================================================
# INFERENCE APIS
# ============================================================================

INFERENCE_APIS = {
    "chat_completions": {
        "method": "POST",
        "endpoint": "/chat/completions", 
        "description": "OpenAI-compatible chat completions for base and fine-tuned models",
        "request": {
            "model": "Qwen/Qwen3-0.5B",  # or "ft:Qwen/Qwen3-0.5B:suffix"
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1.0,
            "stream": False,  # Set to True for streaming
            "tools": [],  # For tool calling
            "tool_choice": "auto"
        },
        "response": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1638360000,
            "model": "Qwen/Qwen3-0.5B",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                    "tool_calls": []  # If tools were used
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        },
        "streaming": "Server-sent events with data: {...} format when stream=True"
    }
}

# ============================================================================  
# MODEL MANAGEMENT APIS
# ============================================================================

MODEL_APIS = {
    "list_models": {
        "method": "GET",
        "endpoint": "/models",
        "description": "List all available models (base and fine-tuned)",
        "response": {
            "object": "list",
            "data": [{
                "id": "Qwen/Qwen3-0.5B",
                "object": "model", 
                "created": 1638360000,
                "owned_by": "learning_v2"
            }]
        }
    },
    "delete_model": {
        "method": "DELETE",
        "endpoint": "/models/{model_id}",
        "description": "Delete a fine-tuned model",
        "response": {"id": "ft:model:suffix", "object": "model", "deleted": True}
    }
}

# ============================================================================
# SUPPORTED MODELS
# ============================================================================

SUPPORTED_MODELS = {
    "base_models": [
        # Qwen 3 family
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.8B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        # Qwen 2.5 family
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        # OLMo 2 family
        "allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "allenai/OLMo-2-1124-13B-Instruct"
    ],
    "training_types": ["sft", "dpo"],
    "gpu_types": ["A10G", "L40S", "A100", "H100"],
    "features": [
        "Tool calling",
        "Streaming responses", 
        "Fine-tuning",
        "Multi-GPU training",
        "JSONL data format",
        "OpenAI compatibility"
    ]
}

# ============================================================================
# INTEGRATION PLAN – Synth OSS
# ==========================================================================
"""
GPU & Resource Selection
------------------------
Synth OSS decides the GPU based on the `ModelFamily` definition:
• Each `ModelConfig` lists `inference_gpus` and `training_gpus`.
• The API’s `InferenceRouter` calls `_select_gpu_for_model`, which chooses the **first recommended GPU** returned by `get_model_gpu_recommendations` (usually the `default_inference_gpu`).
• By default the server picks the first recommended GPU, **but** we can request
another GPU type via a custom header that the server *can* opt to honor:

    X-GPU-Preference: L40S  # or A10G, A100, H100

The current dev deployment already forwards this header to `InferenceRouter`,
so adding it makes the GPU configurable without breaking existing behaviour.

`InferenceConfig` therefore gets a new optional field:

```python
class InferenceConfig(BaseModel):
    stream: bool = False
    gpu_preference: Optional[str] = None  # "A10G", "L40S", "A100", "H100"
    # ...future knobs (temperature, max_tokens, etc.)
```

LM will include `gpu_preference` as that header when `backend="synth"`. If the
header is omitted or the value is not valid for the chosen model, the server
falls back to its default selection. This keeps the API forward-compatible and
provides explicit GPU control when supported.

Only two parts of synth-ai need to change for Synth OSS inference:

1.  LM() class (synth_ai.lm)
2.  The async respond(...) coroutine on that class

Extend LM with backend="synth"; when selected, issue POST requests to
`${LEARNING_V2_URL}/chat/completions`, supporting both streaming and
non-streaming modes and returning the same dict structure as today.

Everything else (file upload, fine-tuning, model listing) lives in the
`synth_ai.learning` package and does NOT affect LM:

synth_ai/learning/
    ├─ files.py
    ├─ training.py
    ├─ models.py
    ├─ client.py
    └─ types.py

Warm-up flow
~~~~~~~~~~~~
`learning_v2` exposes `POST /warmup/{model_id}` and `GET /warmup/status/{model_id}`
(via the Render proxy).  We can exploit that to reduce first-token latency.

LM API addition:

```python
async def warmup(self, model: str | None = None, gpu_preference: str | None = None) -> dict:
    """Pre-spin the container & load weights for *model* on the requested GPU.
    Returns the JSON response from /warmup.  If *model* is None we warm-up
    `self.model`.
    """
```

Implementation sketch (backend == "synth")
------------------------------------------
1.  Determine `model_id = model or self.model`.
2.  Build headers:
    ```python
    headers = {}
    if gpu_preference:
        headers["X-GPU-Preference"] = gpu_preference
    ```
3.  `POST  f"{url}/warmup/{model_id}"`.
4.  Optionally call `GET /warmup/status/{model_id}` in a loop until
    `status == "ready"` (exponential backoff) – or expose a separate
    `await LM.warmup_status(model)` helper.

The method is a *no-op* for the default (OpenAI) backend so existing code keeps
working.

'''


"""
LEARNING_v2 server-side changes required to honor `X-GPU-Preference`
====================================================================
1.  **Add header extraction in API layer**
    • File: `backend/app/services/learning_v2/modal_service/api_openai_complete.py`
    • In `@app.post("/chat/completions")` add:
        ```python
        gpu_pref = req.headers.get("X-GPU-Preference")
        ```
    • Pass `gpu_pref` to `inference_router.route_completion(...)` and
      `route_streaming_completion(...)`.

2.  **Propagate preference through `InferenceRouter`**
    • Update signatures of `route_completion`, `route_streaming_completion` and
      `_select_gpu_for_model` to accept `gpu_preference: Optional[str] = None`.
    • Inside `_select_gpu_for_model`:
        ```python
        if gpu_preference and gpu_preference in valid_gpu_types_for_model:
            return gpu_preference
        ```
      falling back to current auto-selection logic otherwise.

3.  **Validate against model config**
    • Retrieve `model_cfg.inference_gpus` and refuse unsupported choices with
      `HTTP 400` ("GPU type not supported for this model").

4.  **Expose preference in usage / logging**
    • Include the chosen GPU in the response’s `system_fingerprint` or add a
      field `selected_gpu` so clients can confirm the assignment.
    • Log header + selected GPU for observability.

5.  **(Optional) Accept preference in JSON body**
    • If we want parity with LM’s `InferenceConfig`, extend
      `ChatCompletionRequest` with an optional `gpu_preference` field and treat
      it as a fallback if the header is missing.

6.  **Update automated tests**
    • Modify `tests/dev/learning_v2/test_multi_gpu_inference_v2.py` to set the
      header and assert that the correct GPU function (`*_L40S_generate`, etc.)
      was invoked (can mock `get_gpu_function`).

7.  **Documentation**
    • Add the header to `learning_v2_api_structure.txt` and README so external
      callers know how to request a GPU.

These changes are backwards-compatible: calls without the header behave exactly
as before.
"""

"""
Ran tool

Ran tool

Read file: /Users/joshuapurtell/Documents/GitHub/monorepo/backend/app/services/learning/unified_ft_service/api.py

Ran tool
Read file: /Users/joshuapurtell/Documents/GitHub/monorepo/backend/app/services/learning/unified_ft_service/api.py

Ran tool

Warm-up today:

• `POST /warmup/{model_name}` in `unified_ft_service/api.py` kicks off an async worker that:
  1. Calls `get_model_routing(model_name)` → this falls back to `choose_gpu(model_id, req_gpu=None)`  
     so no GPU can be requested.
  2. Fires the routed Modal function once (“dummy request”) to cache weights.

Because the handler never accepts a GPU argument and never reads `X-GPU-Preference`, the server always warms the model with its default GPU.

What has to change
==================

Server-side (learning_v2)
1. Add an optional header or query field  
   `X-GPU-Preference` **or** `?gpu=A100` to `POST /warmup/{model_id}`.
2. Pass the value into `choose_gpu(model_id, req_gpu)` inside
   `warmup_worker()`.  The helper already validates the choice, so no other
   change is required.
3. Expose the selected GPU in the JSON response for visibility.

Client-side (synth-ai LM)
```python
async def warmup(
    self,
    model: str | None = None,
    gpu_preference: str | None = None,
    wait_until_ready: bool = False,
) -> dict:
    mdl = model or self.model
    headers = {}
    if gpu_preference:
        headers["X-GPU-Preference"] = gpu_preference
    resp = await _client.post(f"{url}/warmup/{mdl}", headers=headers)
    if wait_until_ready:
        while resp.json()["status"] != "warmed":
            await asyncio.sleep(2)
            resp = await _client.get(f"{url}/warmup/status/{mdl}")
    return resp.json()
```

So: **the existing endpoint does not yet support GPU selection; we need to add
the small change above on the `learning_v2` side and then LM.warmup can request
specific GPUs.**
"""
