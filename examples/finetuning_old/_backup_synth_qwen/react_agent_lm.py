#!/usr/bin/env python3
"""
Test script to run ReAct agents against Crafter environment using LM class with Synth backend.
This demonstrates using the LM class with Synth models through native integration.

This version uses the new tracing_v3 system with async Turso/SQLite backend.
"""

import argparse
import asyncio
import contextlib
from contextlib import asynccontextmanager
import glob
import itertools
import json
import logging
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import toml
import yaml
from httpx import AsyncClient
from tqdm import tqdm

# Disable httpx logging immediately
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


# Configure logging to suppress noisy third-party logs when in quiet mode
def setup_logging(quiet_mode: bool = False):
    """Setup logging configuration."""
    if quiet_mode:
        # Suppress most third-party logging in quiet mode
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("synth_ai.tracing_v3").setLevel(logging.ERROR)
        logging.getLogger("synth_ai.tracing_v3.turso").setLevel(logging.ERROR)
        logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
        logging.getLogger("aiosqlite").setLevel(logging.ERROR)
        # Suppress httpcore as well (used by httpx)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
    else:
        # Normal logging levels
        logging.getLogger("httpx").setLevel(logging.ERROR)  # Always suppress httpx logs
        logging.getLogger("synth_ai.tracing_v3").setLevel(logging.INFO)


# Set default logging to avoid noisy logs during import
setup_logging(quiet_mode=True)

# Setup environment
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Disable v1 logging to see v3 tracing clearly
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["SYNTH_LOGGING"] = "false"

from synth_ai.lm.config import SynthConfig  # noqa: E402

# Import Synth warmup utilities
from synth_ai.lm.warmup import warmup_synth_model  # noqa: E402

# Import session tracer for v3 tracing
from synth_ai.tracing_v3 import SessionTracer  # noqa: E402
from synth_ai.tracing_v3.abstractions import (  # noqa: E402
    EnvironmentEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    TimeRecord,
)

# Import Crafter hooks for v3
from synth_ai.tracing_v3.hooks import HookManager  # noqa: E402
from synth_ai.tracing_v3.turso.daemon import SqldDaemon  # noqa: E402

# create_experiment_context will be defined as a helper function below
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager  # noqa: E402

# Create a custom hook manager without default print statements
QUIET_HOOKS = HookManager()

# Import LM components (v3 version if available)
try:
    from synth_ai.lm.core.main_v3 import LM  # noqa: E402
except ImportError:
    from synth_ai.lm.core.main_v2 import LM  # noqa: E402

# Configuration constants
HTTP_TIMEOUT = (
    30.0  # Increased from 10.0 for better handling of concurrent load and LM response times
)
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# Use the backend
@asynccontextmanager
async def _noop_async_context():
    yield


async def create_experiment_context(
    db_manager: AsyncSQLTraceManager, experiment_name: str, description: str
) -> dict[str, Any]:
    """Create an experiment context for v3 tracing."""
    experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
    await db_manager.create_experiment(
        experiment_id=experiment_id, name=experiment_name, description=description, configuration={}
    )
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "description": description,
    }


def cleanup_old_files():
    """Clean up old trace files and result files to keep directory clean."""
    # Remove old JSON result files (keep only the latest 5)
    result_files = glob.glob("crafter_lm_synth_results_*.json")
    if len(result_files) > 5:
        # Sort by modification time and keep only the latest 5
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for old_file in result_files[5:]:
            try:
                os.remove(old_file)
                print(f"🗑️  Cleaned up old result file: {old_file}")
            except OSError:
                pass


def _load_env_from_monorepo() -> dict:
    """Load environment variables from monorepo/.env.local if present."""
    env_file = (
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "monorepo/.env.local"
    )
    env_vars = {}

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value

    return env_vars


def _load_testing_yaml_api_key() -> str | None:
    """Load SYNTH_API_KEY from monorepo/tests/prod/testing_info.yaml if present."""
    # First try the new env vars from monorepo/.env.local
    env_vars = _load_env_from_monorepo()

    # Try production key first, then test key
    if "SYNTH_API_KEY_PROD" in env_vars:
        return env_vars["SYNTH_API_KEY_PROD"]
    elif "SYNTH_API_KEY_TEST" in env_vars:
        return env_vars["SYNTH_API_KEY_TEST"]

    # Fallback to the old YAML method
    yaml_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        / "monorepo/tests/prod/testing_info.yaml"
    )
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            return data.get("SYNTH_API_KEY")
    return None


def setup_synth_environment():
    """Setup environment variables for Synth/Modal endpoints.

    Resolution order for the base URL:
    1. Explicit environment variables (SYNTH_BASE_URL or MODAL_BASE_URL)
    2. PROD_API_URL env var used in production integration tests
    3. Hard-coded production constant (https://agent-learning.onrender.com)

    The API key is resolved from the matching *_API_KEY env vars or, if not
    present, from the shared testing_info.yaml used by the prod tests.
    """
    # Load environment variables from monorepo/.env.local
    env_vars = _load_env_from_monorepo()

    synth_base_url = (
        os.getenv("SYNTH_BASE_URL")
        or os.getenv("MODAL_BASE_URL")
        or os.getenv("PROD_API_URL")
        or env_vars.get("SYNTH_BASE_URL_PROD")  # Use production URL from .env.local
        or "https://agent-learning.onrender.com/api"
    )

    synth_api_key = os.getenv("SYNTH_API_KEY") or _load_testing_yaml_api_key()

    # # --- Validate API key format ---
    # if synth_api_key:
    #     VALID_PREFIXES = ("sk-", "sk_live_", "sk_test_")
    #     if not any(synth_api_key.startswith(p) for p in VALID_PREFIXES):
    #         truncated = synth_api_key[:8] if len(synth_api_key) >= 8 else synth_api_key
    #         expected_formats = " or ".join(VALID_PREFIXES)
    #         raise ValueError(
    #             f"Invalid API key format. Expected prefix {expected_formats}. Provided key begins with '{truncated}'."
    #         )
    # else:
    #     raise ValueError(
    #         "SYNTH_API_KEY or MODAL_API_KEY must be provided via environment variables or testing_info.yaml"
    #     )

    # Ensure trailing /v1 for OpenAI-compatible endpoints
    if not synth_base_url.endswith("/v1"):
        synth_base_url = synth_base_url.rstrip("/") + "/v1"
    synth_base_url = synth_base_url.rstrip("/")

    # Propagate to OpenAI SDK env vars expected by LM class
    os.environ["OPENAI_API_BASE"] = synth_base_url
    os.environ["OPENAI_BASE_URL"] = synth_base_url
    os.environ["OPENAI_API_KEY"] = synth_api_key

    return synth_base_url, synth_api_key


async def retry_http_request(client: AsyncClient, method: str, url: str, **kwargs) -> Any:
    """Retry HTTP requests with exponential backoff and jitter."""
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = min(RETRY_DELAY * (2 ** (attempt - 1)), RETRY_DELAY * 2)  # Use RETRY_DELAY
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter
                await asyncio.sleep(total_delay)

            response = await client.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)

            if response.status_code < 500:
                return response

            last_exception = Exception(f"HTTP {response.status_code}: {response.text}")

        except httpx.ReadError as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                read_error_delay = min(1.0 * (2**attempt), 5.0)
                await asyncio.sleep(read_error_delay)
        except Exception as e:
            last_exception = e

    print(
        f"    ❌ HTTP request failed after {MAX_RETRIES} attempts: {type(last_exception).__name__}: {str(last_exception)[:200]}"
    )
    raise last_exception


def create_message(
    content: Any, message_type: str, origin_system_id: Any, turn: int
) -> SessionEventMarkovBlanketMessage:
    """Create a message with origin system ID embedded in content."""
    # Map custom message types to valid v3 message types
    type_mapping = {
        "observation": "system",  # Map observation to system message
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "tool_use": "tool_use",
        "tool_result": "tool_result",
    }

    return SessionEventMarkovBlanketMessage(
        content=json.dumps({"origin_system_id": str(origin_system_id), "payload": content}),
        message_type=type_mapping.get(message_type, "system"),  # Default to system
        time_record=TimeRecord(event_time=time.time(), message_time=turn),
    )


def compress_observation_for_trace(obs: dict[str, Any]) -> dict[str, Any]:
    """Compress observation for trace storage to avoid huge trace files."""
    compressed = obs.copy()

    # Compress semantic map if present
    if "semantic_map" in compressed:
        del compressed["semantic_map"]

    # Compress other large fields
    if "rgb" in compressed:
        del compressed["rgb"]

    return compressed


def format_semantic_map_view_v2(obs: dict[str, Any], view_size: int = 7) -> str:
    """Format a semantic map view around the player with normal names using real Crafter mapping."""
    # Get semantic map
    semantic_map = obs.get("semantic_map")
    if semantic_map is None:
        return "No semantic map available"

    # Convert to numpy array if needed
    sem_arr = np.asarray(semantic_map)
    if sem_arr.ndim == 1:
        # Assuming square map, reshape
        size = int(np.sqrt(sem_arr.size))
        sem_arr = sem_arr.reshape(size, size)

    # Get player position
    player_pos = obs.get("player_position", [sem_arr.shape[0] // 2, sem_arr.shape[1] // 2])
    px, py = int(player_pos[0]), int(player_pos[1])

    # Get real crafter semantic mapping directly from crafter library
    import crafter

    dummyenv = crafter.Env()
    try:
        max_id = (
            max(max(dummyenv._world._mat_ids.values()), max(dummyenv._sem_view._obj_ids.values()))
            + 1
        )
        id_to_item = ["void"] * max_id
        for name, ind in itertools.chain(
            dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()
        ):
            clean = (
                name.__name__
                if hasattr(name, "__name__")
                else (str(name) if name is not None else "none")
            )
            id_to_item[ind] = clean.lower()
    finally:
        with contextlib.suppress(AttributeError, Exception):
            dummyenv.close()

    # Create view
    half = view_size // 2
    lines = []
    visible_items = set()

    for dy in range(-half, half + 1):
        row = []
        for dx in range(-half, half + 1):
            x, y = px + dx, py + dy

            if dx == 0 and dy == 0:
                row.append("you")  # Player
            elif 0 <= x < sem_arr.shape[0] and 0 <= y < sem_arr.shape[1]:
                val = int(sem_arr[x, y])
                # Use the real crafter mapping
                item_name = id_to_item[val] if val < len(id_to_item) else f"unknown_{val}"
                row.append(item_name)
                if item_name not in ["grass", "you", "void"]:
                    visible_items.add(item_name)
            else:
                row.append("void")  # Out of bounds

        lines.append(" ".join(row))

    # Add legend of visible items
    legend = (
        f"Visible items: {', '.join(sorted(visible_items))}"
        if visible_items
        else "No special items visible (mostly grass)"
    )

    return "\n".join(lines) + "\n" + legend


def get_openai_tools():
    """Get OpenAI-compatible tool definitions for Synth models."""
    return [
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform actions in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of actions to perform in sequence (e.g., ['move_right', 'move_right', 'do']). Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning for these actions",
                        },
                    },
                    "required": ["actions", "reasoning"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "End the episode when finished or no progress can be made.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for termination"}
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


# --- Configuration Class ---
class CrafterConfig:
    """Configuration for Crafter evaluation with Synth backend."""

    def __init__(self, config_path: str | None = None):
        # Default values
        self.model_name: str | None = None
        self.num_instances = 1
        self.max_turns = 2
        self.difficulty = "easy"
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.save_traces = True
        self.save_detailed_results = True
        self.verbose = False
        self.quiet = False  # Add quiet mode support
        self.analyze_traces = False

        # V3 tracing settings
        self.enable_v3_tracing = True
        # Standardize to a single shared v3 DB by default; allow env override
        self.v3_trace_dir = os.getenv("SYNTH_TRACES_ROOT", "./traces/v3")
        # Use shared DB path unless explicitly overridden via env or config
        self.turso_db_path = os.getenv(
            "SQLD_DB_PATH", os.path.join(self.v3_trace_dir, "synth_ai.db")
        )
        self.start_sqld_daemon = True  # Whether to start sqld daemon
        self.auto_cleanup = True  # Clean up old files automatically

        # Synth-specific settings
        self.warmup_model = True
        self.warmup_max_attempts = 30
        self.warmup_timeout = 60.0  # Default timeout in seconds
        self.use_synth_backend = True  # Flag to indicate Synth backend

        # Load from TOML if provided
        if config_path and os.path.exists(config_path):
            self.load_from_toml(config_path)

    def load_from_toml(self, config_path: str):
        """Load configuration from TOML file."""
        config = toml.load(config_path)

        eval_config = config.get("eval", {})
        self.model_name = eval_config.get("model_name", self.model_name)
        self.num_instances = eval_config.get("episodes", self.num_instances)
        self.max_turns = eval_config.get("max_steps", self.max_turns)
        self.difficulty = eval_config.get("difficulty", self.difficulty)
        self.seed = eval_config.get("seed", self.seed)

        service_config = config.get("service", {})
        self.service_base_url = service_config.get("base_url", self.service_base_url)
        self.service_timeout = service_config.get("timeout", self.service_timeout)

        output_config = config.get("output", {})
        self.save_traces = output_config.get("save_traces", self.save_traces)
        self.save_detailed_results = output_config.get(
            "save_detailed_results", self.save_detailed_results
        )

        # V3 tracing config
        tracing_config = config.get("tracing_v3", {})
        self.enable_v3_tracing = tracing_config.get("enabled", self.enable_v3_tracing)
        self.v3_trace_dir = tracing_config.get("trace_dir", self.v3_trace_dir)
        self.turso_db_path = tracing_config.get("db_path", self.turso_db_path)
        self.start_sqld_daemon = tracing_config.get("start_daemon", self.start_sqld_daemon)
        self.auto_cleanup = tracing_config.get("auto_cleanup", self.auto_cleanup)

        # Synth config
        synth_config = config.get("synth", {})
        self.warmup_model = synth_config.get("warmup_model", self.warmup_model)
        self.warmup_max_attempts = synth_config.get("warmup_max_attempts", self.warmup_max_attempts)
        self.warmup_timeout = synth_config.get("warmup_timeout", self.warmup_timeout)
        self.use_synth_backend = synth_config.get("use_synth_backend", self.use_synth_backend)


# --- Base ReAct Agent using LM with Synth ---
class BaseReActAgentWithLMSynth:
    """Base ReAct agent using LM class configured for Synth backend."""

    def __init__(
        self,
        model_name: str,
        max_turns: int = 20,
        verbose: bool = False,
        tracer: SessionTracer | None = None,
        episode_id: int = 0,
        quiet: bool = False,
        model_params: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.quiet = quiet
        self.history = []
        self.system_name = "base-react-agent-lm-synth"
        self.tools = get_openai_tools()
        self.tracer = tracer
        self.system_id = f"{self.system_name}_{uuid.uuid4()}"
        self.episode_id = episode_id

        # Default model parameters
        default_model_params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "tool_choice": "auto",
        }

        # Merge user-provided parameters with defaults
        self.model_params = {**default_model_params, **(model_params or {})}

        # Setup Synth environment variables
        setup_synth_environment()

        # Create LM instance with synth provider and configurable parameters
        self.lm = LM(
            model_name=model_name,
            formatting_model_name=model_name,
            temperature=self.model_params["temperature"],
            synth_logging=False,  # Disable v1 tracing
            provider="synth",  # Use synth provider
            session_tracer=tracer,
            system_id=self.system_id,
            enable_v3_tracing=True,
            # Pass additional model parameters
            max_tokens=self.model_params["max_tokens"],
            top_p=self.model_params["top_p"],
            frequency_penalty=self.model_params["frequency_penalty"],
            presence_penalty=self.model_params["presence_penalty"],
            # Qwen3 think mode (propagated by vendor to chat_template_kwargs)
            enable_thinking=self.model_params.get("enable_thinking"),
            # Forward arbitrary extra_body to vendor for features like
            # stop_after_tool_calls. The runner sets this to 1.
            extra_body=self.model_params.get("extra_body"),
        )

        # Agent state tracking
        self.agent_state = {
            "message_history": [],
            "steps_taken": 0,
            "steps_remaining": max_turns,
            "total_tokens_used": 0,
            "tool_calls_made": 0,
            "current_turn": 0,
            "last_failure": None,  # Track last failure for prompting
            "recent_tool_calls": [],
        }

    async def decide(self, obs: str, system_message: str, turn: int) -> dict[str, Any]:
        """Get agent decision based on observation using LM class with Synth backend."""
        # Update agent state
        self.agent_state["current_turn"] = turn
        self.agent_state["steps_taken"] = turn
        self.agent_state["steps_remaining"] = self.max_turns - turn

        # Include last 3 tool calls (reasoning and actions) to provide short action history
        recent_calls = self.agent_state.get("recent_tool_calls", [])
        recent_tail = recent_calls[-3:] if isinstance(recent_calls, list) else []
        if recent_tail:
            lines = ["\nRecent tool calls (last 3):"]
            for entry in recent_tail:
                tnum = entry.get("turn")
                name = entry.get("name")
                reasoning = entry.get("reasoning")
                actions = entry.get("actions")
                actions_str = ", ".join(actions) if isinstance(actions, list) else ""
                lines.append(
                    f"- Turn {tnum}: {name} — reasoning: {reasoning}; actions: {actions_str}"
                )
            obs_with_history = f"{obs}\n" + "\n".join(lines)
        else:
            obs_with_history = obs

        # Create conversation context with unique episode ID to prevent caching
        context = (
            f"Episode {self.episode_id} - Turn {turn + 1}/{self.max_turns}\n\n{obs_with_history}"
        )

        # Build messages in OpenAI format for tools
        # Augment the system message if the previous turn failed to produce a tool call
        local_system_message = system_message
        last_failure = self.agent_state.get("last_failure")
        if last_failure:
            local_system_message = (
                f"{system_message}\n\nIMPORTANT: In the previous turn, no valid tool call was returned. "
                f"Error: {last_failure}. You MUST respond with a single function tool call in the OpenAI tools format."
            )
        messages = [
            {"role": "system", "content": local_system_message},
            {"role": "user", "content": context},
        ]

        # Add to message history
        self.agent_state["message_history"].extend(messages)

        # Truncate history if too long
        max_history_length = 20
        if len(self.agent_state["message_history"]) > max_history_length:
            self.agent_state["message_history"] = [
                self.agent_state["message_history"][0]
            ] + self.agent_state["message_history"][-(max_history_length - 1) :]

        try:
            llm_start = time.time()

            # Optionally print full prompt on final turn when verbose
            if self.verbose and turn == self.max_turns - 1:
                print("\n🔍 FINAL TURN PROMPT:")
                print("=" * 80)
                print(f"System: {local_system_message[:200]}...")
                print(f"\nUser message:\n{context}")
                print("=" * 80)

            # Debug: Print request info only when verbose
            if self.verbose:
                print(f"\n🔍 DEBUG: LM call details (turn {turn})")
                print(f"   Model: {self.model_name}")
                print("   Provider: synth")
                print(f"   Messages: {len(messages)} messages")
                print(f"   Tools: {len(self.tools) if self.tools else 0} tools")
                if self.tools:
                    print(
                        f"   Tool 0 name: {self.tools[0].get('function', {}).get('name', 'unknown')}"
                    )
                    print(f"   Tools structure: {json.dumps(self.tools[0], indent=4)[:300]}...")

            # Call LM with turn number for v3 tracing
            # The LM class should handle Synth routing internally
            if self.verbose:
                print(
                    f"🔍 DEBUG: LM sampling params => max_tokens={self.model_params.get('max_tokens')} temp={self.model_params.get('temperature')} top_p={self.model_params.get('top_p')} tool_choice={self.model_params.get('tool_choice')}"
                )

            # Optional full input logging (system, user, tools). Enable with CRAFTER_LOG_FULL_INPUTS=1
            _log_full_inputs = os.getenv("CRAFTER_LOG_FULL_INPUTS", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            # if _log_full_inputs:
            #     print("\n" + "=" * 80)
            #     print(f"FULL LM INPUT (turn {turn})")
            #     print("-" * 80)
            #     print("System message:\n" + local_system_message)
            #     print("\nUser message:\n" + context)
            #     print("\nMessages JSON:")
            #     print(json.dumps(messages, indent=2))
            #     print("\nTools definition:")
            #     print(json.dumps(self.tools, indent=2))
            #     print("\nSampling/tool params:")
            #     print(
            #         json.dumps(
            #             {
            #                 "tool_choice": self.model_params.get("tool_choice"),
            #                 "extra_body": self.model_params.get("extra_body"),
            #                 "temperature": self.model_params.get("temperature"),
            #                 "max_tokens": self.model_params.get("max_tokens"),
            #                 "top_p": self.model_params.get("top_p"),
            #                 "frequency_penalty": self.model_params.get("frequency_penalty"),
            #                 "presence_penalty": self.model_params.get("presence_penalty"),
            #             },
            #             indent=2,
            #         )
            #     )
            #     print("=" * 80)

            response = await self.lm.respond_async(
                messages=messages,
                turn_number=turn,
                # Pass tools in the format expected by LM class
                tools=self.tools,
                max_tokens=self.model_params["max_tokens"],
                tool_choice=self.model_params.get("tool_choice", "auto"),
                # Pass extra_body per call to ensure backend receives stop_after_tool_calls
                extra_body=self.model_params.get("extra_body"),
            )

            llm_end = time.time()

            # Minimal output: show only tool_call presence, number of actions, and tokens
            completion_tokens = None
            prompt_tokens = None
            toks_per_sec = None
            if hasattr(response, "usage") and isinstance(response.usage, dict):
                completion_tokens = response.usage.get("completion_tokens")
                prompt_tokens = response.usage.get("prompt_tokens")
            # Compute tokens/sec if we have duration and completion tokens
            try:
                if completion_tokens is not None:
                    duration_s = max(1e-6, (llm_end - llm_start))
                    toks_per_sec = round(float(completion_tokens) / duration_s, 2)
            except Exception:
                toks_per_sec = None

            # Parse the response to extract tool calls
            raw_response = response.raw_response
            decision: dict[str, Any]

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_call = response.tool_calls[0]
                parsed_decision = None
                fn = tool_call.get("function") if isinstance(tool_call, dict) else None
                if isinstance(fn, dict) and ("name" in fn):
                    name = fn.get("name", "interact")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        import json as _json

                        args = (
                            _json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                        )
                        if isinstance(args, dict):
                            parsed_decision = {"name": name, "parameters": args}
                    except Exception as _e:
                        parsed_decision = {"name": name, "parameters": {"arguments": args_raw}}
                if (
                    not parsed_decision
                    and isinstance(tool_call, dict)
                    and ("name" in tool_call or "parameters" in tool_call)
                ):
                    parsed_decision = {
                        "name": tool_call.get("name", "interact"),
                        "parameters": tool_call.get("parameters", {}),
                    }
                if parsed_decision:
                    decision = parsed_decision
                    try:
                        pname = decision.get("name")
                        pparams = (
                            decision.get("parameters", {}) if isinstance(decision, dict) else {}
                        )
                        preason = pparams.get("reasoning") if isinstance(pparams, dict) else None
                        pacts = pparams.get("actions") if isinstance(pparams, dict) else None
                        entry = {
                            "turn": turn,
                            "name": pname,
                            "reasoning": preason,
                            "actions": pacts if isinstance(pacts, list) else [],
                        }
                        self.agent_state["recent_tool_calls"].append(entry)
                        if len(self.agent_state["recent_tool_calls"]) > 10:
                            self.agent_state["recent_tool_calls"] = self.agent_state[
                                "recent_tool_calls"
                            ][-10:]
                    except Exception:
                        pass
                    # Clear failure flag on success
                    if self.agent_state.get("last_failure"):
                        self.agent_state["last_failure"] = None
                    params = decision.get("parameters", {}) if isinstance(decision, dict) else {}
                    actions = params.get("actions", []) if isinstance(params, dict) else []
                    num_actions = len(actions) if isinstance(actions, list) else 0
                    # Store metrics for tqdm postfix update in run_episode
                    self.agent_state["last_metrics"] = {
                        "tc": 1,
                        "act": num_actions,
                        "tok": completion_tokens,
                        "in": prompt_tokens,
                        "tps": f"{toks_per_sec}" if toks_per_sec is not None else "-",
                    }
                else:
                    # Unrecognized tool_calls structure: do nothing, record failure
                    failure_msg = "Unrecognized tool_calls structure"
                    self.agent_state["last_failure"] = failure_msg
                    decision = {
                        "name": "interact",
                        "parameters": {"actions": [], "reasoning": failure_msg},
                    }
                    if self.verbose:
                        print(f"🔍 DEBUG: {failure_msg}")
            else:
                # No tool calls: do nothing, record failure for next prompt
                failure_msg = "No valid tool_calls in assistant message"
                self.agent_state["last_failure"] = failure_msg
                decision = {
                    "name": "interact",
                    "parameters": {"actions": [], "reasoning": failure_msg},
                }
                # Store metrics for tqdm postfix update in run_episode
                self.agent_state["last_metrics"] = {
                    "tc": 0,
                    "act": 0,
                    "tok": completion_tokens,
                    "in": prompt_tokens,
                    "tps": f"{toks_per_sec}" if toks_per_sec is not None else "-",
                }

            # Update agent state
            self.agent_state["tool_calls_made"] += 1

            # Add assistant response to history
            assistant_message = {"role": "assistant", "content": raw_response}
            self.agent_state["message_history"].append(assistant_message)

            if self.verbose:
                print(f"🤖 LM Response (turn {turn}): {json.dumps(decision, indent=2)}")
                print(f"📊 Response time: {llm_end - llm_start:.2f}s")
        except Exception as e:
            print(f"❌ Error in LM decide: {e}")
            import traceback

            traceback.print_exc()
            # Record failure and do nothing this turn
            failure_msg = f"Exception during decide: {str(e)}"
            self.agent_state["last_failure"] = failure_msg
            decision = {"name": "interact", "parameters": {"actions": [], "reasoning": failure_msg}}

        return decision

    def _parse_tool_response(self, raw_response: str) -> dict[str, Any]:
        """Parse raw LM response to extract tool calls."""
        # Try to parse JSON if present
        try:
            # Look for JSON in the response
            import re

            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "name" in data:
                    return data
                elif "function" in data:
                    return {
                        "name": data["function"].get("name", "interact"),
                        "parameters": data["function"].get("arguments", {}),
                    }
        except Exception:
            pass

        # Fallback to text parsing
        if "terminate" in raw_response.lower():
            return {"name": "terminate", "parameters": {"reason": "Agent decided to terminate"}}

        # Try to extract actions from the response
        actions = []
        action_keywords = [
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "do",
            "sleep",
            "place_stone",
            "place_table",
            "place_furnace",
            "place_plant",
            "make_wood_pickaxe",
            "make_stone_pickaxe",
            "make_iron_pickaxe",
            "make_wood_sword",
            "make_stone_sword",
            "make_iron_sword",
        ]

        for keyword in action_keywords:
            if keyword in raw_response.lower():
                actions.append(keyword)

        if not actions:
            actions = ["do"]  # Default action

        return {
            "name": "interact",
            "parameters": {
                "actions": actions,  # Return as array of actions
                "reasoning": "Parsed from response",
            },
        }

    def get_system_message(self) -> str:
        """Return system message for agent. Override in subclasses."""
        return """You are an AI agent playing Crafter. Use the available tools to interact with the environment.

CRITICAL RULE: You MUST provide MULTIPLE actions (2-5) in EVERY interact() tool call!

The 'interact' function accepts a LIST of 1-5 actions. ALWAYS provide 2-5 actions for efficiency.

GOOD Examples (what you SHOULD do):
✓ interact(actions=["move_right", "move_right", "do"], reasoning="Move to tree and collect wood")
✓ interact(actions=["move_up", "move_up", "move_right", "do"], reasoning="Navigate to stone and mine it")
✓ interact(actions=["place_table", "make_wood_pickaxe", "move_left"], reasoning="Craft and continue exploring")

BAD Examples (what you should AVOID):
✗ interact(actions=["move_right"], reasoning="Move right") - TOO FEW ACTIONS!
✗ interact(actions=["do"], reasoning="Collect") - TOO FEW ACTIONS!

REMEMBER: Single actions waste time. Always plan 2-5 actions ahead and execute them together!"""

    def format_observation(self, obs: dict[str, Any]) -> str:
        """Format observation for agent. Override in subclasses."""
        return str(obs)


# --- Crafter-specific ReAct Agent ---
class CrafterReActAgentWithLMSynth(BaseReActAgentWithLMSynth):
    """Crafter-specific ReAct agent with enhanced prompting for Synth models."""

    def get_system_message(self) -> str:
        """Return Crafter-specific system message optimized for Synth models."""
        override = os.getenv("CRAFTER_SYSTEM_PROMPT")
        if override:
            return override
        return """You are CrafterAgent playing Crafter survival environment. Your goal is to unlock as many achievements as possible while staying alive.

You will see a semantic map view showing your surroundings. Use this to navigate toward resources.

Key mechanics:
• 'do' action: collect wood from trees, stone from deposits, food from cows/plants
• 'do' does nothing on grass/water - move to find resources first
• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools
• Sleep when energy low to restore and unlock wake_up achievement
• Use semantic map view to navigate toward resources you can see

Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop

KEY ACHIEVEMENTS TO UNLOCK:
Basic Resource Collection (PRIORITY #1):
- collect_wood: Move NEXT TO a tree, then use action="do" to collect wood
- collect_stone: Move NEXT TO stone, then use action="do" (requires wood_pickaxe in inventory)
- collect_coal: Move NEXT TO coal, then use action="do" (requires stone_pickaxe)
- collect_iron: Move NEXT TO iron, then use action="do" (requires stone_pickaxe)
- collect_diamond: Move NEXT TO diamond, then use action="do" (requires iron_pickaxe)

Tool Crafting (enables resource collection):
- make_wood_pickaxe: Use action="make_wood_pickaxe" when you have wood (unlocks ability to mine stone)
- make_stone_pickaxe: Use action="make_stone_pickaxe" when you have wood and stone (unlocks coal/iron mining)
- make_iron_pickaxe: Use action="make_iron_pickaxe" when you have wood, coal, and iron (unlocks diamond mining)

Weapon Crafting (for defense):
- make_wood_sword: Use action="make_wood_sword" when you have wood
- make_stone_sword: Use action="make_stone_sword" when you have wood and stone  
- make_iron_sword: Use action="make_iron_sword" when you have wood, coal, and iron

Survival Actions:
- eat_plant: Use action="eat_plant" when food < 9 and you see a plant nearby
- eat_cow: Move NEXT TO cow, use action="do" to kill it, then action="eat_cow"
- collect_drink: Move NEXT TO water, then use action="drink" when drink < 9
- sleep: Use action="sleep" when energy < 5 (restores energy to 9)

Building/Placing:
- place_table: Use action="place_table" when you have wood (enables advanced crafting)
- place_furnace: Use action="place_furnace" when you have stone (for smelting)
- place_plant: Use action="place_plant" when you have sapling (grows into tree)
- place_stone: Use action="place_stone" when you have stone (creates barrier)

Combat:
- defeat_zombie: Move NEXT TO zombie, then use action="do" repeatedly to attack
- defeat_skeleton: Move NEXT TO skeleton, then use action="do" repeatedly to attack

CRITICAL: The action="do" is your INTERACTION button! Use it when adjacent to:
- Trees → get wood
- Stone/Coal/Iron/Diamond → mine resources (need appropriate pickaxe)
- Enemies → attack them
- Cows → kill for food

Simple Strategy:
1. Look for resources (trees, stones) in the semantic map
2. Move toward the nearest resource
3. When adjacent to a resource, use action="do" to collect it
4. If you have wood, try action="make_wood_pickaxe"
5. Repeat: find resources, move to them, use "do"

Critical Gameplay Tips:
- You must be ADJACENT (one tile away) to objects to interact with them
- Use "do" when next to: trees (for wood), stone (for stone), coal, iron, diamond
- Use "do" to attack zombies/skeletons when adjacent
- First priority: Find a tree, move next to it, then use "do" to collect wood
- Wood is essential for crafting your first pickaxe
- With wood_pickaxe you can mine stone, with stone_pickaxe you can mine iron, etc.

CRITICAL INSTRUCTION: You MUST ALWAYS provide MULTIPLE actions (2-5) in EVERY interact() tool call!

The 'interact' function accepts a LIST of 1-5 actions. NEVER use single actions - always plan 2-5 actions ahead!

MANDATORY action sequences (ALWAYS use multiple):
✓ interact(actions=["move_right", "move_right", "do"], reasoning="Move to tree and collect wood") 
✓ interact(actions=["move_up", "move_up", "move_right", "do"], reasoning="Navigate and collect")
✓ interact(actions=["place_table", "make_wood_pickaxe", "move_left", "move_left"], reasoning="Craft and explore")
✓ interact(actions=["do", "move_right", "do", "move_right", "do"], reasoning="Collect multiple resources")

FORBIDDEN (NEVER do this):
✗ interact(actions=["move_right"], ...) - WRONG! Too few actions!
✗ interact(actions=["do"], ...) - WRONG! Too few actions!

RULE: If you use less than 2 actions, you are playing inefficiently. Always think 2-5 steps ahead!

Key Strategy:
1. Plan a sequence of moves to reach resources
2. Execute multiple moves in one tool call (e.g., ["move_right", "move_right", "move_up"])
3. When adjacent to a resource, use "do" to collect it
4. Chain crafting actions together (e.g., ["place_table", "make_wood_pickaxe"])

Remember:
- Use "do" when ADJACENT to trees (for wood), stones, or other resources
- Collect wood FIRST before trying to craft anything
- Be efficient - use multiple actions per tool call!
- Focus on unlocking achievements by collecting resources and crafting items."""

    def format_observation(self, obs: dict[str, Any]) -> str:
        """Format Crafter observation with semantic map view."""
        # Get semantic map view
        semantic_view = format_semantic_map_view_v2(obs, view_size=7)

        # Extract key information
        inventory = obs.get("inventory", {})
        # Try both possible keys for achievements
        achievements = obs.get("achievements_status", obs.get("achievements_info", {}))
        health = obs.get("health", 10)
        food = obs.get("food", 10)
        drink = obs.get("drink", 10)
        energy = obs.get("energy", 10)

        # Count achievements
        achieved = sum(1 for v in achievements.values() if v)
        total_achievements = len(achievements)

        # Format inventory (only show non-zero items)
        inv_items = []
        for item, count in inventory.items():
            if count > 0:
                inv_items.append(f"{item}: {count}")
        inv_str = ", ".join(inv_items) if inv_items else "empty"

        # List unlocked achievements
        unlocked = [k for k, v in achievements.items() if v]
        unlocked_str = ", ".join(unlocked) if unlocked else "none"

        # Recent achievements (from info if available)
        recent_str = ""

        suppress_reminder = os.getenv("CRAFTER_SUPPRESS_OBS_REMINDER")
        base = (
            f"=== SEMANTIC MAP VIEW (7x7) ===\n"
            f"{semantic_view}\n\n"
            f"=== STATUS ===\n"
            f"Health: {health}/10 | Food: {food}/10 | Drink: {drink}/10 | Energy: {energy}/10\n"
            f"Inventory: {inv_str}\n"
            f"Achievements: {achieved}/{total_achievements} unlocked\n"
            f"Unlocked: {unlocked_str}\n"
            f"{recent_str}\n\n"
            # f"What do you see in the map? What actions should you take? "
        )
        if suppress_reminder:
            return base
        return (
            base
            # + "\n\nREMINDER: You MUST provide 2-5 actions in your interact() tool call. Plan multiple steps ahead!\n"
            # + 'Example: interact(actions=["move_right", "move_right", "do"], reasoning="Move to tree and collect wood")'
        )


async def run_episode(
    episode_id: int,
    config: CrafterConfig,
    session_tracer: SessionTracer | None = None,
    progress_bar: tqdm | None = None,
    quiet: bool = False,
    model_params: dict[str, Any] | None = None,
):
    """Run a single episode."""
    episode_start_time = time.time()

    # Create agent - always disable verbose for cleaner output
    agent = CrafterReActAgentWithLMSynth(
        model_name=config.model_name,
        max_turns=config.max_turns,
        verbose=False,  # Always disable verbose logging in agent
        tracer=session_tracer,
        episode_id=episode_id,
        quiet=True,  # Always use quiet mode for agent
        model_params=model_params,
    )

    # Initialize environment
    async with AsyncClient(base_url=config.service_base_url) as client:
        try:
            # Initialize environment with unique seed for each episode
            # Use simple sequential seeds: 1, 2, 3, 4, etc.
            episode_seed = episode_id + 1  # Start from 1 instead of 0

            init_response = await retry_http_request(
                client,
                "POST",
                "/env/CrafterClassic/initialize",
                json={"config": {"difficulty": config.difficulty, "seed": episode_seed}},
            )

            init_data = init_response.json()
            instance_id = init_data["env_id"]
            obs = init_data["observation"]

            # Start initial timestep and send initial observation as message
            if session_tracer:
                async with session_tracer.timestep("init", turn_number=0):
                    obs_msg = create_message(
                        compress_observation_for_trace(obs),
                        "observation",
                        f"crafter_env_{instance_id}",
                        0,
                    )
                    await session_tracer.record_message(
                        content=obs_msg.content, message_type=obs_msg.message_type
                    )

            # Run episode
            episode_reward = 0
            termination_reason = None
            step_results = []
            consecutive_no_tool_calls = 0

            # Create progress bar for this episode
            episode_progress = tqdm(
                total=config.max_turns,
                desc=f"Episode {episode_id}",
                position=episode_id,
                leave=True,
                ncols=100,
            )

            for turn in range(config.max_turns):
                episode_progress.update(1)

                # Use timestep context for this turn
                timestep_name = f"turn_{turn + 1}"
                async with (
                    session_tracer.timestep(timestep_name, turn_number=turn + 1)
                    if session_tracer
                    else _noop_async_context()
                ):
                    # Get agent decision
                    obs_formatted = agent.format_observation(obs)
                    system_msg = agent.get_system_message()

                    decision = await agent.decide(obs_formatted, system_msg, turn)
                    # Update tqdm postfix with latest metrics from agent
                    try:
                        metrics = agent.agent_state.get("last_metrics")
                        if isinstance(metrics, dict):
                            episode_progress.set_postfix(metrics, refresh=False)
                    except Exception:
                        pass

                    # Handle termination
                    if decision["name"] == "terminate":
                        termination_reason = decision["parameters"]["reason"]
                        break

                    # Detect consecutive no-tool-call responses and abort after 3
                    decision_params = (
                        decision.get("parameters") if isinstance(decision, dict) else None
                    )
                    decision_actions = (
                        decision_params.get("actions", [])
                        if isinstance(decision_params, dict)
                        else []
                    )
                    if (
                        decision.get("name") == "interact"
                        and isinstance(decision_actions, list)
                        and len(decision_actions) == 0
                    ):
                        consecutive_no_tool_calls += 1
                        print(f"🔍 DEBUG: consecutive_no_tool_calls={consecutive_no_tool_calls}")
                    else:
                        consecutive_no_tool_calls = 0
                    if consecutive_no_tool_calls >= 3:
                        # Gracefully end the episode without recording this problematic turn
                        termination_reason = "no_tool_calls_abort"
                        break

                    # Execute actions in sequence
                    actions = (
                        decision["parameters"].get("actions", [])
                        if isinstance(decision.get("parameters"), dict)
                        else []
                    )

                    # Ensure control variables are defined even if no actions are taken this turn
                    done = False
                    reward = 0.0
                    info = {}

                    # Define action mapping
                    crafter_action_map = {
                        "noop": 0,
                        "move_left": 1,
                        "move_right": 2,
                        "move_up": 3,
                        "move_down": 4,
                        "do": 5,
                        "sleep": 6,
                        "place_stone": 7,
                        "place_table": 8,
                        "place_furnace": 9,
                        "place_plant": 10,
                        "make_wood_pickaxe": 11,
                        "make_stone_pickaxe": 12,
                        "make_iron_pickaxe": 13,
                        "make_wood_sword": 14,
                        "make_stone_sword": 15,
                        "make_iron_sword": 16,
                    }

                    # Execute each action in the sequence (may be empty)
                    for action in actions:
                        # Convert action name to integer
                        action_int = crafter_action_map.get(action, 0)  # Default to noop

                        # Get state before action
                        state_before = {"observation": obs} if "obs" in locals() else {}
                        prev_obs = obs.copy()

                        # Step environment
                        step_response = await retry_http_request(
                            client,
                            "POST",
                            "/env/CrafterClassic/step",
                            json={
                                "env_id": instance_id,
                                "action": {
                                    "tool_calls": [
                                        {"tool": "interact", "args": {"action": action_int}}
                                    ]
                                },
                            },
                        )
                        step_data = step_response.json()

                        # Check if response has expected structure
                        if "observation" not in step_data:
                            print(
                                f"\n❌ Error: Missing observation in step response. Keys: {list(step_data.keys())}"
                            )
                            if "error" in step_data:
                                print(f"   Error message: {step_data['error']}")
                            # Try to recover or break
                            break

                        obs = step_data["observation"]
                        reward = step_data.get("reward", 0)  # Default to 0 if None
                        done = step_data.get("done", False)  # Default to False if None
                        info = step_data.get("info", {})

                        # Calculate achievement reward if not provided by service
                        if (reward == 0 or reward is None) and (
                            "achievements_status" in obs and "achievements_status" in prev_obs
                        ):
                            prev_achievements = prev_obs["achievements_status"]
                            curr_achievements = obs["achievements_status"]
                            new_unlocks = sum(
                                1
                                for k in curr_achievements
                                if curr_achievements.get(k) and not prev_achievements.get(k)
                            )
                            if new_unlocks > 0:
                                reward = float(new_unlocks)  # +1 for each new achievement

                        if reward is not None:
                            episode_reward += reward

                        # Record step result
                        step_results.append(
                            {
                                "turn": turn,
                                "action": action,
                                "reward": reward,
                                "done": done,
                                "info": info,
                            }
                        )

                        # Record environment event for hooks to catch
                        if session_tracer:
                            # Create environment event with state transition
                            env_event = EnvironmentEvent(
                                time_record=TimeRecord(event_time=time.time(), message_time=turn),
                                system_instance_id=f"crafter_env_{instance_id}",
                                system_state_before={"public_state": prev_obs},
                                system_state_after={"public_state": obs},
                                reward=reward,  # This now includes calculated achievement rewards
                                terminated=done,
                                metadata={"action": action, "action_int": action_int, "info": info},
                            )
                            await session_tracer.record_event(env_event)

                            # Also record runtime event for invalid action detection
                            runtime_event = RuntimeEvent(
                                time_record=TimeRecord(event_time=time.time(), message_time=turn),
                                system_instance_id=f"crafter_runtime_{instance_id}",
                                actions=[action_int],
                                metadata={
                                    "action_name": action,
                                    "action_int": action_int,
                                    "reward": reward,
                                    "state_before": state_before,
                                    "state_after": {"observation": obs},
                                },
                            )
                            await session_tracer.record_event(runtime_event)

                        if done:
                            break

                    # After all actions (or none), send final observation message
                    if session_tracer:
                        obs_msg = create_message(
                            compress_observation_for_trace(obs),
                            "observation",
                            f"crafter_env_{instance_id}",
                            turn + 1,
                        )
                        await session_tracer.record_message(
                            content=obs_msg.content, message_type=obs_msg.message_type
                        )

                    if done:
                        break

            # Close progress bar
            episode_progress.close()

            # Terminate instance
            terminate_response = await retry_http_request(
                client, "POST", "/env/CrafterClassic/terminate", json={"env_id": instance_id}
            )

        except Exception as e:
            if "episode_progress" in locals():
                episode_progress.close()
            print(f"\n❌ Episode {episode_id} failed: {e}")
            if config.verbose:
                import traceback

                traceback.print_exc()
            return {
                "episode_id": episode_id,
                "error": str(e),
                "duration": time.time() - episode_start_time,
            }

    # Extract final achievements
    final_achievements = []
    if obs and "achievements_status" in obs:
        final_achievements = [k for k, v in obs["achievements_status"].items() if v]

    # Return results
    return {
        "episode_id": episode_id,
        "total_reward": episode_reward,
        "steps": len(step_results),
        "termination_reason": termination_reason,
        "duration": time.time() - episode_start_time,
        "step_results": step_results,
        "achievements_unlocked": final_achievements,
    }


# --- Main ---
async def main():
    """Main entry point with v3 tracing."""
    parser = argparse.ArgumentParser(description="Run Crafter evaluation with LM Synth backend")
    parser.add_argument("--config", type=str, help="Path to TOML config file")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--episodes", type=int, help="Number of episodes (overrides config)")
    parser.add_argument("--max-steps", type=int, help="Max steps per episode (overrides config)")
    parser.add_argument(
        "--difficulty", type=str, choices=["easy", "normal", "hard"], help="Difficulty override"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress most output except results")
    parser.add_argument("--no-traces", action="store_true", help="Disable trace saving")
    parser.add_argument("--analyze", action="store_true", help="Analyze traces after running")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip model warmup")
    parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Don't start sqld daemon (assumes it's already running)",
    )

    # Qwen3 thinking mode flags (mutually exclusive)
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=True)",
    )
    think_group.add_argument(
        "--no-think",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode (chat_template_kwargs.enable_thinking=False)",
    )
    parser.set_defaults(enable_thinking=None)

    # Model parameter arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model responses (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--frequency-penalty", type=float, default=0.0, help="Frequency penalty (default: 0.0)"
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=0.0, help="Presence penalty (default: 0.0)"
    )
    parser.add_argument(
        "--tool-choice",
        type=str,
        choices=["auto", "required", "none"],
        default="auto",
        help="Tool choice mode (default: auto)",
    )

    args = parser.parse_args()

    # Load configuration
    config = CrafterConfig(args.config)

    # Setup Synth environment variables
    setup_synth_environment()

    # Clean up old files to keep directory clean
    if config.auto_cleanup:
        cleanup_old_files()

    # Apply command-line overrides
    if args.model:
        config.model_name = args.model
    if args.episodes:
        config.num_instances = args.episodes
    if args.max_steps:
        config.max_turns = args.max_steps
    if args.difficulty:
        config.difficulty = args.difficulty
    if args.verbose:
        config.verbose = True
    if args.quiet:
        config.quiet = True
        if not args.verbose:  # Don't show this if verbose is also on
            print("🔇 Quiet mode enabled - suppressing verbose logs")
    else:
        config.quiet = False
    if args.no_daemon:
        config.start_sqld_daemon = False

    # Environment overrides for model parameters (fail-fast on bad values)
    env_temp = os.getenv("CRAFTER_TEMPERATURE")
    if env_temp is not None:
        args.temperature = float(env_temp)
    env_max_tok = os.getenv("CRAFTER_MAX_TOKENS")
    if env_max_tok is not None:
        args.max_tokens = int(env_max_tok)
    env_tool_choice = os.getenv("CRAFTER_TOOL_CHOICE")
    if env_tool_choice is not None:
        if env_tool_choice not in {"auto", "required", "none"}:
            raise ValueError(f"Invalid CRAFTER_TOOL_CHOICE: {env_tool_choice}")
        args.tool_choice = env_tool_choice
    env_top_p = os.getenv("CRAFTER_TOP_P")
    if env_top_p is not None:
        args.top_p = float(env_top_p)
    env_freq_pen = os.getenv("CRAFTER_FREQUENCY_PENALTY")
    if env_freq_pen is not None:
        args.frequency_penalty = float(env_freq_pen)
    env_pres_pen = os.getenv("CRAFTER_PRESENCE_PENALTY")
    if env_pres_pen is not None:
        args.presence_penalty = float(env_pres_pen)

    # Resolve stop-after-tool-calls from environment (wrapper sets this)
    try:
        _satc = int(os.getenv("CRAFTER_STOP_AFTER_TOOL_CALLS", "1"))
    except Exception:
        _satc = 1
    _extra_body = {"stop_after_tool_calls": _satc} if _satc and _satc > 0 else {}

    # Create model parameters dictionary from command line arguments
    model_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "tool_choice": args.tool_choice,
        # Request early stop after N tool call blocks to avoid spillover
        "extra_body": _extra_body,
    }
    # Optionally carry thinking mode through to LM config
    if args.enable_thinking is not None:
        model_params["enable_thinking"] = args.enable_thinking

    # Configure logging based on quiet mode
    setup_logging(quiet_mode=config.quiet)

    # Display configuration (only if not in quiet mode)
    if not config.quiet:
        print("🎮 Crafter ReAct Agent Evaluation (LM with Synth Backend - v3)")
        print(f"Model: {config.model_name}")
        print("Model Parameters:")
        print(f"  Temperature: {model_params['temperature']}")
        print(f"  Max Tokens: {model_params['max_tokens']}")
        print(f"  Top-p: {model_params['top_p']}")
        print(f"  Frequency Penalty: {model_params['frequency_penalty']}")
        print(f"  Presence Penalty: {model_params['presence_penalty']}")
        print(f"Service: {config.service_base_url}")
        print(f"Instances: {config.num_instances}")
        print(f"Max Turns: {config.max_turns}")
        print(f"Difficulty: {config.difficulty}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)

    if args.no_traces:
        config.save_traces = False
        config.enable_v3_tracing = False
    if args.analyze:
        config.analyze_traces = True
    if args.skip_warmup:
        config.warmup_model = False

    # Ensure model is specified
    if not config.model_name:
        parser.error("Model name must be specified via --model or config file")

    # Test service health
    async with AsyncClient(base_url=config.service_base_url) as client:
        try:
            health_resp = await retry_http_request(client, "GET", "/health")
            health_data = health_resp.json()
            print(f"✅ Crafter service is healthy: {health_data}")
        except Exception as e:
            print(f"❌ Failed to connect to Crafter service: {e}")
            return

    # Warm up the model if requested
    if config.warmup_model and not args.skip_warmup:
        print(f"\n🔥 Warming up {config.model_name} on Synth backend...")
        try:
            synth_base_url = os.getenv("SYNTH_BASE_URL")  # or os.getenv('MODAL_BASE_URL')
            synth_api_key = os.getenv("SYNTH_API_KEY")  # or os.getenv('MODAL_API_KEY')
            if synth_base_url and synth_api_key:
                synth_config = SynthConfig(
                    base_url=synth_base_url,
                    api_key=synth_api_key,
                    timeout=config.warmup_timeout,  # Use configurable timeout
                )
                warmed = await warmup_synth_model(config.model_name, synth_config)
                if warmed:
                    print("✅ Model warmed up successfully!")
                else:
                    print("⚠️  Warmup did not complete; continuing anyway...")
            else:
                print("⚠️  Missing SYNTH_BASE_URL or SYNTH_API_KEY, skipping warmup")
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
            print("Continuing anyway...")

    # Set up v3 tracing if enabled
    trace_manager = None
    experiment_ctx = None
    sqld_daemon = None

    if config.enable_v3_tracing:
        # Create trace directory first
        os.makedirs(config.v3_trace_dir, exist_ok=True)

        # Start sqld daemon if requested
        if config.start_sqld_daemon:
            print("\n🚀 Starting sqld daemon for v3 tracing...")
            sqld_daemon = SqldDaemon(db_path=config.turso_db_path)
            sqld_daemon.__enter__()  # Start the daemon
            await asyncio.sleep(2)  # Give it time to start
            print("✅ sqld daemon started")

        # Initialize trace manager with proper URL format
        # If SQLD_DB_PATH is a directory managed by sqld, use its data file
        _db_path = config.turso_db_path
        if os.path.isdir(_db_path):
            _candidate = os.path.join(_db_path, "dbs", "default", "data")
            if os.path.exists(_candidate):
                _db_path = _candidate
        db_url = f"sqlite+aiosqlite:///{os.path.abspath(_db_path)}"
        trace_manager = AsyncSQLTraceManager(db_url=db_url)
        await trace_manager.initialize()

        # Create experiment context
        experiment_ctx = await create_experiment_context(
            db_manager=trace_manager,
            experiment_name=f"crafter_lm_synth_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Crafter LM Synth experiment with {config.model_name} on {config.difficulty} difficulty, using LM class with v3 tracing",
        )

        print(f"\n📊 V3 Tracing enabled. Traces will be saved to: {config.turso_db_path}")
        print(f"   Experiment: {experiment_ctx['experiment_name']}")

    # Run episodes with bounded concurrency using asyncio.Semaphore
    # Control concurrency with env var CRAFTER_CONCURRENCY (default 5)
    try:
        _conc_str = os.getenv("CRAFTER_CONCURRENCY")
        max_concurrency = int(_conc_str) if _conc_str else 5
    except Exception:
        max_concurrency = 5
    concurrency_limiter = asyncio.Semaphore(max_concurrency)

    print(f"\n🚀 Running {config.num_instances} episodes (concurrency={max_concurrency})...")

    episode_seeds = []  # Track seeds used for each episode

    # Prepare episode tasks
    episode_tasks = []
    session_ids = []

    for i in range(config.num_instances):
        # Calculate episode seed for logging (simple sequential: 1, 2, 3, etc)
        episode_seed = i + 1
        episode_seeds.append(episode_seed)

        # Create session tracer for this episode if v3 tracing is enabled
        session_tracer = None
        if config.enable_v3_tracing and trace_manager:
            session_tracer = SessionTracer(hooks=QUIET_HOOKS)  # Use quiet hooks
            session_tracer.db = trace_manager  # Use existing manager
            session_tracer._initialized = True

            # Generate session ID
            session_id = f"crafter_episode_{i}_{uuid.uuid4().hex[:8]}"
            session_ids.append(session_id)

        # Create episode task with proper session context
        async def run_episode_with_session(ep_id, cfg, tracer, pb, quiet, sess_id, model_params):
            if tracer:
                async with tracer.session(
                    session_id=sess_id,
                    metadata={
                        "episode_id": ep_id,
                        "experiment_id": experiment_ctx["experiment_id"]
                        if experiment_ctx
                        else None,
                    },
                ):
                    return await run_episode(ep_id, cfg, tracer, pb, quiet, model_params)
            else:
                return await run_episode(ep_id, cfg, tracer, pb, quiet, model_params)

        # Freeze per-iteration values to avoid late-binding bugs in closures
        this_tracer = session_tracer
        this_session_id = session_ids[i] if session_ids else None

        async def _limited_episode(ep_idx=i, tracer=this_tracer, sess_id=this_session_id):
            async with concurrency_limiter:
                return await run_episode_with_session(
                    ep_idx, config, tracer, None, args.quiet, sess_id, model_params
                )

        episode_task = _limited_episode()
        episode_tasks.append(episode_task)

    print("\n📤 Starting episodes...")
    start_time = time.time()

    # Run all episodes in parallel and fail fast on first error
    try:
        results = await asyncio.gather(*episode_tasks, return_exceptions=False)
    except Exception as e:
        print(f"\n❌ Run aborted due to error: {e}")
        # Ensure resources are cleaned up before exiting
        if trace_manager:
            await trace_manager.close()
        if sqld_daemon:
            sqld_daemon.__exit__(None, None, None)
            print("\n✅ Stopped sqld daemon")
        raise

    end_time = time.time()
    parallel_time = end_time - start_time

    print(f"\n✅ Completed {len(episode_tasks)} episodes in {parallel_time:.2f} seconds")

    # Process results and handle any exceptions
    successful_results = []
    failed_results = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Episode {i} failed: {result}")
            failed_results.append({"episode_id": i, "error": str(result)})
        else:
            successful_results.append(result)

            # Link session to experiment if tracing enabled
            if (
                config.enable_v3_tracing
                and trace_manager
                and experiment_ctx
                and i < len(session_ids)
            ):
                await trace_manager.link_session_to_experiment(
                    session_ids[i], experiment_ctx["experiment_id"]
                )

    # Use successful results for analysis
    results = successful_results + failed_results

    # Analyze results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)

    successful_episodes = [r for r in results if "error" not in r]
    failed_episodes = [r for r in results if "error" in r]

    if successful_episodes:
        total_reward = sum(r["total_reward"] for r in successful_episodes)
        total_steps = sum(r["steps"] for r in successful_episodes)
        avg_reward = total_reward / len(successful_episodes)
        avg_steps = total_steps / len(successful_episodes)

        print(f"Episodes completed: {len(successful_episodes)}/{config.num_instances}")
        print(f"Failed episodes: {len(failed_episodes)}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per episode: {avg_reward:.2f}")
        print(f"Total steps: {total_steps}")
        print(f"Average steps per episode: {avg_steps:.2f}")

        # Show seeds used
        if episode_seeds:
            print("\nSeeds used:")
            for i, seed in enumerate(episode_seeds[: len(successful_episodes)]):
                print(f"  Episode {i}: seed {seed}")

        # Extract unique achievements
        all_achievements = set()
        achievement_counts = defaultdict(int)

        for result in successful_episodes:
            # Use the achievements_unlocked field we added
            if "achievements_unlocked" in result:
                for achievement in result["achievements_unlocked"]:
                    all_achievements.add(achievement)
                    achievement_counts[achievement] += 1

        # Extract and count all actions from successful episodes
        action_counts = defaultdict(int)
        total_actions = 0

        for result in successful_episodes:
            if "step_results" in result:
                for step in result["step_results"]:
                    if "action" in step:
                        action_counts[step["action"]] += 1
                        total_actions += 1

        print(f"Unique achievements unlocked: {len(all_achievements)}")
        if all_achievements:
            print("\nAchievements unlocked:")
            for achievement, count in sorted(achievement_counts.items()):
                print(
                    f"  - {achievement}: {count} episodes ({count / len(successful_episodes) * 100:.1f}%)"
                )

        # Display action counts
        if action_counts:
            print(f"\nAction counts (total: {total_actions}):")
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_actions * 100 if total_actions > 0 else 0
                print(f"  - {action}: {count} ({percentage:.1f}%)")
    else:
        print("No successful episodes completed.")

    # Save detailed results
    if config.save_detailed_results and config.enable_v3_tracing and trace_manager:
        # For v3, results are automatically saved in the database
        print(f"\n💾 Results available in Turso database: {config.turso_db_path}")
        print(f"   Experiment ID: {experiment_ctx['experiment_id']}")
        print("   Use the filter_traces_sft_turso.py script to extract fine-tuning data")
    elif config.save_detailed_results:
        # Fallback to JSON if no tracing - write under temp/ (git-ignored)
        from pathlib import Path

        out_dir = Path(os.getenv("SYNTH_OUTPUT_DIR", "temp")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = (
            out_dir / f"crafter_lm_synth_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "model": config.model_name,
                        "episodes": config.num_instances,
                        "max_steps": config.max_turns,
                        "difficulty": config.difficulty,
                        "backend": "synth",
                        "tracing": "v3",
                    },
                    "results": results,
                    "summary": {
                        "successful_episodes": len(successful_episodes),
                        "failed_episodes": len(failed_episodes),
                        "total_reward": total_reward if successful_episodes else 0,
                        "avg_reward": avg_reward if successful_episodes else 0,
                        "unique_achievements": list(all_achievements)
                        if successful_episodes
                        else [],
                    },
                },
                f,
                indent=2,
            )
        print(f"\n💾 Detailed results saved to: {results_path}")

    # Print a markdown row compatible with Environments/crafter.md tables
    if successful_episodes:
        # Columns: | model | trajectories | avg achievements | adj score | unique | steps sum | avg steps |
        model_label = config.model_name.replace("/", "/")
        trajectories = len(successful_episodes)
        avg_ach = avg_reward  # our reward == achievements unlocked per episode

        # Compute weighted scores (shaped and K-Score) from final achievements across episodes
        # K coefficients taken from crafter.md (representative weights)
        k_weights = {
            "collect_drink": 0.1,
            "collect_sapling": 0.1,
            "wake_up": 0.1,
            "collect_wood": 1.0,
            "collect_stone": 1.0,
            "eat_cow": 1.0,
            "defeat_zombie": 1.0,
            "defeat_skeleton": 1.0,
            "make_wood_pickaxe": 3.0,
            "place_table": 3.0,
            "collect_coal": 3.0,
            "make_stone_pickaxe": 10.0,
            "place_furnace": 10.0,
            "collect_iron": 10.0,
            "make_stone_sword": 10.0,
            "make_wood_sword": 3.0,
            "place_plant": 0.1,
        }

        # Aggregate final achievements across successful episodes
        from collections import Counter

        ach_counter: Counter[str] = Counter()
        for ep in successful_episodes:
            for name in ep.get("achievements_unlocked", []):
                ach_counter[name] += 1

        shaped_total = 0.0
        for name, count in ach_counter.items():
            k = k_weights.get(name, 1.0)
            shaped_total += k * count

        # Shaped reward per episode average
        shaped_reward_avg = shaped_total / trajectories if trajectories > 0 else 0.0
        k_score_avg = shaped_reward_avg / 20.0  # normalize roughly to match table scale

        # unique = len(all_achievements)  # unused
        steps_sum = total_steps
        avg_steps_md = avg_steps
        print("\nMarkdown row:")
        print(
            f"| {model_label:<15} | {trajectories:7d} | {avg_ach:8.2f} | {shaped_reward_avg:13.3f} | {k_score_avg:12.3f} | {steps_sum:12.3f} | {avg_steps_md:8.3f} |"
        )

    # Cleanup
    if trace_manager:
        await trace_manager.close()

    if sqld_daemon:
        sqld_daemon.__exit__(None, None, None)
        print("\n✅ Stopped sqld daemon")


if __name__ == "__main__":
    asyncio.run(main())


# === SEMANTIC MAP VIEW (15x15) ===
# stone coal iron coal coal coal coal
# stone stone iron coal coal coal coal
# stone stone zombie coal coal iron iron
# stone stone stone you stone iron iron
# stone stone stone stone stone stone stone
# stone stone stone stone stone stone stone
# stone stone stone stone stone stone stone
# Visible items: coal, iron, stone, zombie

# === STATUS ===
# Health: 10/10 | Food: 10/10 | Drink: 10/10 | Energy: 10/10
# Inventory: health: 9, food: 7, drink: 7, energy: 9, wood: 1, wood_pickaxe: 1
# Achievements: 4/22 unlocked
# Unlocked: collect_wood, make_wood_pickaxe, place_table, wake_up
