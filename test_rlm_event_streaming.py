#!/usr/bin/env python3
"""Test script for RLM event streaming.

This script tests the event streaming functionality for RLM completions by:
1. Creating a simple RLM graph definition
2. Submitting a run request to the local synth-graph-service
3. Streaming events from the /events/stream endpoint
4. Displaying events as they arrive

Usage:
    # With API key in environment
    export OPENAI_API_KEY=sk-...
    uv run python test_rlm_event_streaming.py

    # Or pass API key as argument
    uv run python test_rlm_event_streaming.py --api-key sk-...

    # Or use GROQ
    export GROQ_API_KEY=...
    uv run python test_rlm_event_streaming.py --provider groq
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

import httpx

# Simple RLM v1 graph that only uses local tools (no codex_exec)
SIMPLE_RLM_GRAPH = """
name: simple_rlm_test
metadata:
  type: TestGraph
  description: Simple RLM test graph for event streaming

start_nodes: [rlm_compute]

nodes:
  rlm_compute:
    name: rlm_compute
    type: DagNode
    input_mapping: '{"query": state.get("query", "What is 2+2?"), "context": state.get("context", "")}'
    implementation:
      type: rlm_compute
      rlm_impl: v1
      model_name: gpt-4o-mini
      max_iterations: 8
      max_root_calls: 8
      max_time_ms: 45000
      max_cost_usd: 0.25
      answer_schema: "simple_answer"
      tools:
        - materialize_context
        - local_grep
        - local_search
        - view_lines
      system_prompt: |
        You are a helpful assistant. You MUST use tools to read context before answering.
        Always call materialize_context(field_name="context", filename="context.txt"),
        then use local_search or local_grep to find the answer in the context.
      user_prompt: |
        Question: <input>query</input>
        The answer is in the provided context. Use the tools, then answer concisely.

control_edges:
  rlm_compute: []
"""


async def stream_events(base_url: str, run_id: str, timeout: float = 60.0) -> None:
    """Stream events from the /events/stream endpoint."""
    url = f"{base_url}/v1/runs/{run_id}/events/stream"
    params = {"run_id": run_id}

    print(f"\n{'=' * 80}")
    print(f"Streaming events for run_id: {run_id}")
    print(f"{'=' * 80}\n")

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url, params=params) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                print(f"ERROR: Failed to stream events: {response.status_code}")
                print(f"Response: {error_text.decode()}")
                return

            print("Event stream connected. Waiting for events...\n")
            event_count = 0

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        wrapper = json.loads(data_str)
                        # Event is nested: {"stream_id": "...", "event": {...}}
                        event_data = wrapper.get(
                            "event", wrapper
                        )  # Fallback to wrapper if no nested event
                        event_count += 1

                        event_type = event_data.get("event_type", "unknown")
                        ts_ms = event_data.get("ts_ms", 0)
                        seq = event_data.get("seq", 0)
                        payload = event_data.get("payload", {})

                        print(f"[{event_count}] seq={seq} ts={ts_ms} | {event_type}")

                        if event_type == "rlm_iteration_started":
                            iteration = payload.get("iteration", "?")
                            agent_id = payload.get("agent_id", "")
                            print(f"    Iteration: {iteration}, Agent: {agent_id}")
                        elif event_type == "rlm_tool_call_started":
                            tool = payload.get("tool", "?")
                            call_id = payload.get("call_id", "")
                            print(f"    Tool started: {tool} (call_id={call_id})")
                        elif event_type == "rlm_tool_call_completed":
                            tool = payload.get("tool", "?")
                            elapsed_ms = payload.get("elapsed_ms", "?")
                            preview = payload.get("result_preview", "")
                            print(f"    Tool completed: {tool} ({elapsed_ms}ms)")
                            if preview:
                                print(f"    Result preview: {preview[:160]}...")
                        elif event_type == "rlm_tool_call_failed":
                            tool = payload.get("tool", "?")
                            error = payload.get("error", "")
                            print(f"    Tool failed: {tool}")
                            if error:
                                print(f"    Error: {error}")
                        elif event_type == "rlm_limits_hit":
                            limits = payload.get("limits_hit", {})
                            print(f"    Limits hit: {json.dumps(limits, indent=6)}")
                        elif event_type == "rlm_llm_response":
                            completion_tokens = payload.get("completion_tokens")
                            prompt_tokens = payload.get("prompt_tokens")
                            print(
                                "    LLM response received"
                                + (
                                    f" (prompt={prompt_tokens}, completion={completion_tokens})"
                                    if prompt_tokens or completion_tokens
                                    else ""
                                )
                            )
                        elif event_type == "rlm_completed":
                            answer_preview = payload.get("answer_preview", "")
                            print(f"    Answer preview: {answer_preview[:100]}...")
                        elif event_type == "run_completed":
                            print("    Run completed!")
                            break
                        elif event_type == "run_failed":
                            reason = payload.get("reason", "Unknown error")
                            print(f"    Error: {reason}")
                            if "error" in payload:
                                print(
                                    f"    Error details: {json.dumps(payload.get('error'), indent=4)}"
                                )
                            break
                        elif event_type in [
                            "run_queued",
                            "run_started",
                            "graph_validated",
                            "node_started",
                            "rlm_started",
                        ]:
                            # Just acknowledge these events
                            pass
                        else:
                            # Print payload for other event types
                            if payload:
                                print(
                                    f"    Payload: {json.dumps(payload, indent=4, default=str)[:300]}..."
                                )

                    except json.JSONDecodeError as e:
                        print(f"WARNING: Failed to parse event JSON: {e}")
                        print(f"  Raw line: {line[:200]}")

                elif line.startswith("event: "):
                    # SSE event type
                    event_type = line[7:]
                    print(f"Event type: {event_type}")

            print(f"\n{'=' * 80}")
            print(f"Event stream ended. Total events received: {event_count}")
            print(f"{'=' * 80}\n")


async def create_run(
    base_url: str,
    graph_yaml: str,
    api_key: Optional[str] = None,
    provider: str = "openai",
) -> str:
    """Create a run and return the run_id."""
    url = f"{base_url}/v1/runs"

    # Prepare graph YAML - inject API key if provided
    if api_key:
        # Add api_key to the implementation if not present
        if "api_key:" not in graph_yaml:
            # Find the implementation section and add api_key
            lines = graph_yaml.split("\n")
            new_lines = []
            in_implementation = False
            for line in lines:
                new_lines.append(line)
                if "implementation:" in line:
                    in_implementation = True
                elif in_implementation and line.strip() and not line.startswith(" "):
                    # End of implementation block
                    in_implementation = False
                elif in_implementation and "model_name:" in line:
                    # Add api_key after model_name
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + f'api_key: "{api_key}"')
                    in_implementation = False
            graph_yaml = "\n".join(new_lines)

    # Set model based on provider
    if provider == "groq":
        graph_yaml = graph_yaml.replace("gpt-4o-mini", "llama-3.1-70b-versatile")
    elif provider == "google":
        graph_yaml = graph_yaml.replace("gpt-4o-mini", "gemini-2.0-flash-exp")

    payload = {
        "graph_yaml": graph_yaml,
        "inputs": {
            "query": "What is 2+2? Explain your reasoning.",
            "context": (
                "This is a short context file for the test.\n"
                "The correct answer is: 4.\n"
                "You should find the answer in this text.\n"
            ),
        },
        "options": {
            "rlm_event_level": "verbose",  # minimal, standard, or verbose
        },
    }

    print(f"Creating run at {url}...")
    print(f"Graph YAML length: {len(graph_yaml)} chars")
    print(f"Provider: {provider}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        if response.status_code != 200:
            error_text = response.text
            print(f"ERROR: Failed to create run: {response.status_code}")
            print(f"Response: {error_text}")
            sys.exit(1)

        result = response.json()
        run_id = result.get("run_id")
        if not run_id:
            print(f"ERROR: No run_id in response: {result}")
            sys.exit(1)

        print(f"✓ Run created: {run_id}")
        return run_id


def get_api_key(provider: str, api_key_arg: Optional[str]) -> Optional[str]:
    """Get API key from argument or environment."""
    if api_key_arg:
        return api_key_arg

    # Try provider-specific env vars first
    if provider == "groq":
        return os.getenv("GROQ_API_KEY")
    elif provider == "google":
        return os.getenv("GOOGLE_API_KEY")
    else:
        # Try generic first, then OpenAI-specific
        return os.getenv("GRAPH_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def check_service_has_api_key(base_url: str) -> bool:
    """Check if the service has API key configured by trying a simple request."""
    # This is a heuristic - we'll let the actual run fail if no key is available
    return True  # Assume service has it configured, let the run fail gracefully


async def main():
    parser = argparse.ArgumentParser(
        description="Test RLM event streaming from local synth-graph-service"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8091",
        help="Base URL for synth-graph-service (default: http://localhost:8091)",
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key (or set OPENAI_API_KEY/GROQ_API_KEY/GRAPH_LLM_API_KEY)",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "groq", "google"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--graph-yaml",
        help="Path to custom graph YAML file (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout for event streaming in seconds (default: 120)",
    )

    args = parser.parse_args()

    # Get API key (optional - service may have it in its own environment)
    api_key = get_api_key(args.provider, args.api_key)
    if not api_key:
        print("WARNING: No API key found in test script environment.")
        print("The synth-graph-service may have the API key in its own environment.")
        print("If the run fails with 'missing LLM API key', set one of:")
        if args.provider == "groq":
            print("  - GROQ_API_KEY environment variable (for synth-graph-service)")
        elif args.provider == "google":
            print("  - GOOGLE_API_KEY environment variable (for synth-graph-service)")
        else:
            print("  - OPENAI_API_KEY environment variable (for synth-graph-service)")
            print("  - GRAPH_LLM_API_KEY environment variable (for synth-graph-service)")
        print("  - Or pass --api-key to this script to inject it into the graph")
        print("\nContinuing anyway - will let the service handle API key lookup...\n")

    # Load graph YAML
    if args.graph_yaml:
        with open(args.graph_yaml) as f:
            graph_yaml = f.read()
    else:
        graph_yaml = SIMPLE_RLM_GRAPH

    # Create run
    try:
        run_id = await create_run(
            args.base_url, graph_yaml, api_key=api_key, provider=args.provider
        )
    except Exception as e:
        print(f"ERROR: Failed to create run: {e}")
        sys.exit(1)

    # Stream events
    try:
        await stream_events(args.base_url, run_id, timeout=args.timeout)
    except Exception as e:
        print(f"ERROR: Failed to stream events: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
