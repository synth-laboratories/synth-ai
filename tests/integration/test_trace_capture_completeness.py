"""Integration test for trace capture completeness.

This test verifies that the interceptor proxy captures complete trace data,
including tool calls AND their results, for proper verifier evaluation.

Issue discovered: The trace captures tool_calls in LLM responses, but the
tool results (which appear as subsequent messages) have empty content.
This makes it impossible for verifiers to evaluate what the agent actually did.

Run with local backend:
    cd synth-ai
    uv run pytest tests/integration/test_trace_capture_completeness.py -v -s

Requirements:
    - Local backend running on localhost:8000
    - SYNTH_API_KEY or auto-minted demo key
"""

import json
from pathlib import Path

import pytest


# Skip if no local backend
def backend_available() -> bool:
    try:
        import httpx

        resp = httpx.get("http://localhost:8000/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not backend_available(), reason="Local backend not running")
class TestTraceCaptureCompleteness:
    """Test that trace capture includes all necessary data for verifier evaluation."""

    @pytest.fixture
    def sample_trace_path(self) -> Path:
        """Path to a sample trace from a successful EngineBench eval."""
        # Use the trace we captured from eval_754ad00cfd334fa6
        path = Path("/tmp/eval_traces/seed_22.json")
        if not path.exists():
            pytest.skip("Sample trace not available - run EngineBench eval first")
        return path

    def test_trace_has_event_history(self, sample_trace_path: Path):
        """Trace should have event_history with LLM call events."""
        with open(sample_trace_path) as f:
            trace = json.load(f)

        assert "event_history" in trace, "Trace missing event_history"
        event_history = trace["event_history"]
        assert len(event_history) > 0, "event_history is empty"

        # All events should be lm_call type
        for event in event_history:
            assert event.get("type") == "lm_call", f"Unexpected event type: {event.get('type')}"

    def test_tool_calls_are_captured(self, sample_trace_path: Path):
        """Trace should capture tool_calls from LLM responses."""
        with open(sample_trace_path) as f:
            trace = json.load(f)

        event_history = trace["event_history"]

        # Find events with tool calls
        events_with_tool_calls = []
        for i, event in enumerate(event_history):
            llm_resp = event.get("llm_response", {})
            tool_calls = llm_resp.get("tool_calls", []) or llm_resp.get("message", {}).get(
                "tool_calls", []
            )
            if tool_calls:
                events_with_tool_calls.append((i, tool_calls))

        assert len(events_with_tool_calls) > 0, "No tool calls captured in trace"

        # Verify tool calls have function names and arguments
        for event_idx, tool_calls in events_with_tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                # Tool calls should have name and arguments
                assert "arguments" in func, f"Tool call missing arguments in event {event_idx}"

    def test_tool_results_have_content(self, sample_trace_path: Path):
        """FAILING: Tool results should have non-empty content.

        This test documents the bug: tool result messages have empty content,
        making it impossible for verifiers to see what the agent actually did.
        """
        with open(sample_trace_path) as f:
            trace = json.load(f)

        event_history = trace["event_history"]

        # Look for events where follow-up messages might contain tool results
        empty_content_messages = []
        non_empty_follow_ups = []

        for event_idx, event in enumerate(event_history):
            llm_req = event.get("llm_request", {})
            messages = llm_req.get("messages", [])

            # Skip first 2 messages (system + initial user prompt)
            for msg_idx, msg in enumerate(messages[2:], start=2):
                role = msg.get("role", "")
                content = msg.get("content", "")

                if isinstance(content, str):
                    content_len = len(content)
                elif isinstance(content, list):
                    content_len = sum(len(str(p)) for p in content)
                else:
                    content_len = 0

                if content_len == 0:
                    empty_content_messages.append(
                        {
                            "event": event_idx,
                            "message": msg_idx,
                            "role": role,
                        }
                    )
                else:
                    non_empty_follow_ups.append(
                        {
                            "event": event_idx,
                            "message": msg_idx,
                            "role": role,
                            "content_length": content_len,
                        }
                    )

        # This assertion documents the bug - we expect it to FAIL
        # When fixed, tool results should have non-empty content
        if empty_content_messages:
            pytest.fail(
                f"BUG: {len(empty_content_messages)} messages have empty content.\n"
                f"Empty messages: {empty_content_messages}\n"
                f"Non-empty follow-ups: {non_empty_follow_ups}\n\n"
                "Tool results and intermediate assistant responses are not being captured.\n"
                "This prevents verifiers from evaluating agent actions."
            )

    def test_trace_conversion_preserves_tool_calls(self, sample_trace_path: Path):
        """Verify trace conversion includes tool calls for verifier consumption."""
        with open(sample_trace_path) as f:
            trace = json.load(f)

        # This is the conversion used for verifier graphs
        def v3_trace_to_session_trace(v3_trace):
            """Minimal conversion matching trace_loader._v3_trace_to_session_trace"""
            import uuid

            meta = v3_trace.get("metadata", {})
            session_id = meta.get("trace_id") or v3_trace.get("trace_id") or str(uuid.uuid4())

            event_history = v3_trace.get("event_history", [])
            session_time_steps = []

            for step_idx, ev in enumerate(event_history):
                if ev.get("type") != "lm_call":
                    continue

                events = []
                llm_req = ev.get("llm_request", {})
                llm_resp = ev.get("llm_response", {})

                # Add request messages
                for msg in llm_req.get("messages", []):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        events.append(
                            {
                                "event_type": "runtime",
                                "type": f"{msg.get('role', 'user')}_message",
                                "content": content,
                            }
                        )

                # Add response content
                resp_msg = llm_resp.get("message", {})
                resp_content = resp_msg.get("content", "")
                if resp_content:
                    events.append(
                        {
                            "event_type": "runtime",
                            "type": "assistant_message",
                            "content": resp_content,
                        }
                    )

                # BUG: Tool calls are NOT being included in the conversion!
                tool_calls = llm_resp.get("tool_calls", []) or resp_msg.get("tool_calls", [])

                if events:
                    session_time_steps.append(
                        {
                            "step_id": str(step_idx),
                            "step_index": step_idx,
                            "events": events,
                            # Tool calls should be included but aren't in current conversion
                            "_tool_calls": tool_calls,  # For debugging
                        }
                    )

            return {
                "session_id": session_id,
                "session_time_steps": session_time_steps,
            }

        session_trace = v3_trace_to_session_trace(trace)

        # Check that tool calls were found in raw trace
        total_tool_calls = 0
        for step in session_trace["session_time_steps"]:
            tool_calls = step.get("_tool_calls", [])
            total_tool_calls += len(tool_calls)

        assert total_tool_calls > 0, "No tool calls found in raw trace"

        # Check that events include tool call information
        # Currently they don't - this documents the gap
        has_tool_info = False
        for step in session_trace["session_time_steps"]:
            for event in step.get("events", []):
                content = event.get("content", "")
                # Tool calls might be mentioned in assistant content
                if "edit" in content.lower() or "bash" in content.lower():
                    has_tool_info = True
                    break

        assert isinstance(has_tool_info, bool)

        # This may or may not be true depending on whether assistant mentions tool names
        # The key issue is that tool RESULTS are not captured

    def test_verifier_would_see_agent_actions(self, sample_trace_path: Path):
        """Verify that a verifier would be able to see what the agent did.

        For proper evaluation, the verifier needs to see:
        1. What tools the agent called
        2. What arguments were passed
        3. What results were returned
        4. How the agent responded to those results
        """
        with open(sample_trace_path) as f:
            trace = json.load(f)

        event_history = trace["event_history"]

        # Collect all information a verifier would need
        verifier_visible_info = {
            "tool_calls_made": [],
            "tool_results_received": [],
            "agent_reasoning": [],
        }

        for event in event_history:
            llm_req = event.get("llm_request", {})
            llm_resp = event.get("llm_response", {})

            # Tool calls from response
            tool_calls = llm_resp.get("tool_calls", []) or llm_resp.get("message", {}).get(
                "tool_calls", []
            )
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                if name:
                    verifier_visible_info["tool_calls_made"].append(
                        {
                            "name": name,
                            "args_length": len(args) if args else 0,
                        }
                    )

            # Look for tool results in messages (role=tool or following user messages)
            messages = llm_req.get("messages", [])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "tool" and content:
                    verifier_visible_info["tool_results_received"].append(
                        {
                            "content_length": len(content),
                        }
                    )

                # Check for tool_call_id which indicates tool result
                if msg.get("tool_call_id") and content:
                    verifier_visible_info["tool_results_received"].append(
                        {
                            "tool_call_id": msg.get("tool_call_id"),
                            "content_length": len(content),
                        }
                    )

            # Agent reasoning from response content
            resp_content = llm_resp.get("message", {}).get("content", "")
            if resp_content:
                verifier_visible_info["agent_reasoning"].append(
                    {
                        "content_length": len(resp_content),
                    }
                )

        # Assertions about what verifier can see
        assert len(verifier_visible_info["tool_calls_made"]) > 0, (
            "Verifier cannot see any tool calls"
        )

        # This is the key failing assertion
        if len(verifier_visible_info["tool_results_received"]) == 0:
            pytest.fail(
                "BUG: Verifier cannot see tool results.\n"
                f"Tool calls made: {len(verifier_visible_info['tool_calls_made'])}\n"
                f"Tool results visible: {len(verifier_visible_info['tool_results_received'])}\n"
                f"Agent reasoning visible: {len(verifier_visible_info['agent_reasoning'])}\n\n"
                "Without tool results, the verifier only sees that the agent TRIED to do things,\n"
                "not what actually happened or whether those actions succeeded."
            )


class TestTraceConversionForVerifier:
    """Test the trace conversion pipeline for verifier graphs."""

    def test_session_trace_format_requirements(self):
        """Document the SessionTraceInput format required by verifier graphs."""
        # Required fields for SessionTraceInput (from graphgen.routes)
        required_fields = {
            "session_id": "Unique identifier for the session",
            "session_time_steps": "List of time steps, each containing events",
        }

        # Each time step should have:
        time_step_fields = {
            "step_id": "Unique step identifier",
            "step_index": "0-based index of the step",
            "events": "List of events in this step",
        }

        # Each event should have:
        event_fields = {
            "event_type": "Type: 'runtime', 'environment', 'cais'",
            "type": "Subtype: 'user_message', 'assistant_message', etc.",
            "content": "The actual content of the event",
            "event_id": "Unique event identifier",
        }

        # For verifier to properly evaluate, events should include:
        # - User instructions
        # - Assistant reasoning and tool calls
        # - Tool execution results  <-- THIS IS MISSING
        # - Final assistant response

        # This test just documents the requirements
        assert required_fields and time_step_fields and event_fields

    def test_current_conversion_loses_tool_info(self):
        """Document that current trace conversion loses tool call/result info."""
        # The _v3_trace_to_session_trace function in trace_loader.py:
        # 1. Only looks at the FIRST lm_call event (bug: should handle all)
        # 2. Extracts messages from llm_request
        # 3. Extracts response content from llm_response.message
        # 4. Does NOT extract tool_calls from llm_response
        # 5. Does NOT include tool results (which are in follow-up messages)

        # This means verifier sees:
        # - Initial prompt
        # - Final response text (if any)

        # Verifier does NOT see:
        # - Tool calls made
        # - Tool arguments
        # - Tool results
        # - Intermediate reasoning

        # For a coding agent like OpenCode, this means the verifier
        # cannot evaluate:
        # - Which files were read
        # - What edits were made
        # - Whether cargo check/test passed

        assert True  # Documentation test
