"""
Example demonstrating DuckDB integration with tracing_v2.
"""
import os
import time
from datetime import datetime, timedelta
from openai import OpenAI

from synth_ai.tracing_v2.session_tracer import SessionTracer, SessionEventMessage, TimeRecord
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager


def run_example_session(tracer: SessionTracer):
    """Run an example session with multiple LLM calls."""
    # Start a session
    session = tracer.start_session("example-session-001")
    
    # Simulate multiple timesteps
    for i in range(3):
        timestep = tracer.start_timestep(f"step-{i}", iteration=i)
        
        # Record a user message
        user_msg = SessionEventMessage(
            content={"role": "user", "text": f"What is {i} + {i}?"},
            message_type="user_input",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i
            )
        )
        tracer.record_message(user_msg)
        
        # Make an LLM call (this will be captured by langfuse)
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What is {i} + {i}?"}
            ],
            max_tokens=50
        )
        
        # Capture the LLM interaction as a CAIS event
        cais_event = tracer.capture_llm_call(
            system_id="math_assistant",
            system_state_before={"question_number": i},
            system_state_after={"answer": response.choices[0].message.content}
        )
        
        # Record the assistant response
        assistant_msg = SessionEventMessage(
            content={"role": "assistant", "text": response.choices[0].message.content},
            message_type="assistant_response",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i
            )
        )
        tracer.record_message(assistant_msg)
        
        # Simulate some processing time
        time.sleep(0.5)
    
    # End session (will automatically upload to DuckDB if configured)
    tracer.end_session()


def analyze_traces():
    """Demonstrate analytics capabilities with DuckDB."""
    # Create DuckDB manager
    db_manager = DuckDBTraceManager("example_traces.duckdb")
    
    print("\n=== Model Usage Statistics ===")
    model_stats = db_manager.get_model_usage()
    print(model_stats)
    
    print("\n=== Session Summary ===")
    session_summary = db_manager.get_session_summary()
    print(session_summary)
    
    print("\n=== Expensive Calls (> $0.001) ===")
    expensive_calls = db_manager.get_expensive_calls(0.001)
    print(expensive_calls)
    
    print("\n=== Custom Query: Average tokens per model ===")
    custom_query = """
        SELECT 
            model_name,
            AVG(prompt_tokens) as avg_prompt_tokens,
            AVG(completion_tokens) as avg_completion_tokens,
            AVG(total_tokens) as avg_total_tokens
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY model_name
    """
    result = db_manager.query_traces(custom_query)
    print(result)
    
    # Export to Parquet for further analysis
    print("\n=== Exporting to Parquet ===")
    db_manager.export_to_parquet("./trace_exports")
    print("Exported to ./trace_exports/")
    
    db_manager.close()


def main():
    """Main example function."""
    print("DuckDB Integration Example for tracing_v2")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using mock mode.")
        # In real usage, you'd set the API key
        # For demo, we'll skip the actual LLM calls
    
    # Create tracer with DuckDB integration
    tracer = SessionTracer(
        traces_dir="example_traces",
        duckdb_path="example_traces.duckdb"
    )
    
    print("\n1. Running example session...")
    try:
        run_example_session(tracer)
        print("Session completed and uploaded to DuckDB!")
    except Exception as e:
        print(f"Session failed: {e}")
        print("(This is expected if OPENAI_API_KEY is not set)")
    finally:
        tracer.close()
    
    print("\n2. Analyzing traces with DuckDB...")
    analyze_traces()
    
    print("\n=== Example Complete ===")
    print("Check example_traces.duckdb for the database")
    print("Check ./trace_exports/ for Parquet files")


if __name__ == "__main__":
    main()