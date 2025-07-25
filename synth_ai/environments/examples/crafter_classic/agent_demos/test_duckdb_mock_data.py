#!/usr/bin/env python3
"""
Test DuckDB integration with mock data to verify the pipeline works.
"""
import sys
from pathlib import Path
from datetime import datetime

# Add synth_ai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    CAISEvent, EnvironmentEvent, SessionMetadum
)
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor


def create_mock_session():
    """Create a mock session with DuckDB storage."""
    print("Creating mock session with DuckDB storage...")
    
    # Create tracer with DuckDB
    tracer = SessionTracer(
        traces_dir="test_traces",
        duckdb_path="test_mock_traces.duckdb"
    )
    
    # Start session
    session = tracer.start_session("mock-crafter-session-001")
    print(f"Started session: {session.session_id}")
    
    # Add metadata
    tracer.add_session_metadata("episode_config", {
        "instance_num": 0,
        "task_instance_id": "test-123",
        "difficulty": "easy",
        "max_turns": 20,
        "model": "gpt-3.5-turbo"
    })
    
    # Simulate multiple timesteps
    total_reward = 0
    for i in range(5):
        timestep = tracer.start_timestep(f"step-{i}")
        
        # User message
        user_msg = SessionEventMessage(
            content={
                "role": "user",
                "text": f"Current state: Health: {20-i}, Inventory: wood={i}, Position: ({i}, {i})"
            },
            message_type="user_input",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i * 2
            )
        )
        tracer.record_message(user_msg)
        
        # Simulate CAIS event (LLM call)
        cais_event = CAISEvent(
            system_instance_id="llm-agent",
            model_name="gpt-3.5-turbo",
            prompt_tokens=150 + i * 10,
            completion_tokens=50 + i * 5,
            total_tokens=200 + i * 15,
            cost=0.001 * (i + 1),
            latency_ms=100 + i * 20,
            system_state_before={"thinking": True},
            system_state_after={"action_chosen": f"collect_wood_{i}"},
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i * 2
            )
        )
        tracer.record_event(cais_event)
        
        # Assistant response
        assistant_msg = SessionEventMessage(
            content={
                "role": "assistant",
                "text": f"I'll collect wood at position ({i}, {i}). This will help me craft tools later."
            },
            message_type="assistant_response",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i * 2 + 1
            )
        )
        tracer.record_message(assistant_msg)
        
        # Environment event
        reward = 1.0 if i > 2 else 0.0  # Reward for later actions
        total_reward += reward
        env_event = EnvironmentEvent(
            system_instance_id="crafter-env",
            reward=reward,
            terminated=False,
            system_state_before={"inventory": {"wood": i}},
            system_state_after={"inventory": {"wood": i + 1}},
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i * 2 + 1
            )
        )
        tracer.record_event(env_event)
    
    # End session
    print(f"Ending session with total reward: {total_reward}")
    tracer.end_session(save=True, upload_to_db=True)
    tracer.close()
    
    return total_reward


def test_filtering():
    """Test filtering the mock data."""
    print("\n" + "="*50)
    print("Testing DuckDB filtering")
    print("="*50)
    
    db_path = "test_mock_traces.duckdb"
    
    with DuckDBTraceManager(db_path) as db:
        # Check data was saved
        sessions = db.conn.execute("SELECT * FROM session_traces").df()
        print(f"\nSessions in DB: {len(sessions)}")
        
        events = db.conn.execute("SELECT * FROM events").df()
        print(f"Events in DB: {len(events)}")
        
        messages = db.conn.execute("SELECT * FROM messages").df()
        print(f"Messages in DB: {len(messages)}")
        
        # Check model usage
        model_usage = db.get_model_usage()
        if not model_usage.empty:
            print("\nModel usage:")
            for _, row in model_usage.iterrows():
                print(f"  {row['model_name']}: {row['call_count']} calls, "
                      f"{row['total_tokens']} tokens, ${row['total_cost']:.4f}")
    
    # Test extraction
    print("\n" + "="*50)
    print("Testing data extraction")
    print("="*50)
    
    with FinetuningDataExtractor(db_path) as extractor:
        # Get successful sessions
        successful = extractor.get_successful_sessions(min_reward=0.0)
        print(f"\nSuccessful sessions: {len(successful)}")
        
        if not successful.empty:
            session_id = successful.iloc[0]['session_id']
            print(f"Session reward: {successful.iloc[0]['total_reward']}")
            
            # Extract OpenAI format
            openai_data = extractor.extract_openai_format(
                session_ids=[session_id],
                min_reward=0.0
            )
            
            print(f"\nExtracted {len(openai_data)} training examples")
            
            if openai_data:
                example = openai_data[0]
                print(f"Messages in example: {len(example['messages'])}")
                print("\nFirst 3 messages:")
                for msg in example['messages'][:3]:
                    print(f"  [{msg['role']}]: {msg['content'][:80]}...")
                
                # Save to file
                output_file = "test_training_data.jsonl"
                with open(output_file, 'w') as f:
                    import json
                    for ex in openai_data:
                        f.write(json.dumps(ex) + '\n')
                print(f"\n✅ Saved training data to: {output_file}")


def main():
    """Run the full test."""
    # Create mock session
    reward = create_mock_session()
    
    # Test filtering
    test_filtering()
    
    # Run the filter script
    print("\n" + "="*50)
    print("Testing filter script")
    print("="*50)
    
    import subprocess
    result = subprocess.run([
        sys.executable,
        "filter_traces_sft_duckdb.py",
        "-d", "test_mock_traces.duckdb",
        "-o", "test_filtered_training.jsonl",
        "--min-reward", "0.0"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Filter script ran successfully!")
        print("\nOutput:")
        print(result.stdout)
    else:
        print("❌ Filter script failed!")
        print("\nError:")
        print(result.stderr)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()