"""
Utilities for extracting fine-tuning data from DuckDB traces.
"""
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
from pathlib import Path

from .manager import DuckDBTraceManager


class FinetuningDataExtractor:
    """Extract and prepare fine-tuning data from DuckDB traces."""
    
    def __init__(self, db_path: str):
        """Initialize with DuckDB path."""
        self.db_manager = DuckDBTraceManager(db_path)
    
    def close(self):
        """Close database connection."""
        self.db_manager.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_successful_sessions(self, min_reward: float = 0.0) -> pd.DataFrame:
        """Get sessions with positive rewards."""
        query = """
            SELECT DISTINCT s.session_id, s.created_at, s.metadata,
                   SUM(e.reward) as total_reward,
                   COUNT(DISTINCT e.id) as env_events
            FROM session_traces s
            JOIN events e ON s.session_id = e.session_id
            WHERE e.event_type = 'environment' AND e.reward IS NOT NULL
            GROUP BY s.session_id, s.created_at, s.metadata
            HAVING SUM(e.reward) > ?
            ORDER BY total_reward DESC
        """
        return self.db_manager.query_traces(query, [min_reward])
    
    def get_session_conversations(self, session_id: str) -> List[Dict[str, Any]]:
        """Extract conversation history from a session."""
        # Get all messages in chronological order
        query = """
            SELECT message_type, content, message_time, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY message_time, timestamp
        """
        messages_df = self.db_manager.query_traces(query, [session_id])
        
        conversations = []
        for _, msg in messages_df.iterrows():
            content = json.loads(msg['content']) if isinstance(msg['content'], str) else msg['content']
            
            # Extract role and text based on message type
            if msg['message_type'] == 'SessionEventMessage':
                # Parse the actual content
                actual_content = content.get('content', {})
                if isinstance(actual_content, dict):
                    # Check for different content structures
                    if 'role' in actual_content and 'text' in actual_content:
                        # Standard chat format
                        role = actual_content.get('role', 'user')
                        text = actual_content.get('text', '')
                    elif 'payload' in actual_content:
                        # Crafter format - extract meaningful info from payload
                        payload = actual_content['payload']
                        origin = actual_content.get('origin_system_id', '')
                        
                        # Determine role based on origin system
                        if 'inventory' in str(payload):
                            role = 'user'  # Environment state
                            # Format the state info
                            if isinstance(payload, dict) and 'inventory' in payload:
                                inv = payload['inventory']
                                text = f"Current state: Health={inv.get('health', 0)}, Food={inv.get('food', 0)}, Inventory={payload.get('entities', {})}"
                            else:
                                text = f"State: {json.dumps(payload)[:200]}"
                        elif isinstance(payload, list) and payload and 'tool' in payload[0]:
                            role = 'assistant'  # Agent action
                            # Extract action from tool calls
                            actions = []
                            for tool_call in payload:
                                if tool_call.get('tool') == 'interact' and 'actions' in tool_call.get('args', {}):
                                    actions.extend(tool_call['args']['actions'])
                            text = f"I'll perform actions: {', '.join(actions)}" if actions else str(payload)
                        else:
                            # Skip other message types for training
                            continue
                    else:
                        # Skip messages without useful content
                        continue
                else:
                    # Skip non-dict content
                    continue
                
                if text:  # Only add messages with actual text
                    conversations.append({
                        'role': role,
                        'content': text,
                        'message_time': msg['message_time']
                    })
        
        return conversations
    
    def get_llm_calls_for_session(self, session_id: str) -> pd.DataFrame:
        """Get all LLM calls for a session."""
        query = """
            SELECT model_name, prompt_tokens, completion_tokens, 
                   total_tokens, cost, latency_ms, message_time,
                   system_state_before, system_state_after
            FROM events
            WHERE session_id = ? AND event_type = 'cais'
            ORDER BY message_time
        """
        return self.db_manager.query_traces(query, [session_id])
    
    def extract_openai_format(self, session_ids: Optional[List[str]] = None,
                            min_reward: float = 0.0) -> List[Dict[str, Any]]:
        """Extract training data in OpenAI fine-tuning format."""
        if session_ids is None:
            # Get all successful sessions
            sessions = self.get_successful_sessions(min_reward)
            session_ids = sessions['session_id'].tolist()
        
        training_examples = []
        
        for session_id in session_ids:
            conversations = self.get_session_conversations(session_id)
            
            if not conversations:
                continue
            
            # Group into prompt-response pairs
            messages = []
            for conv in conversations:
                if conv['role'] in ['system', 'user', 'assistant']:
                    messages.append({
                        'role': conv['role'],
                        'content': conv['content']
                    })
            
            if messages:
                training_examples.append({
                    'messages': messages,
                    'metadata': {
                        'session_id': session_id,
                        'source': 'duckdb_traces'
                    }
                })
        
        return training_examples
    
    def extract_gemini_format(self, session_ids: Optional[List[str]] = None,
                            min_reward: float = 0.0) -> List[Dict[str, Any]]:
        """Extract training data in Gemini/Vertex AI format."""
        if session_ids is None:
            sessions = self.get_successful_sessions(min_reward)
            session_ids = sessions['session_id'].tolist()
        
        training_examples = []
        
        for session_id in session_ids:
            conversations = self.get_session_conversations(session_id)
            
            if not conversations:
                continue
            
            # Gemini format uses text_input and output
            current_input = []
            
            for i, conv in enumerate(conversations):
                if conv['role'] in ['system', 'user']:
                    current_input.append(conv['content'])
                elif conv['role'] == 'assistant' and current_input:
                    training_examples.append({
                        'text_input': '\n'.join(current_input),
                        'output': conv['content']
                    })
                    current_input = []
        
        return training_examples
    
    def filter_by_achievements(self, min_achievements: int = 1) -> List[str]:
        """Get session IDs that achieved certain milestones."""
        query = """
            SELECT DISTINCT s.session_id
            FROM session_traces s
            JOIN events e ON s.session_id = e.session_id
            WHERE e.event_type = 'environment' 
            AND e.system_state_after IS NOT NULL
            AND (
                json_extract(e.system_state_after, '$.inventory.wood') > 0
                OR json_extract(e.system_state_after, '$.inventory.stone') > 0
                OR json_extract(e.system_state_after, '$.inventory.coal') > 0
                OR json_extract(e.system_state_after, '$.inventory.iron') > 0
            )
        """
        result = self.db_manager.query_traces(query)
        return result['session_id'].tolist() if not result.empty else []
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a session."""
        # Get basic session info
        session_query = """
            SELECT * FROM session_traces WHERE session_id = ?
        """
        session = self.db_manager.query_traces(session_query, [session_id])
        
        # Get rewards
        reward_query = """
            SELECT SUM(reward) as total_reward, 
                   COUNT(*) as reward_events,
                   MAX(reward) as max_reward
            FROM events
            WHERE session_id = ? AND event_type = 'environment' AND reward IS NOT NULL
        """
        rewards = self.db_manager.query_traces(reward_query, [session_id])
        
        # Get LLM usage
        llm_query = """
            SELECT COUNT(*) as llm_calls,
                   SUM(total_tokens) as total_tokens,
                   SUM(cost) as total_cost,
                   AVG(latency_ms) as avg_latency
            FROM events
            WHERE session_id = ? AND event_type = 'cais'
        """
        llm_stats = self.db_manager.query_traces(llm_query, [session_id])
        
        metrics = {
            'session_id': session_id,
            'num_timesteps': session.iloc[0]['num_timesteps'] if not session.empty else 0,
            'num_events': session.iloc[0]['num_events'] if not session.empty else 0,
            'total_reward': rewards.iloc[0]['total_reward'] if not rewards.empty else 0,
            'max_reward': rewards.iloc[0]['max_reward'] if not rewards.empty else 0,
            'llm_calls': llm_stats.iloc[0]['llm_calls'] if not llm_stats.empty else 0,
            'total_tokens': llm_stats.iloc[0]['total_tokens'] if not llm_stats.empty else 0,
            'total_cost': llm_stats.iloc[0]['total_cost'] if not llm_stats.empty else 0,
            'avg_latency_ms': llm_stats.iloc[0]['avg_latency'] if not llm_stats.empty else 0
        }
        
        return metrics
    
    def export_filtered_traces(self, output_dir: str, 
                             min_reward: float = 0.0,
                             format: str = 'openai') -> int:
        """Export filtered traces to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get successful sessions
        sessions = self.get_successful_sessions(min_reward)
        
        if format == 'openai':
            data = self.extract_openai_format(
                session_ids=sessions['session_id'].tolist(),
                min_reward=min_reward
            )
            filename = 'training_data_openai.jsonl'
            
            with open(output_path / filename, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
                    
        elif format == 'gemini':
            data = self.extract_gemini_format(
                session_ids=sessions['session_id'].tolist(),
                min_reward=min_reward
            )
            filename = 'training_data_gemini.jsonl'
            
            with open(output_path / filename, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
        
        # Also export metadata
        metadata = []
        for session_id in sessions['session_id'].tolist():
            metrics = self.get_session_metrics(session_id)
            metadata.append(metrics)
        
        with open(output_path / 'session_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(data)