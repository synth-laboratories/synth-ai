#!/usr/bin/env python3
"""
Compare experiments between gpt-4.1-nano and gpt-4.1-mini.
Analyzes performance differences, achievement patterns, and instance difficulty.
"""

import duckdb
import pandas as pd
from typing import Dict, List, Any
import json

# Experiment IDs from the runs
EXPERIMENTS = {
    "gpt-4o-mini": "137683ed-3bd5-4bd3-9162-dae0371ddd3d",
    "gpt-4o": "207307d5-4105-4a18-bb93-89936047fa18"
}

def connect_to_db():
    """Connect to the DuckDB database."""
    return duckdb.connect("synth_ai/traces/crafter_traces.duckdb")

def get_experiment_summary(conn, experiment_id: str) -> Dict[str, Any]:
    """Get basic experiment information."""
    query = """
    SELECT 
        e.id,
        e.name,
        e.description,
        e.created_at,
        sv.branch,
        sv.commit
    FROM experiments e
    LEFT JOIN experimental_systems es ON e.id = es.experiment_id
    LEFT JOIN system_versions sv ON es.system_version_id = sv.id
    WHERE e.id = ?
    """
    
    result = conn.execute(query, [experiment_id]).fetchone()
    if result:
        return {
            "experiment_id": result[0],
            "name": result[1],
            "description": result[2],
            "created_at": result[3],
            "branch": result[4],
            "commit": result[5]
        }
    return None

def get_session_stats(conn, experiment_id: str) -> pd.DataFrame:
    """Get session-level statistics for an experiment."""
    query = """
    SELECT 
        st.session_id,
        st.created_at,
        st.num_timesteps,
        st.num_events,
        st.num_messages,
        st.metadata
    FROM session_traces st
    WHERE st.experiment_id = ?
    ORDER BY st.created_at
    """
    
    return conn.execute(query, [experiment_id]).df()

def get_achievement_analysis(conn, experiment_id: str) -> Dict[str, Any]:
    """Analyze achievements for an experiment."""
    # Get session traces with achievement data
    query = """
    SELECT 
        st.session_id,
        st.metadata
    FROM session_traces st
    WHERE st.experiment_id = ? 
    AND st.metadata IS NOT NULL
    """
    
    results = conn.execute(query, [experiment_id]).fetchall()
    
    all_achievements = []
    session_achievements = []
    
    for session_id, metadata in results:
        if metadata:
            # Parse the JSON metadata
            try:
                import json
                metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
                
                for meta_item in metadata_list:
                    if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                        data = meta_item.get('data', {})
                        if 'achievements' in data:
                            achievements_dict = data['achievements']
                            num_achievements = data.get('num_achievements', 0)
                            
                            # Extract unlocked achievements
                            unlocked = [ach for ach, unlocked in achievements_dict.items() if unlocked]
                            all_achievements.extend(unlocked)
                            
                            session_achievements.append({
                                'session_id': session_id,
                                'num_achievements': num_achievements,
                                'unlocked_achievements': unlocked,
                                'total_achievements': len(achievements_dict)
                            })
            except Exception as e:
                print(f"Error parsing metadata for session {session_id}: {e}")
    
    # Count achievements
    achievement_counts = {}
    for ach in all_achievements:
        achievement_counts[ach] = achievement_counts.get(ach, 0) + 1
    
    return {
        "total_achievements": len(all_achievements),
        "unique_achievements": len(set(all_achievements)),
        "achievement_counts": achievement_counts,
        "achievement_list": all_achievements,
        "session_achievements": session_achievements
    }

def get_model_usage_analysis(conn, experiment_id: str) -> pd.DataFrame:
    """Analyze model usage and costs."""
    query = """
    SELECT 
        e.model_name,
        e.provider,
        COUNT(*) as call_count,
        SUM(e.prompt_tokens) as total_prompt_tokens,
        SUM(e.completion_tokens) as total_completion_tokens,
        SUM(e.total_tokens) as total_tokens,
        SUM(e.cost) as total_cost,
        AVG(e.latency_ms) as avg_latency_ms,
        AVG(e.prompt_tokens) as avg_prompt_tokens,
        AVG(e.completion_tokens) as avg_completion_tokens
    FROM session_traces st
    JOIN events e ON st.session_id = e.session_id
    WHERE st.experiment_id = ? 
    AND e.event_type = 'lm_cais'
    GROUP BY e.model_name, e.provider
    """
    
    return conn.execute(query, [experiment_id]).df()

def get_session_performance_comparison(conn) -> pd.DataFrame:
    """Compare session performance between experiments."""
    query = """
    SELECT 
        st.experiment_id,
        e.name as experiment_name,
        COUNT(st.session_id) as total_sessions,
        AVG(st.num_timesteps) as avg_timesteps,
        AVG(st.num_events) as avg_events,
        AVG(st.num_messages) as avg_messages,
        SUM(st.num_timesteps) as total_timesteps,
        SUM(st.num_events) as total_events,
        SUM(st.num_messages) as total_messages
    FROM session_traces st
    JOIN experiments e ON st.experiment_id = e.id
    WHERE st.experiment_id IN (?, ?)
    GROUP BY st.experiment_id, e.name
    ORDER BY e.name
    """
    
    return conn.execute(query, [EXPERIMENTS["gpt-4o-mini"], EXPERIMENTS["gpt-4o"]]).df()

def get_achievement_comparison(conn) -> pd.DataFrame:
    """Compare achievements between experiments."""
    # This is a more complex query to extract achievements from metadata
    query = """
    WITH achievement_data AS (
        SELECT 
            st.experiment_id,
            e.name as experiment_name,
            st.session_id,
            e.metadata,
            e.event_metadata
        FROM session_traces st
        JOIN experiments e ON st.experiment_id = e.experiment_id
        JOIN events ev ON st.session_id = ev.session_id
        WHERE st.experiment_id IN (?, ?)
        AND ev.event_type = 'environment'
        AND ev.metadata IS NOT NULL
    )
    SELECT 
        experiment_id,
        experiment_name,
        COUNT(DISTINCT session_id) as sessions_with_achievements,
        COUNT(*) as total_achievement_events
    FROM achievement_data
    GROUP BY experiment_id, experiment_name
    """
    
    return conn.execute(query, [EXPERIMENTS["gpt-4o-mini"], EXPERIMENTS["gpt-4o"]]).df()

def analyze_instance_difficulty(conn) -> Dict[str, Any]:
    """Analyze which instances were more difficult for each model."""
    query = """
    SELECT 
        st.experiment_id,
        e.name as experiment_name,
        st.session_id,
        st.num_timesteps,
        st.num_events,
        st.metadata
    FROM session_traces st
    JOIN experiments e ON st.experiment_id = e.id
    WHERE st.experiment_id IN (?, ?)
    ORDER BY st.experiment_id, st.session_id
    """
    
    df = conn.execute(query, [EXPERIMENTS["gpt-4o-mini"], EXPERIMENTS["gpt-4o"]]).df()
    
    # Group by experiment and analyze session patterns
    analysis = {}
    for experiment_id in [EXPERIMENTS["gpt-4o-mini"], EXPERIMENTS["gpt-4o"]]:
        exp_data = df[df['experiment_id'] == experiment_id]
        analysis[experiment_id] = {
            "total_sessions": len(exp_data),
            "avg_timesteps": exp_data['num_timesteps'].mean(),
            "avg_events": exp_data['num_events'].mean(),
            "max_timesteps": exp_data['num_timesteps'].max(),
            "min_timesteps": exp_data['num_timesteps'].min(),
            "session_lengths": exp_data['num_timesteps'].tolist()
        }
    
    return analysis

def main():
    """Main analysis function."""
    print("🔍 COMPARING GPT-4O-MINI vs GPT-4O EXPERIMENTS")
    print("=" * 80)
    
    conn = connect_to_db()
    
    # Get experiment summaries
    print("\n📋 EXPERIMENT SUMMARIES")
    print("-" * 40)
    
    for model_name, exp_id in EXPERIMENTS.items():
        summary = get_experiment_summary(conn, exp_id)
        if summary:
            print(f"\n{model_name.upper()}:")
            print(f"  Name: {summary['name']}")
            print(f"  ID: {summary['experiment_id']}")
            print(f"  Created: {summary['created_at']}")
            print(f"  Git: {summary['branch']} @ {summary['commit'][:8]}")
    
    # Session performance comparison
    print("\n📊 SESSION PERFORMANCE COMPARISON")
    print("-" * 40)
    
    perf_df = get_session_performance_comparison(conn)
    print(perf_df.to_string(index=False))
    
    # Achievement analysis
    print("\n🏆 ACHIEVEMENT ANALYSIS")
    print("-" * 40)
    
    for model_name, exp_id in EXPERIMENTS.items():
        print(f"\n{model_name.upper()}:")
        achievement_data = get_achievement_analysis(conn, exp_id)
        print(f"  Total Achievements: {achievement_data['total_achievements']}")
        print(f"  Unique Achievements: {achievement_data['unique_achievements']}")
        print(f"  Achievement Counts: {achievement_data['achievement_counts']}")
    
    # Model usage analysis
    print("\n💰 MODEL USAGE ANALYSIS")
    print("-" * 40)
    
    for model_name, exp_id in EXPERIMENTS.items():
        print(f"\n{model_name.upper()}:")
        usage_df = get_model_usage_analysis(conn, exp_id)
        if not usage_df.empty:
            print(usage_df.to_string(index=False))
        else:
            print("  No model usage data found")
    
    # Instance difficulty analysis
    print("\n🎯 INSTANCE DIFFICULTY ANALYSIS")
    print("-" * 40)
    
    difficulty_analysis = analyze_instance_difficulty(conn)
    
    for model_name, exp_id in EXPERIMENTS.items():
        data = difficulty_analysis[exp_id]
        print(f"\n{model_name.upper()}:")
        print(f"  Total Sessions: {data['total_sessions']}")
        print(f"  Avg Timesteps: {data['avg_timesteps']:.1f}")
        print(f"  Avg Events: {data['avg_events']:.1f}")
        print(f"  Timestep Range: {data['min_timesteps']} - {data['max_timesteps']}")
    
    # Performance comparison summary
    print("\n📈 PERFORMANCE COMPARISON SUMMARY")
    print("-" * 40)
    
    mini_data = difficulty_analysis[EXPERIMENTS["gpt-4o-mini"]]
    full_data = difficulty_analysis[EXPERIMENTS["gpt-4o"]]
    
    print(f"GPT-4O-MINI:")
    print(f"  Sessions: {mini_data['total_sessions']}")
    print(f"  Avg Timesteps: {mini_data['avg_timesteps']:.1f}")
    print(f"  Avg Events: {mini_data['avg_events']:.1f}")
    
    print(f"\nGPT-4O:")
    print(f"  Sessions: {full_data['total_sessions']}")
    print(f"  Avg Timesteps: {full_data['avg_timesteps']:.1f}")
    print(f"  Avg Events: {full_data['avg_events']:.1f}")
    
    # Calculate improvements
    timestep_improvement = ((full_data['avg_timesteps'] - mini_data['avg_timesteps']) / mini_data['avg_timesteps']) * 100
    event_improvement = ((full_data['avg_events'] - mini_data['avg_events']) / mini_data['avg_events']) * 100
    
    print(f"\n📊 IMPROVEMENTS:")
    print(f"  Timesteps: {timestep_improvement:+.1f}%")
    print(f"  Events: {event_improvement:+.1f}%")
    
    conn.close()

if __name__ == "__main__":
    main() 