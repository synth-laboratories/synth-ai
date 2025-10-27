#!/usr/bin/env python3
"""
Query experiments and sessions from DuckDB.
"""
import argparse
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
import pandas as pd


def list_experiments(db_path: str):
    """List all experiments in the database."""
    with DuckDBTraceManager(db_path) as db:
        df = db.conn.execute("""
            SELECT 
                e.id,
                e.name,
                e.description,
                e.created_at,
                COUNT(DISTINCT st.session_id) as num_sessions,
                COUNT(DISTINCT ev.id) as num_events,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost ELSE 0 END) as total_cost,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
            FROM experiments e
            LEFT JOIN session_traces st ON e.id = st.experiment_id
            LEFT JOIN events ev ON st.session_id = ev.session_id
            GROUP BY e.id, e.name, e.description, e.created_at
            ORDER BY e.created_at DESC
        """).df()
        
        if df.empty:
            print("No experiments found in database.")
            return
        
        print(f"\n{'='*100}")
        print(f"{'Experiments in ' + db_path:^100}")
        print(f"{'='*100}\n")
        
        for _, row in df.iterrows():
            print(f"🧪 {row['name']} (id: {row['id'][:8]}...)")
            print(f"   Created: {row['created_at']}")
            print(f"   Description: {row['description']}")
            print(f"   Sessions: {row['num_sessions']}")
            print(f"   Events: {row['num_events']:,}")
            if row['total_cost'] and row['total_cost'] > 0:
                print(f"   Cost: ${row['total_cost']:.4f}")
            if row['total_tokens'] and row['total_tokens'] > 0:
                print(f"   Tokens: {row['total_tokens']:,}")
            print()


def show_experiment_details(db_path: str, experiment_id: str):
    """Show detailed information about a specific experiment."""
    with DuckDBTraceManager(db_path) as db:
        # Get experiment info
        exp_df = db.conn.execute("""
            SELECT * FROM experiments WHERE id LIKE ?
        """, [f"{experiment_id}%"]).df()
        
        if exp_df.empty:
            print(f"No experiment found matching ID: {experiment_id}")
            return
        
        exp = exp_df.iloc[0]
        print(f"\n{'='*100}")
        print(f"Experiment: {exp['name']} ({exp['id']})")
        print(f"{'='*100}\n")
        
        # Get system info
        sys_df = db.conn.execute("""
            SELECT 
                s.name as system_name,
                s.description as system_desc,
                sv.branch,
                sv.commit,
                sv.created_at as version_created
            FROM experimental_systems es
            JOIN systems s ON es.system_id = s.id
            JOIN system_versions sv ON es.system_version_id = sv.id
            WHERE es.experiment_id = ?
        """, [exp['id']]).df()
        
        if not sys_df.empty:
            print("Systems:")
            for _, sys in sys_df.iterrows():
                print(f"  - {sys['system_name']}: {sys['system_desc']}")
                print(f"    Branch: {sys['branch']} @ {sys['commit'][:8]}")
                print(f"    Version created: {sys['version_created']}")
            print()
        
        # Get session statistics
        sessions_df = db.get_experiment_sessions(exp['id'])
        
        if not sessions_df.empty:
            print(f"Sessions: {len(sessions_df)}")
            print(f"Total events: {sessions_df['total_events'].sum():,}")
            print(f"Total messages: {sessions_df['total_messages'].sum():,}")
            print(f"Total cost: ${sessions_df['total_cost'].sum():.4f}")
            print(f"Total tokens: {sessions_df['total_tokens'].sum():,}")
            
            # Show session list
            print("\nSession list:")
            for _, sess in sessions_df.iterrows():
                print(f"  - {sess['session_id']} ({sess['created_at']})")
                print(f"    Events: {sess['total_events']}, Messages: {sess['total_messages']}")
                if sess['total_cost'] > 0:
                    print(f"    Cost: ${sess['total_cost']:.4f}, Tokens: {sess['total_tokens']:,}")


def main():
    parser = argparse.ArgumentParser(description="Query experiments from DuckDB")
    parser.add_argument("-d", "--db", default="crafter_traces.duckdb", help="DuckDB database path")
    parser.add_argument("-e", "--experiment", help="Show details for specific experiment ID (can be partial)")
    
    args = parser.parse_args()
    
    if args.experiment:
        show_experiment_details(args.db, args.experiment)
    else:
        list_experiments(args.db)


if __name__ == "__main__":
    main()