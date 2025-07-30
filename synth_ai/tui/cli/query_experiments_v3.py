#!/usr/bin/env python3
"""
Query experiments and sessions from Turso/sqld using v3 tracing.
"""

import argparse
import asyncio
from typing import Optional
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager
import pandas as pd


async def list_experiments(db_url: str):
    """List all experiments in the database."""
    db = AsyncSQLTraceManager(db_url)
    await db.initialize()

    try:
        df = await db.query_traces("""
            SELECT 
                e.experiment_id,
                e.name,
                e.description,
                e.created_at,
                COUNT(DISTINCT st.session_id) as num_sessions,
                COUNT(DISTINCT ev.id) as num_events,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
            FROM experiments e
            LEFT JOIN session_traces st ON e.experiment_id = st.experiment_id
            LEFT JOIN events ev ON st.session_id = ev.session_id
            GROUP BY e.experiment_id, e.name, e.description, e.created_at
            ORDER BY e.created_at DESC
        """)

        if df.empty:
            print("No experiments found in database.")
            return

        print(f"\n{'=' * 100}")
        print(f"{'Experiments in ' + db_url:^100}")
        print(f"{'=' * 100}\n")

        for _, row in df.iterrows():
            print(f"🧪 {row['name']} (id: {row['experiment_id'][:8]}...)")
            print(f"   Created: {row['created_at']}")
            print(f"   Description: {row['description']}")
            print(f"   Sessions: {row['num_sessions']}")
            print(f"   Events: {row['num_events']:,}")
            if row["total_cost"] and row["total_cost"] > 0:
                print(f"   Cost: ${row['total_cost']:.4f}")
            if row["total_tokens"] and row["total_tokens"] > 0:
                print(f"   Tokens: {int(row['total_tokens']):,}")
            print()
    finally:
        await db.close()


async def show_experiment_details(db_url: str, experiment_id: str):
    """Show detailed information about a specific experiment."""
    db = AsyncSQLTraceManager(db_url)
    await db.initialize()

    try:
        # Get experiment info
        exp_df = await db.query_traces(
            """
            SELECT * FROM experiments WHERE experiment_id LIKE :exp_id
        """,
            {"exp_id": f"{experiment_id}%"},
        )

        if exp_df.empty:
            print(f"No experiment found matching ID: {experiment_id}")
            return

        exp = exp_df.iloc[0]
        print(f"\n{'=' * 100}")
        print(f"Experiment: {exp['name']} ({exp['experiment_id']})")
        print(f"{'=' * 100}\n")

        # Get session statistics
        sessions_df = await db.get_sessions_by_experiment(exp["experiment_id"])

        if sessions_df:
            print(f"Sessions: {len(sessions_df)}")

            # Get aggregated stats
            stats_df = await db.query_traces(
                """
                SELECT 
                    COUNT(DISTINCT ev.id) as total_events,
                    COUNT(DISTINCT m.id) as total_messages,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                    SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
                FROM session_traces st
                LEFT JOIN events ev ON st.session_id = ev.session_id
                LEFT JOIN messages m ON st.session_id = m.session_id
                WHERE st.experiment_id = :exp_id
            """,
                {"exp_id": exp["experiment_id"]},
            )

            if not stats_df.empty:
                stats = stats_df.iloc[0]
                print(f"Total events: {int(stats['total_events']):,}")
                print(f"Total messages: {int(stats['total_messages']):,}")
                print(f"Total cost: ${stats['total_cost']:.4f}")
                print(f"Total tokens: {int(stats['total_tokens']):,}")

            # Show session list
            print("\nSession list:")
            for sess in sessions_df:
                print(f"  - {sess['session_id']} ({sess['created_at']})")
                print(
                    f"    Timesteps: {sess['num_timesteps']}, Events: {sess['num_events']}, Messages: {sess['num_messages']}"
                )
    finally:
        await db.close()


async def show_model_usage(db_url: str, model_name: Optional[str] = None):
    """Show model usage statistics."""
    db = AsyncSQLTraceManager(db_url)
    await db.initialize()

    try:
        df = await db.get_model_usage(model_name=model_name)

        if df.empty:
            print("No model usage data found.")
            return

        print(f"\n{'=' * 100}")
        print(f"{'Model Usage Statistics':^100}")
        print(f"{'=' * 100}\n")

        print(df.to_string(index=False))
    finally:
        await db.close()


async def main():
    parser = argparse.ArgumentParser(description="Query experiments from Turso/sqld (v3)")
    parser.add_argument(
        "-u", "--url", default="sqlite+libsql://http://127.0.0.1:8080", help="Turso database URL"
    )
    parser.add_argument(
        "-e", "--experiment", help="Show details for specific experiment ID (can be partial)"
    )
    parser.add_argument("-m", "--model", help="Show usage for specific model")
    parser.add_argument("--usage", action="store_true", help="Show model usage statistics")

    args = parser.parse_args()

    if args.usage or args.model:
        await show_model_usage(args.url, args.model)
    elif args.experiment:
        await show_experiment_details(args.url, args.experiment)
    else:
        await list_experiments(args.url)


if __name__ == "__main__":
    asyncio.run(main())
