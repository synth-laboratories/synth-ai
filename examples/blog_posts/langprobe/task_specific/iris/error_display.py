#!/usr/bin/env python3
"""Rich error display utilities for prompt learning jobs."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


def format_error_message(
    job_detail: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]] = None,
    job_id: str = "",
) -> str:
    """Format a rich, detailed error message from job details and events.
    
    Args:
        job_detail: Job detail dict from API
        events: List of job events (optional)
        job_id: Job ID for context
    
    Returns:
        Formatted error message string
    """
    lines = []
    
    # Extract error from multiple possible fields
    error_message = (
        job_detail.get("error_message") or
        job_detail.get("error") or
        job_detail.get("failure_reason") or
        "Unknown error"
    )
    
    lines.append("=" * 80)
    lines.append("❌ JOB FAILED")
    lines.append("=" * 80)
    
    if job_id:
        lines.append(f"Job ID: {job_id}")
    
    # Status and timing
    status = job_detail.get("status", "unknown")
    lines.append(f"Status: {status}")
    
    created_at = job_detail.get("created_at")
    started_at = job_detail.get("started_at")
    finished_at = job_detail.get("finished_at")
    
    if created_at:
        lines.append(f"Created: {created_at}")
    if started_at:
        lines.append(f"Started: {started_at}")
    if finished_at:
        lines.append(f"Finished: {finished_at}")
        if started_at:
            try:
                start = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                finish = datetime.fromisoformat(finished_at.replace('Z', '+00:00'))
                duration = (finish - start).total_seconds()
                lines.append(f"Duration: {duration:.1f}s")
            except Exception:
                pass
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("ERROR MESSAGE")
    lines.append("-" * 80)
    lines.append(error_message)
    
    # Try to extract more details from error message if it's JSON-like
    if isinstance(error_message, str) and ("{" in error_message or "[" in error_message):
        try:
            # Try to parse as JSON if it looks like JSON
            if error_message.strip().startswith(("{", "[")):
                error_json = json.loads(error_message)
                if isinstance(error_json, dict):
                    lines.append("\nParsed error details:")
                    for key, value in error_json.items():
                        if key not in ("error", "message", "detail"):  # Already shown
                            lines.append(f"  {key}: {value}")
        except Exception:
            pass
    
    # Show events if available
    if events:
        error_events = [
            e for e in events
            if (
                e.get("type", "").endswith(".failed") or
                e.get("type", "").endswith(".error") or
                "error" in str(e.get("message", "")).lower() or
                "failed" in str(e.get("message", "")).lower()
            )
        ]
        
        if error_events:
            lines.append("")
            lines.append("-" * 80)
            lines.append("ERROR EVENTS")
            lines.append("-" * 80)
            for ev in error_events[-10:]:  # Show last 10 error events
                ev_type = ev.get("type", "unknown")
                ev_msg = ev.get("message", "")
                ev_data = ev.get("data")
                ev_time = ev.get("timestamp", "")
                
                lines.append(f"\n[{ev_type}]")
                if ev_time:
                    lines.append(f"  Time: {ev_time}")
                lines.append(f"  Message: {ev_msg}")
                
                if ev_data:
                    lines.append("  Data:")
                    if isinstance(ev_data, dict):
                        for key, value in ev_data.items():
                            # Truncate long values
                            val_str = str(value)
                            if len(val_str) > 200:
                                val_str = val_str[:200] + "..."
                            lines.append(f"    {key}: {val_str}")
                    else:
                        val_str = str(ev_data)
                        if len(val_str) > 500:
                            val_str = val_str[:500] + "..."
                        lines.append(f"    {val_str}")
        
        # Show recent non-error events for context
        recent_events = [e for e in events if e not in error_events][-5:]
        if recent_events:
            lines.append("")
            lines.append("-" * 80)
            lines.append("RECENT EVENTS (for context)")
            lines.append("-" * 80)
            for ev in recent_events:
                ev_type = ev.get("type", "unknown")
                ev_msg = ev.get("message", "")
                ev_time = ev.get("timestamp", "")
                lines.append(f"  [{ev_type}] {ev_msg}")
                if ev_time:
                    lines.append(f"    Time: {ev_time}")
    
    # Show job metadata if available
    metadata = job_detail.get("metadata")
    if metadata:
        lines.append("")
        lines.append("-" * 80)
        lines.append("JOB METADATA")
        lines.append("-" * 80)
        for key, value in metadata.items():
            if key not in ("error", "error_message"):  # Already shown
                val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                lines.append(f"  {key}: {val_str}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


async def fetch_and_display_error(
    client: Any,
    job_id: str,
    backend_url: str = "",
) -> None:
    """Fetch job details and events, then display rich error message.
    
    Args:
        client: PromptLearningClient instance
        job_id: Job ID to fetch
        backend_url: Backend URL for context
    """
    try:
        # Fetch job detail
        job_detail = await client.get_job(job_id)
        
        # Fetch events
        events = None
        try:
            events = await client.get_events(job_id, limit=50)
        except Exception as ev_err:
            print(f"⚠️  Could not fetch events: {ev_err}")
        
        # Format and display
        error_msg = format_error_message(job_detail, events, job_id)
        print(error_msg)
        
        # Additional helpful context
        if backend_url:
            print(f"\n💡 Tip: Check backend logs at {backend_url} for more details")
        
    except Exception as e:
        print(f"\n❌ Failed to fetch error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

