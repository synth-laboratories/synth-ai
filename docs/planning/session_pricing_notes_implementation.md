# Session Pricing Notes - Comprehensive Implementation Plan

## Executive Summary

This document provides a comprehensive implementation plan for adding pricing notes functionality to synth-ai sessions. Pricing notes allow users to document pricing-related information, model selection rationale, cost optimization strategies, or billing context for individual sessions.

**Key Decision**: Store pricing notes in `session_traces.metadata` JSON field (no schema migration required).

---

## Table of Contents

1. [Database Schema](#database-schema)
2. [New Code Files](#new-code-files)
3. [Modified Files](#modified-files)
4. [Data Flow & Architecture](#data-flow--architecture)
5. [Implementation Phases](#implementation-phases)
6. [Testing Strategy](#testing-strategy)
7. [Migration Considerations](#migration-considerations)
8. [API Contracts](#api-contracts)

---

## Database Schema

### Current Schema (No Changes Required)

**Table**: `session_traces`
- **Column**: `metadata` (JSONText) - Already exists, stores arbitrary JSON
- **Storage Format**: JSON stored as TEXT in SQLite/Turso
- **Current Usage**: Stores session-level metadata (user_id, experiment_id, model_config, etc.)

### Metadata Structure

```json
{
  "pricing_notes": "Used GPT-4o-mini for cost efficiency. Estimated 50% savings vs GPT-4o.",
  "pricing_strategy": "cost_optimized",
  "model_selection_rationale": "Switched from GPT-4o to GPT-4o-mini after initial tests showed comparable quality at 50% cost.",
  "cost_breakdown": {
    "estimated_savings_usd": 0.25,
    "baseline_model": "gpt-4o",
    "selected_model": "gpt-4o-mini"
  }
}
```

### Database Queries

**Extract pricing notes**:
```sql
SELECT 
    session_id,
    JSON_EXTRACT(metadata, '$.pricing_notes') as pricing_notes,
    JSON_EXTRACT(metadata, '$.pricing_strategy') as pricing_strategy,
    created_at
FROM session_traces
WHERE JSON_EXTRACT(metadata, '$.pricing_notes') IS NOT NULL;
```

**Update pricing notes**:
```sql
UPDATE session_traces
SET metadata = json_set(
    COALESCE(metadata, '{}'),
    '$.pricing_notes',
    'New pricing note here'
)
WHERE session_id = 'abc123';
```

**Index Considerations**:
- No new indexes needed (metadata is JSONText, not directly indexed)
- JSON extraction queries may be slower on large datasets
- Consider adding generated column or view if performance becomes an issue

---

## New Code Files

### 1. `synth_ai/cli/commands/session/__init__.py`

**Purpose**: Command group registration for session commands.

**Content**:
```python
"""Session management commands for synth-ai CLI."""

from __future__ import annotations

import click

from .note import note_command
from .show import show_command


def register(cli: click.Group) -> None:
    """Register session command group."""
    
    @cli.group(name="session", help="Manage session pricing notes and details")
    def session_group():
        """Session management commands."""
        pass
    
    session_group.add_command(note_command, name="note")
    session_group.add_command(show_command, name="show")
    
    # Register aliases
    cli.add_command(note_command, name="session-note")
    cli.add_command(show_command, name="session-show")
```

**Lines**: ~30

### 2. `synth_ai/cli/commands/session/note.py`

**Purpose**: Command to set/update pricing notes for a session.

**Content**:
```python
"""Command to set pricing notes for a session."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from synth_ai.cli._storage import load_storage


async def _set_pricing_note(
    db_url: str,
    session_id: str,
    notes: str,
    strategy: str | None = None,
    rationale: str | None = None,
) -> None:
    """Set pricing notes for a session."""
    create_storage, storage_config = load_storage()
    db = create_storage(storage_config(connection_string=db_url))
    await db.initialize()
    
    try:
        # Get current metadata
        result = await db.query_traces(
            "SELECT metadata FROM session_traces WHERE session_id = :session_id",
            {"session_id": session_id},
        )
        
        if result is None or result.empty:
            raise click.ClickException(f"Session {session_id} not found")
        
        # Parse existing metadata
        metadata_raw = result.iloc[0].get("metadata")
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(metadata_raw, dict):
            metadata = dict(metadata_raw)
        else:
            metadata = {}
        
        # Update pricing notes
        metadata["pricing_notes"] = notes
        if strategy:
            metadata["pricing_strategy"] = strategy
        if rationale:
            metadata["model_selection_rationale"] = rationale
        
        # Update database
        metadata_json = json.dumps(metadata, default=str, separators=(",", ":"))
        await db.query_traces(
            """
            UPDATE session_traces 
            SET metadata = :metadata 
            WHERE session_id = :session_id
            """,
            {"session_id": session_id, "metadata": metadata_json},
        )
        
        click.echo(f"✅ Pricing notes updated for session {session_id}")
        
    finally:
        await db.close()


@click.command("note")
@click.argument("session_id")
@click.argument("notes", required=False)
@click.option(
    "--strategy",
    help="Pricing strategy (e.g., 'cost_optimized', 'quality_first', 'balanced')",
)
@click.option(
    "--rationale",
    help="Model selection rationale",
)
@click.option(
    "--url",
    "db_url",
    default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
    help="Database URL",
)
@click.option(
    "--file",
    "notes_file",
    type=click.File("r"),
    help="Read notes from file instead of argument",
)
def note_command(
    session_id: str,
    notes: str | None,
    strategy: str | None,
    rationale: str | None,
    db_url: str,
    notes_file: Any | None,
) -> None:
    """Set pricing notes for a session.
    
    Examples:
        synth-ai session note abc123 "Used GPT-4o-mini for cost efficiency"
        synth-ai session note abc123 --strategy cost_optimized --rationale "50% savings"
        synth-ai session note abc123 --file notes.txt
    """
    # Get notes from file or argument
    if notes_file:
        notes_text = notes_file.read().strip()
    elif notes:
        notes_text = notes
    else:
        # Interactive mode - prompt for notes
        notes_text = click.prompt("Enter pricing notes", type=str)
    
    if not notes_text:
        raise click.ClickException("Pricing notes cannot be empty")
    
    asyncio.run(_set_pricing_note(db_url, session_id, notes_text, strategy, rationale))
```

**Lines**: ~120

### 3. `synth_ai/cli/commands/session/show.py`

**Purpose**: Command to display session details including pricing notes.

**Content**:
```python
"""Command to show session details including pricing notes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synth_ai.cli._storage import load_storage


async def _get_session_details(db_url: str, session_id: str) -> dict[str, Any] | None:
    """Get full session details including pricing notes."""
    create_storage, storage_config = load_storage()
    db = create_storage(storage_config(connection_string=db_url))
    await db.initialize()
    
    try:
        # Get session with metadata
        result = await db.query_traces(
            """
            SELECT 
                s.session_id,
                s.created_at,
                s.num_timesteps,
                s.num_events,
                s.num_messages,
                s.metadata,
                e.experiment_id,
                e.name as experiment_name,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost_usd,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
            FROM session_traces s
            LEFT JOIN experiments e ON s.experiment_id = e.experiment_id
            LEFT JOIN events ev ON s.session_id = ev.session_id
            WHERE s.session_id = :session_id
            GROUP BY s.session_id
            """,
            {"session_id": session_id},
        )
        
        if result is None or result.empty:
            return None
        
        row = result.iloc[0]
        
        # Parse metadata
        metadata_raw = row.get("metadata")
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(metadata_raw, dict):
            metadata = dict(metadata_raw)
        else:
            metadata = {}
        
        return {
            "session_id": row.get("session_id"),
            "created_at": row.get("created_at"),
            "num_timesteps": row.get("num_timesteps", 0),
            "num_events": row.get("num_events", 0),
            "num_messages": row.get("num_messages", 0),
            "experiment_id": row.get("experiment_id"),
            "experiment_name": row.get("experiment_name"),
            "total_cost_usd": float(row.get("total_cost_usd", 0.0) or 0.0),
            "total_tokens": int(row.get("total_tokens", 0) or 0),
            "pricing_notes": metadata.get("pricing_notes"),
            "pricing_strategy": metadata.get("pricing_strategy"),
            "model_selection_rationale": metadata.get("model_selection_rationale"),
            "metadata": metadata,
        }
        
    finally:
        await db.close()


def _render_session_panel(details: dict[str, Any]) -> Panel:
    """Render session details as a rich panel."""
    lines: list[str] = []
    
    lines.append(f"[bold]Session:[/bold] {details['session_id']}")
    lines.append(f"Created: {details['created_at']}")
    
    if details.get("experiment_name"):
        lines.append(f"Experiment: {details['experiment_name']} ([dim]{details['experiment_id']}[/dim])")
    
    lines.append("")
    lines.append(
        f"[bold]Stats[/bold]  "
        f"Timesteps: {details['num_timesteps']:,}  "
        f"Events: {details['num_events']:,}  "
        f"Messages: {details['num_messages']:,}"
    )
    lines.append(
        f"[bold]Cost[/bold]  "
        f"Total: ${details['total_cost_usd']:.4f}  "
        f"Tokens: {details['total_tokens']:,}"
    )
    
    # Pricing notes section
    if details.get("pricing_notes"):
        lines.append("")
        lines.append("[bold]💰 Pricing Notes[/bold]")
        lines.append(details["pricing_notes"])
        
        if details.get("pricing_strategy"):
            lines.append(f"[dim]Strategy:[/dim] {details['pricing_strategy']}")
        
        if details.get("model_selection_rationale"):
            lines.append("")
            lines.append("[bold]Model Selection Rationale[/bold]")
            lines.append(details["model_selection_rationale"])
    else:
        lines.append("")
        lines.append("[dim]No pricing notes set. Use 'synth-ai session note <session_id> <notes>' to add notes.[/dim]")
    
    body = "\n".join(lines)
    return Panel(body, title="Session Details", border_style="cyan")


@click.command("show")
@click.argument("session_id")
@click.option(
    "--url",
    "db_url",
    default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
    help="Database URL",
)
def show_command(session_id: str, db_url: str) -> None:
    """Show session details including pricing notes.
    
    Examples:
        synth-ai session show abc123
        synth-ai session-show abc123
    """
    console = Console()
    
    async def _run():
        details = await _get_session_details(db_url, session_id)
        if details is None:
            console.print(f"[red]Session {session_id} not found[/red]")
            return
        
        panel = _render_session_panel(details)
        console.print(panel)
    
    asyncio.run(_run())
```

**Lines**: ~180

### 4. `synth_ai/tracing_v3/turso/models.py` (Helper Methods)

**Purpose**: Add helper methods to SessionTrace model for accessing pricing notes.

**New Methods** (add to `SessionTrace` class):
```python
@property
def pricing_notes(self) -> str | None:
    """Get pricing notes from metadata."""
    if self.session_metadata:
        return self.session_metadata.get("pricing_notes")
    return None

@property
def pricing_strategy(self) -> str | None:
    """Get pricing strategy from metadata."""
    if self.session_metadata:
        return self.session_metadata.get("pricing_strategy")
    return None

def set_pricing_notes(
    self,
    notes: str,
    strategy: str | None = None,
    rationale: str | None = None,
) -> None:
    """Set pricing notes in metadata."""
    if self.session_metadata is None:
        self.session_metadata = {}
    self.session_metadata["pricing_notes"] = notes
    if strategy:
        self.session_metadata["pricing_strategy"] = strategy
    if rationale:
        self.session_metadata["model_selection_rationale"] = rationale
```

**Lines**: ~25 (additions to existing file)

---

## Modified Files

### 1. `synth_ai/cli/watch.py`

**Changes Required**:

#### a. Update `_experiment_detail()` query (Line ~121)

**Current**:
```python
sessions = await db.get_sessions_by_experiment(exp["experiment_id"])
```

**New**: Extract pricing notes from metadata in query or post-process:
```python
sessions = await db.get_sessions_by_experiment(exp["experiment_id"])

# Extract pricing notes from metadata for each session
for session in sessions:
    metadata = session.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            metadata = {}
    elif not isinstance(metadata, dict):
        metadata = {}
    
    session["pricing_notes"] = metadata.get("pricing_notes")
    session["pricing_strategy"] = metadata.get("pricing_strategy")
```

**Lines Changed**: ~15

#### b. Update `_render_experiment_panel()` (Line ~162)

**Current** (Line ~185-189):
```python
for s in sessions[:25]:
    lines.append(
        f"  - {s['session_id']}  [dim]{s['created_at']}[/dim]  "
        f"steps={s['num_timesteps']} events={s['num_events']} msgs={s['num_messages']}"
    )
```

**New**:
```python
for s in sessions[:25]:
    line = (
        f"  - {s['session_id']}  [dim]{s['created_at']}[/dim]  "
        f"steps={s['num_timesteps']} events={s['num_events']} msgs={s['num_messages']}"
    )
    lines.append(line)
    
    # Add pricing notes if present
    if s.get("pricing_notes"):
        notes_preview = s["pricing_notes"][:60] + "..." if len(s["pricing_notes"]) > 60 else s["pricing_notes"]
        lines.append(f"    💰 [dim]{notes_preview}[/dim]")
```

**Lines Changed**: ~10

#### c. Update `_traces_table()` query (Line ~419)

**Current**:
```python
df = await db.query_traces("SELECT * FROM session_summary ORDER BY created_at DESC")
```

**New**: Extract pricing notes from metadata:
```python
df = await db.query_traces(
    """
    SELECT 
        s.*,
        JSON_EXTRACT(s.metadata, '$.pricing_notes') as pricing_notes
    FROM session_summary s
    ORDER BY s.created_at DESC
    """
)
```

**Optional**: Add "Notes" column to table (with `--show-notes` flag):
```python
if show_notes:
    table.add_column("Notes", justify="left", max_width=40)
    # ... in row loop ...
    if show_notes:
        notes = str(r.get("pricing_notes", ""))[:40] if r.get("pricing_notes") else "-"
        table.add_row(..., notes)
```

**Lines Changed**: ~20

**Total Changes**: ~45 lines

### 2. `synth_ai/cli/recent.py`

**Changes Required**:

#### Update `_fetch_recent()` query (Line ~43)

**Current** (Line ~50-73):
```python
query = """
    WITH windowed_sessions AS (
        SELECT *
        FROM session_traces
        WHERE created_at >= :start_time
    )
    SELECT 
        e.experiment_id,
        e.name,
        ...
    FROM windowed_sessions ws
    ...
"""
```

**New**: Add pricing notes extraction (optional, for verbose mode):
```python
query = """
    WITH windowed_sessions AS (
        SELECT 
            *,
            JSON_EXTRACT(metadata, '$.pricing_notes') as pricing_notes
        FROM session_traces
        WHERE created_at >= :start_time
    )
    SELECT 
        e.experiment_id,
        e.name,
        ...
        COUNT(DISTINCT CASE WHEN ws.pricing_notes IS NOT NULL THEN ws.session_id END) as sessions_with_notes
    FROM windowed_sessions ws
    ...
"""
```

**Lines Changed**: ~5 (optional enhancement)

### 3. `synth_ai/tracing_v3/storage/base.py` or `turso/storage.py`

**Changes Required**: Ensure `get_sessions_by_experiment()` returns metadata.

**Check**: Verify that `get_sessions_by_experiment()` method includes metadata in results. If not, update query to include metadata column.

**Lines Changed**: ~5-10 (if needed)

### 4. `synth_ai/cli/root.py` or `synth_ai/cli/__init__.py`

**Changes Required**: Register session command group.

**In `synth_ai/cli/__init__.py`** (Line ~68-92):
```python
# Register session commands
_maybe_call("synth_ai.cli.commands.session", "register", cli)
```

**Lines Changed**: ~2

---

## Data Flow & Architecture

### Setting Pricing Notes

```
User Command: synth-ai session note abc123 "Notes here"
    ↓
CLI Handler (note.py)
    ↓
Database Query: SELECT metadata FROM session_traces WHERE session_id = ...
    ↓
Parse JSON metadata
    ↓
Update metadata["pricing_notes"] = "Notes here"
    ↓
Database Update: UPDATE session_traces SET metadata = ... WHERE session_id = ...
    ↓
Success confirmation
```

### Displaying Pricing Notes

```
User Command: synth-ai session show abc123
    ↓
CLI Handler (show.py)
    ↓
Database Query: SELECT ... JSON_EXTRACT(metadata, '$.pricing_notes') ...
    ↓
Parse results
    ↓
Render Rich Panel with pricing notes
    ↓
Display to user
```

### Integration with Watch/Recent Commands

```
User Command: synth-ai experiment abc123
    ↓
watch.py: _experiment_detail()
    ↓
Database Query: get_sessions_by_experiment()
    ↓
Post-process: Extract pricing_notes from metadata JSON
    ↓
watch.py: _render_experiment_panel()
    ↓
Display sessions with pricing notes inline
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goal**: Basic functionality to set and retrieve pricing notes.

**Tasks**:
1. ✅ Add helper methods to `SessionTrace` model (`models.py`)
2. ✅ Create `synth_ai/cli/commands/session/__init__.py`
3. ✅ Create `synth_ai/cli/commands/session/note.py`
4. ✅ Create `synth_ai/cli/commands/session/show.py`
5. ✅ Register session commands in `cli/__init__.py`
6. ✅ Test basic set/get functionality

**Deliverables**:
- `synth-ai session note <id> <notes>` works
- `synth-ai session show <id>` displays notes
- Notes persist in database

### Phase 2: Display Integration (Week 1-2)

**Goal**: Show pricing notes in existing CLI views.

**Tasks**:
1. ✅ Update `watch.py` `_experiment_detail()` to extract pricing notes
2. ✅ Update `watch.py` `_render_experiment_panel()` to display notes
3. ✅ Optional: Add `--show-notes` flag to `_traces_table()`
4. ✅ Update `recent.py` query (optional enhancement)
5. ✅ Test display in experiment views

**Deliverables**:
- Pricing notes appear in `synth-ai experiment <id>` output
- Notes shown inline with session listings

### Phase 3: Enhanced Features (Week 2)

**Goal**: Add convenience features and polish.

**Tasks**:
1. ✅ Add `--strategy` and `--rationale` options to `note` command
2. ✅ Add `--file` option to read notes from file
3. ✅ Add interactive mode (prompt if notes not provided)
4. ✅ Improve error handling and validation
5. ✅ Add help text and examples

**Deliverables**:
- Full-featured `note` command with all options
- Better UX with interactive prompts

### Phase 4: Testing & Documentation (Week 2-3)

**Goal**: Comprehensive testing and documentation.

**Tasks**:
1. ✅ Unit tests for helper methods
2. ✅ Integration tests for CLI commands
3. ✅ Test with various metadata formats (existing sessions)
4. ✅ Test edge cases (empty notes, special characters, long notes)
5. ✅ Update documentation
6. ✅ Add examples to help text

**Deliverables**:
- Test suite with >80% coverage
- Updated documentation
- Examples in help text

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/tracing_v3/test_session_pricing_notes.py`

**Test Cases**:
1. `test_pricing_notes_property()` - Get notes from metadata
2. `test_set_pricing_notes()` - Set notes in metadata
3. `test_pricing_notes_none()` - Handle missing metadata
4. `test_pricing_strategy()` - Get strategy from metadata
5. `test_set_pricing_notes_with_strategy()` - Set notes with strategy

**Lines**: ~100

### Integration Tests

**File**: `tests/integration/cli/test_session_commands.py`

**Test Cases**:
1. `test_session_note_command()` - Set notes via CLI
2. `test_session_show_command()` - Display notes via CLI
3. `test_session_note_with_file()` - Read notes from file
4. `test_session_note_interactive()` - Interactive mode
5. `test_session_note_update_existing()` - Update existing notes
6. `test_session_note_invalid_session()` - Error handling

**Lines**: ~200

### CLI Display Tests

**File**: `tests/integration/cli/test_watch_pricing_notes.py`

**Test Cases**:
1. `test_experiment_panel_shows_notes()` - Notes appear in experiment view
2. `test_traces_table_with_notes()` - Notes in traces table (if implemented)
3. `test_recent_shows_notes_count()` - Notes count in recent view (if implemented)

**Lines**: ~150

### Edge Case Tests

**File**: `tests/unit/tracing_v3/test_pricing_notes_edge_cases.py`

**Test Cases**:
1. `test_empty_notes()` - Empty string handling
2. `test_special_characters()` - JSON special chars in notes
3. `test_very_long_notes()` - Notes >1000 chars
4. `test_unicode_notes()` - Unicode characters
5. `test_existing_metadata_preserved()` - Other metadata not overwritten
6. `test_malformed_metadata()` - Handle corrupted JSON

**Lines**: ~120

**Total Test Lines**: ~570

---

## Migration Considerations

### Backward Compatibility

**No Breaking Changes**:
- Existing sessions without pricing notes continue to work
- Metadata structure is additive (only adds new keys)
- All queries handle missing `pricing_notes` gracefully

### Data Migration

**Not Required**: No migration needed since we're using existing `metadata` column.

**Optional Cleanup**:
- If users want to migrate from other note systems, provide migration script
- Not in scope for initial implementation

### Performance Considerations

**JSON Extraction Performance**:
- SQLite JSON functions (`JSON_EXTRACT`) are efficient for small datasets
- For large datasets (>100k sessions), consider:
  - Generated column: `ALTER TABLE session_traces ADD COLUMN pricing_notes_text TEXT GENERATED ALWAYS AS (JSON_EXTRACT(metadata, '$.pricing_notes')) STORED;`
  - Index on generated column if needed
- Monitor query performance in production

**Query Optimization**:
- Current queries already use `LEFT JOIN` - no performance impact
- JSON extraction adds minimal overhead
- Consider caching if display becomes slow

---

## API Contracts

### CLI Commands

#### `synth-ai session note <session_id> <notes> [OPTIONS]`

**Arguments**:
- `session_id` (required): Session ID (full or partial match)
- `notes` (optional): Pricing notes text (if not provided, prompts interactively)

**Options**:
- `--strategy <strategy>`: Pricing strategy tag
- `--rationale <rationale>`: Model selection rationale
- `--file <path>`: Read notes from file
- `--url <db_url>`: Database URL override

**Exit Codes**:
- `0`: Success
- `1`: Error (session not found, invalid input, etc.)

**Output**:
- Success: `✅ Pricing notes updated for session <id>`
- Error: Error message to stderr

#### `synth-ai session show <session_id> [OPTIONS]`

**Arguments**:
- `session_id` (required): Session ID (full or partial match)

**Options**:
- `--url <db_url>`: Database URL override

**Exit Codes**:
- `0`: Success
- `1`: Session not found

**Output**:
- Rich panel with session details and pricing notes
- If no notes: Hint message to add notes

### Internal APIs

#### `SessionTrace.pricing_notes` (property)

**Returns**: `str | None`
**Raises**: None
**Side Effects**: None

#### `SessionTrace.set_pricing_notes(notes: str, strategy: str | None = None, rationale: str | None = None)`

**Parameters**:
- `notes` (required): Pricing notes text
- `strategy` (optional): Pricing strategy tag
- `rationale` (optional): Model selection rationale

**Returns**: `None`
**Raises**: None
**Side Effects**: Modifies `self.session_metadata`

---

## File Summary

### New Files (3)
1. `synth_ai/cli/commands/session/__init__.py` (~30 lines)
2. `synth_ai/cli/commands/session/note.py` (~120 lines)
3. `synth_ai/cli/commands/session/show.py` (~180 lines)

**Total New Lines**: ~330

### Modified Files (4)
1. `synth_ai/tracing_v3/turso/models.py` (+25 lines)
2. `synth_ai/cli/watch.py` (+45 lines)
3. `synth_ai/cli/recent.py` (+5 lines, optional)
4. `synth_ai/cli/__init__.py` (+2 lines)

**Total Modified Lines**: ~77

### Test Files (3)
1. `tests/unit/tracing_v3/test_session_pricing_notes.py` (~100 lines)
2. `tests/integration/cli/test_session_commands.py` (~200 lines)
3. `tests/integration/cli/test_watch_pricing_notes.py` (~150 lines)
4. `tests/unit/tracing_v3/test_pricing_notes_edge_cases.py` (~120 lines)

**Total Test Lines**: ~570

### Grand Total
- **New Code**: ~330 lines
- **Modified Code**: ~77 lines
- **Test Code**: ~570 lines
- **Total**: ~977 lines

---

## Dependencies

### External Dependencies
- None (uses existing dependencies)

### Internal Dependencies
- `synth_ai.cli._storage` - Database connection
- `synth_ai.tracing_v3.turso.models` - SessionTrace model
- `rich` - CLI formatting (already used)
- `click` - CLI framework (already used)

---

## Risk Assessment

### Low Risk
- ✅ No schema changes required
- ✅ Backward compatible
- ✅ Additive feature (doesn't modify existing behavior)
- ✅ Well-isolated code (new command group)

### Medium Risk
- ⚠️ JSON extraction performance on large datasets
- ⚠️ Metadata parsing edge cases (malformed JSON)
- ⚠️ CLI UX (interactive mode, file input)

### Mitigation Strategies
1. **Performance**: Monitor query times, add generated column if needed
2. **Edge Cases**: Comprehensive test coverage, graceful error handling
3. **UX**: Clear error messages, helpful prompts, examples in help text

---

## Future Enhancements

### Phase 5+ (Post-MVP)

1. **Rich Text Formatting**: Markdown support in notes
2. **Templates**: Pre-defined note templates for common scenarios
3. **Bulk Operations**: Set notes for multiple sessions
4. **Search/Filter**: Filter sessions by pricing notes content
5. **Auto-generation**: Generate notes based on cost patterns
6. **Export**: Include notes in cost reports
7. **API**: REST API for programmatic access
8. **Validation**: Schema validation for pricing_strategy values
9. **History**: Track note changes over time
10. **Tags**: Categorize notes with tags

---

## Success Criteria

### MVP Success Metrics
1. ✅ Users can set pricing notes via CLI
2. ✅ Users can view pricing notes via CLI
3. ✅ Notes appear in experiment views
4. ✅ Notes persist correctly in database
5. ✅ No performance degradation
6. ✅ >80% test coverage
7. ✅ Documentation complete

### User Acceptance Criteria
1. ✅ Can add notes to existing sessions
2. ✅ Can update notes
3. ✅ Notes visible in session details
4. ✅ Notes visible in experiment views
5. ✅ Clear error messages for invalid inputs
6. ✅ Help text is clear and helpful

---

## Timeline Estimate

- **Phase 1**: 2-3 days
- **Phase 2**: 2-3 days
- **Phase 3**: 1-2 days
- **Phase 4**: 2-3 days

**Total**: 7-11 days (1.5-2 weeks)

---

## Open Questions

1. **Note Length Limit**: Should we enforce a max length? (Recommendation: 10,000 chars)
2. **Strategy Values**: Should we validate/enumerate strategy values? (Recommendation: No, keep flexible)
3. **Note History**: Should we track changes? (Recommendation: Not in MVP)
4. **Permissions**: Any access control needed? (Recommendation: No, local database)
5. **Export Format**: How should notes be exported? (Recommendation: Include in JSON exports)

---

## Conclusion

This implementation plan provides a comprehensive roadmap for adding pricing notes functionality to synth-ai. The approach is low-risk, backward-compatible, and well-scoped. The feature enhances user experience by allowing documentation of pricing decisions directly in the session data.

**Key Strengths**:
- No schema migration required
- Clean separation of concerns
- Comprehensive test coverage planned
- Clear implementation phases
- Well-documented API contracts

**Next Steps**:
1. Review and approve this plan
2. Create feature branch
3. Begin Phase 1 implementation
4. Iterate based on feedback

