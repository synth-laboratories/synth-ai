# Context Files for Session Pricing Notes Implementation

## Primary Documentation (Required Reading)

### 1. Implementation Plan
- **`synth-ai/docs/planning/session_pricing_notes_implementation.md`**
  - Complete implementation plan with all phases
  - File-by-file breakdown with line counts
  - Code examples and API contracts
  - Testing strategy

### 2. Original Requirements
- **`synth-ai/docs/planning/session_pricing_notes.md`**
  - Original feature requirements
  - Design decisions (Option 1 vs Option 2)
  - Use cases and examples

## Core Code Files (Implementation Targets)

### Database Models
- **`synth-ai/synth_ai/tracing_v3/turso/models.py`**
  - `SessionTrace` class (Line 81-113)
  - `JSONText` type decorator (Line 64-78)
  - Current metadata structure

### CLI Display Files (Need Modification)
- **`synth-ai/synth_ai/cli/watch.py`**
  - `_experiment_detail()` function (Line ~121)
  - `_render_experiment_panel()` function (Line ~162)
  - `_traces_table()` function (Line ~419)
  - Current session display logic

- **`synth-ai/synth_ai/cli/recent.py`**
  - `_fetch_recent()` function (Line ~43)
  - Recent experiments query

### CLI Command Structure Examples
- **`synth-ai/synth_ai/cli/commands/status/__init__.py`**
  - Example of command group registration pattern

- **`synth-ai/synth_ai/cli/commands/baseline/core.py`** or **`synth-ai/synth_ai/cli/commands/help/core.py`**
  - Example of command implementation pattern

- **`synth-ai/synth_ai/cli/__init__.py`**
  - How commands are registered (Line ~68-92)
  - Command registration pattern

### Storage/Database Layer
- **`synth-ai/synth_ai/cli/_storage.py`**
  - Database connection helpers
  - `load_storage()` function

- **`synth-ai/synth_ai/tracing_v3/storage/base.py`** or **`synth-ai/synth_ai/tracing_v3/turso/storage.py`**
  - `get_sessions_by_experiment()` method (if exists)
  - Database query patterns

## Reference Files (For Understanding Patterns)

### Database Query Examples
- **`synth-ai/synth_ai/cli/watch.py`**
  - See `_fetch_experiments()` (Line ~57) for query pattern
  - See `_experiment_detail()` (Line ~121) for session queries

- **`synth-ai/synth_ai/tracing_v3/trace_utils.py`**
  - `load_session_trace()` function (Line ~80)
  - How metadata is loaded and parsed

### CLI Command Examples
- **`synth-ai/synth_ai/cli/commands/baseline/list.py`**
  - Example of CLI command with options
  - Rich table rendering

- **`synth-ai/synth_ai/cli/commands/help/core.py`**
  - Simple command example

### Rich Console Examples
- **`synth-ai/synth_ai/cli/watch.py`**
  - Rich Panel usage (Line ~162-194)
  - Rich Table usage (Line ~86-118)

## Testing Reference Files

### Test Structure Examples
- **`synth-ai/tests/integration/cli/test_cli_prompt_learning_shell_scripts.py`**
  - CLI command testing patterns

- **`synth-ai/tests/unit/cli/test_codex_command.py`**
  - Unit test patterns for CLI commands

## Implementation Checklist

### Files to Create (New)
1. `synth-ai/synth_ai/cli/commands/session/__init__.py`
2. `synth-ai/synth_ai/cli/commands/session/note.py`
3. `synth-ai/synth_ai/cli/commands/session/show.py`
4. `synth-ai/tests/unit/tracing_v3/test_session_pricing_notes.py`
5. `synth-ai/tests/integration/cli/test_session_commands.py`
6. `synth-ai/tests/integration/cli/test_watch_pricing_notes.py`
7. `synth-ai/tests/unit/tracing_v3/test_pricing_notes_edge_cases.py`

### Files to Modify (Existing)
1. `synth-ai/synth_ai/tracing_v3/turso/models.py` - Add helper methods
2. `synth-ai/synth_ai/cli/watch.py` - Extract and display notes
3. `synth-ai/synth_ai/cli/recent.py` - Optional enhancement
4. `synth-ai/synth_ai/cli/__init__.py` - Register session commands

## Key Implementation Details

### Metadata JSON Structure
```json
{
  "pricing_notes": "string",
  "pricing_strategy": "string (optional)",
  "model_selection_rationale": "string (optional)"
}
```

### SQLite JSON Functions
- `JSON_EXTRACT(metadata, '$.pricing_notes')` - Extract value
- `json_set(metadata, '$.pricing_notes', 'value')` - Update value
- `COALESCE(metadata, '{}')` - Handle NULL metadata

### Database Column
- Table: `session_traces`
- Column: `metadata` (JSONText type)
- No schema migration needed

## Quick Start Guide for Implementer

1. **Read**: `session_pricing_notes_implementation.md` (complete plan)
2. **Review**: `models.py` to understand SessionTrace structure
3. **Study**: `watch.py` to see how sessions are displayed
4. **Follow**: Command registration pattern in `cli/__init__.py`
5. **Implement**: Phase 1 (core infrastructure) first
6. **Test**: Use test files as reference for test structure

## Critical Implementation Notes

- **No database migration** - uses existing `metadata` column
- **Backward compatible** - existing sessions work fine
- **JSON parsing** - handle both string and dict metadata formats
- **Error handling** - graceful handling of malformed JSON
- **Rich formatting** - use Rich library for CLI output (already imported)

