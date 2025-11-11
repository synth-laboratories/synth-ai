# Session Pricing Notes - Implementation Plan

## Overview
Add the ability to attach pricing notes to sessions in synth-ai. These notes will help users document pricing-related information, model selection rationale, cost optimization strategies, or billing context for individual sessions.

## Current State

### Database Schema
- **`session_traces` table** (`synth_ai/tracing_v3/turso/models.py`):
  - `session_id` (PRIMARY KEY)
  - `metadata` (JSONText) - Currently stores arbitrary JSON metadata
  - Cost is aggregated from `events` table (`cost_usd` field)

### Display Locations
1. **`synth_ai/cli/watch.py`**:
   - `_experiments_table()` - Shows experiments with cost
   - `_render_experiment_panel()` - Shows experiment details with sessions
   - `_traces_table()` - Shows recent sessions with cost
   - `_recent_table()` - Shows recent experiments with cost

2. **`synth_ai/cli/recent.py`**:
   - Shows experiments with cost in last K hours

3. **`synth_ai/cli/commands/status/subcommands/pricing.py`**:
   - Shows model pricing rates

### Pricing Infrastructure
- **`synth_ai/pricing/model_pricing.py`**: Contains `MODEL_PRICES` dictionary with token rates
- Cost calculation happens in `synth_ai/tracing_v3/utils.py` (`calculate_cost()`)
- Cost stored in `events` table as `cost_usd` (in cents)

## Implementation Options

### Option 1: Store in `session_traces.metadata` JSON (Recommended)
**Pros:**
- No schema migration needed
- Flexible - can store structured data
- Already exists and is used for other metadata

**Cons:**
- Less queryable (would need JSON queries)
- No direct database constraint/validation

**Implementation:**
```python
# In session_traces.metadata:
{
    "pricing_notes": "Used GPT-4o-mini for cost efficiency. Estimated 50% savings vs GPT-4o.",
    "pricing_strategy": "cost_optimized",
    "model_selection_rationale": "..."
}
```

### Option 2: Add dedicated `pricing_notes` column
**Pros:**
- Directly queryable
- Clear schema
- Better for filtering/searching

**Cons:**
- Requires database migration
- Less flexible for structured data

**Implementation:**
```sql
ALTER TABLE session_traces ADD COLUMN pricing_notes TEXT;
```

## Recommended Approach: Option 1 (metadata JSON)

### Phase 1: Data Storage
1. **Update `SessionTrace` model** (`synth_ai/tracing_v3/turso/models.py`):
   - No changes needed - `session_metadata` already supports JSON
   - Add helper methods/properties for accessing pricing notes

2. **Add helper methods**:
   ```python
   # In SessionTrace class
   @property
   def pricing_notes(self) -> str | None:
       """Get pricing notes from metadata."""
       if self.session_metadata:
           return self.session_metadata.get("pricing_notes")
       return None
   
   def set_pricing_notes(self, notes: str) -> None:
       """Set pricing notes in metadata."""
       if self.session_metadata is None:
           self.session_metadata = {}
       self.session_metadata["pricing_notes"] = notes
   ```

### Phase 2: CLI Commands
1. **Add command to set pricing notes**:
   - `synth ai session note <session_id> <notes>` or
   - `synth ai session pricing-note <session_id> <notes>`

2. **Update display functions** to show pricing notes:
   - `_render_experiment_panel()` in `watch.py` - Show notes for each session
   - `_traces_table()` in `watch.py` - Add "Notes" column (optional, can be verbose)
   - Consider a `--show-notes` flag for verbose mode

### Phase 3: Display Integration
1. **Update `_render_experiment_panel()`** (`synth_ai/cli/watch.py`):
   ```python
   for s in sessions[:25]:
       line = f"  - {s['session_id']}  [dim]{s['created_at']}[/dim]  "
       line += f"steps={s['num_timesteps']} events={s['num_events']} msgs={s['num_messages']}"
       if s.get('pricing_notes'):
           line += f"\n    💰 {s['pricing_notes']}"
       lines.append(line)
   ```

2. **Add new command for viewing session details**:
   - `synth ai session show <session_id>` - Show full session details including pricing notes

### Phase 4: Query Support
1. **Update database queries** to include pricing notes:
   - Modify queries in `watch.py` and `recent.py` to extract `pricing_notes` from metadata JSON
   - SQLite JSON extraction: `JSON_EXTRACT(metadata, '$.pricing_notes')`

## File Changes Summary

### New Files
- `synth_ai/cli/commands/session/__init__.py` - New command group
- `synth_ai/cli/commands/session/note.py` - Command to set pricing notes
- `synth_ai/cli/commands/session/show.py` - Command to show session details

### Modified Files
1. **`synth_ai/tracing_v3/turso/models.py`**:
   - Add helper methods to `SessionTrace` for pricing notes

2. **`synth_ai/cli/watch.py`**:
   - Update `_experiment_detail()` query to extract pricing notes
   - Update `_render_experiment_panel()` to display notes
   - Update `_traces_table()` query (optional)

3. **`synth_ai/cli/recent.py`**:
   - Update query to extract pricing notes (optional)

4. **`synth_ai/cli/root.py`**:
   - Register new session commands

## Example Usage

```bash
# Set pricing notes for a session
synth ai session note abc123 "Used GPT-4o-mini for cost efficiency. Estimated 50% savings vs GPT-4o."

# View session details (including notes)
synth ai session show abc123

# View experiment with sessions (notes shown inline)
synth ai watch experiment abc123
```

## Database Query Examples

```sql
-- Extract pricing notes from metadata
SELECT 
    session_id,
    JSON_EXTRACT(metadata, '$.pricing_notes') as pricing_notes,
    created_at
FROM session_traces
WHERE JSON_EXTRACT(metadata, '$.pricing_notes') IS NOT NULL;

-- Update pricing notes
UPDATE session_traces
SET metadata = json_set(
    COALESCE(metadata, '{}'),
    '$.pricing_notes',
    'New pricing note here'
)
WHERE session_id = 'abc123';
```

## Future Enhancements
1. **Rich text formatting** in notes (markdown support)
2. **Pricing note templates** for common scenarios
3. **Bulk operations** - Set notes for multiple sessions
4. **Search/filter** sessions by pricing notes
5. **Integration with cost analysis** - Auto-generate notes based on cost patterns
6. **Export pricing notes** with cost reports

## Testing Considerations
1. Test setting notes on new sessions
2. Test updating notes on existing sessions
3. Test displaying notes in various CLI views
4. Test with sessions that have no notes
5. Test JSON extraction queries
6. Test with special characters in notes

