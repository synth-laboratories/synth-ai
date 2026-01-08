# TUI Migration Differences: Original vs SolidJS

## Overview
The TUI has been migrated from `@opentui/core` (imperative) to `@opentui/solid` (declarative JSX).

## Key Differences

### 1. **Title/Header**
- **Original**: `"Synth AI"`
- **Migrated**: `"Synth AI (SolidJS migration scaffold)"`
- **Location**: Header bar at top of screen

### 2. **Tab Navigation**
- **Original** (`src/components/layout.ts`):
  - `Create New Job (n)` ← **Missing in migrated version**
  - `View Jobs (b)`
  - `View Job's Events (e)`
  - `View Logs (g)`

- **Migrated** (`src/solid/app.tsx`):
  - `[b] Jobs`
  - `[e] Events`
  - `[g] Logs`
  - `[shift+g] Agent` ← **New in migrated version** (OpenCode integration)

**Note:** The original uses a `key-hint` primitive that formats tabs as `"Description (key)"`; the migrated version uses `"[%key%] Label"` style.

### 3. **Framework Architecture**

#### Original (`@opentui/core`)
- **Imperative API**: Manual creation and management of renderables
- **Example**:
  ```typescript
  const headerText = new TextRenderable(renderer, {
    id: "header-text",
    content: "Synth AI",
    fg: "#e2e8f0",
  })
  headerBox.add(headerText)
  ```

#### Migrated (`@opentui/solid`)
- **Declarative JSX**: React-like component syntax
- **Example**:
  ```tsx
  <text fg="#e2e8f0">Synth AI (SolidJS migration scaffold)</text>
  ```

### 4. **State Management**

#### Original
- Manual state updates via renderer methods
- Explicit re-rendering required
- Direct DOM manipulation through renderables

#### Migrated
- **SolidJS reactive primitives**: `createSignal`, `createMemo`, `createEffect`
- **Automatic reactivity**: UI updates when signals change
- **Example**:
  ```tsx
  const activePane = createMemo(() => data.ctx.state.appState.activePane)
  <text fg={activePane() === "jobs" ? "#f8fafc" : "#94a3b8"}>[b] Jobs</text>
  ```

### 5. **Code Organization**

#### Original
- `src/components/layout.ts` - Imperative layout builder
- `src/index.ts` - Main entry point with manual renderer setup
- Manual component lifecycle management

#### Migrated
- `src/solid/app.tsx` - Main SolidJS component tree
- `src/solid/layout.ts` - Layout calculation utilities
- `src/solid/data.ts` - Data hooks and state management
- Component-based architecture with JSX

### 6. **New Features in Migrated Version**

1. **OpenCode Agent Pane** (`[shift+g] Agent`)
   - Full OpenCode integration
   - Chat interface for AI agent
   - Session management
   - Located in `src/solid/opencode/`

2. **Better Error Handling**
   - `<ErrorBoundary>` components
   - Graceful degradation

3. **Reactive Layout**
   - Automatic layout recalculation on resize
   - `useTerminalDimensions()` hook
   - Responsive design

### 7. **Visual Differences**

Based on reference screenshots vs current implementation (key “CSS-like” diffs):

| Aspect | Original | Migrated |
|--------|----------|----------|
| Title | "Synth AI" | "Synth AI (SolidJS migration scaffold)" |
| Tabs | 4 tabs (including Create New Job) | 4 tabs (including Agent) |
| Tab label format | `Description (key)` | `[key] Label` |
| Layout | Same 2-pane layout | Same 2-pane layout |
| Footer | Same status/keybindings | Same status/keybindings |

#### 7.1 **Selection highlight + list presentation (biggest visual delta)**

- **Original** uses `SelectRenderable` with *background-based* selection:
  - selected row background `#2563eb` + selected text `#ffffff`
  - focused background `#1e293b`
  - shows a description column, scroll indicator, and wraps selection
- **Migrated** renders rows as plain `<text>` lines:
  - selection is indicated by `>` + foreground color `#60a5fa`
  - no selected background band, no separate description column, no scroll indicator

#### 7.2 **Pane padding / density**

- **Original** lists are packed tightly (the select widget manages its own internal padding).
- **Migrated** Jobs pane adds `paddingLeft={1}` and `paddingTop={1}`, reducing density and shifting content inward.

#### 7.3 **Header + tab insets**

- **Original** header/tabs have no explicit left padding.
- **Migrated** adds `paddingLeft={1}` (header + tabs), so text is inset 1 column.

#### 7.4 **Focus/active-state affordances**

- **Original** has clear focus states via the widget’s focused/selected backgrounds.
- **Migrated** relies mainly on text color changes; focus is less visually obvious.

### 8. **Missing Features**

- **`[n] Create New Job` tab** - Not yet migrated
  - Original has `create-job-modal.ts` modal
  - Migrated version doesn't expose this tab
  - Functionality may exist but not accessible via tab

### 9. **Visual Parity Checklist (keep Agent pane)**

To reach “gold” parity while keeping `[shift+g] Agent`:

1. Recreate `key-hint` tab styling (`Description (key)`) for tabs/shortcuts *or* restyle migrated tabs to match reference.
2. Replace “plain text list” rendering in Jobs/Events/Logs with a selectable list component that supports:
   - selectedBackgroundColor / selectedTextColor
   - focusedBackgroundColor / focusedTextColor
   - description column + scroll indicator
3. Match padding/density (remove extra top padding in Jobs pane; align insets with reference).
4. Re-add the **Create New Job** flow (`n`) in the migrated shell (modal + footer hinting), without removing Agent.

### 9.1 **Remote branch review: `origin/feat/job-details` (merge candidates into SolidJS)**

This branch contains a bunch of high-quality TUI work that we should port into `src/solid/` while keeping all Agent/OpenCode functionality:

1. **Centralized theme constants** (`synth_ai/tui/app/src/theme.ts`)
   - A single `BOX` style object (border colors, focused border color, text/bg colors). Great for visual parity + consistency.

2. **Reusable “list panel” abstraction (left sidebar)** (`synth_ai/tui/app/src/components/list-panel.ts`, `synth_ai/tui/app/src/ui/list-panels/*`)
   - A generic `ListPanelConfig<T>` + renderer that drives `SelectRenderable`.
   - Enables “gold” selection visuals: background selection color, description column, scroll indicator.
   - Cleanly supports multiple list modes (jobs vs logs) without duplicating UI wiring.

3. **Better jobs list presentation** (`synth_ai/tui/app/src/formatters/job-card.ts`, `synth_ai/tui/app/src/ui/list-panels/jobs-list.ts`)
   - Compact job “card” with `status | date` semantics and job type display.
   - Pairs well with the reference UI’s description column.

4. **Detail-panel modularization** (`synth_ai/tui/app/src/ui/detail-panels/*`, `synth_ai/tui/app/src/ui/render.ts`)
   - Separates “left list panel” vs “right detail panels” rendering.
   - We should mimic this in SolidJS to shrink `src/solid/app.tsx` and make parity work less risky.

5. **Real-time Job Details via SSE** (`synth_ai/tui/app/src/api/job-details-stream.ts`)
   - SSE client for `/api/prompt-learning/online/jobs/{job_id}/events/stream?since_seq=...`
   - Explicitly designed to work across job types (eval/learning/prompt-learning), which is ideal for our unified jobs view.

6. **Job details formatting improvements** (`synth_ai/tui/app/src/formatters/job-details.ts`)
   - Status-aware timestamps (started/last update/finished only when relevant).
   - Uses normalized job status helpers (`utils/job-status.ts`).

7. **Keyboard + focus routing improvements (incl. Create New Job)** (`synth_ai/tui/app/src/handlers/keyboard.ts`)
   - Introduces “list panel id” concept and routes pane switching accordingly.
   - Adds `n` → Create Job modal as a first-class global shortcut.

8. **Create New Job flow improvements** (`synth_ai/tui/app/src/modals/create-job-modal.ts`)
   - Wizard-style modal with LocalAPI scanning, “create new” pathing, and deploy workflow.
   - NDJSON parsing for deploy status, integrates with logs/task-apps experience.
   - This is exactly the “gold” Create Job flow we should bring into SolidJS.

9. **Jobs refresh/selection robustness** (`synth_ai/tui/app/src/api/jobs.ts`)
   - Merges prompt-learning + learning job lists, better status messaging, and safer auto-select behavior.

### 10. **File Structure Comparison**

```
Original:
src/
├── components/
│   └── layout.ts          # Imperative layout builder
├── index.ts               # Entry point
└── ...

Migrated:
src/
├── solid/
│   ├── app.tsx            # Main SolidJS component
│   ├── layout.ts          # Layout utilities
│   ├── data.ts            # Data hooks
│   └── opencode/          # OpenCode integration
└── ...
```

## Migration Status

✅ **Completed**:
- Core layout structure
- Centralized theme constants (`src/solid/theme.ts`)
- Gold-style tab navigation using `KeyHint` component
- Jobs pane with background-based selection highlight and description column (`JobsList`)
- Logs pane with background-based selection highlight (`LogsList`)
- Jobs detail panels extracted (`JobsDetail`)
- Logs detail panel extracted (`LogsDetail`)
- Formatters modularized (`src/solid/formatters/`)
- Job status utilities ported (`src/solid/utils/job-status.ts`)
- Header styling alignment (removed scaffold suffix, fixed insets)
- Footer styling alignment using `KeyHint`
- OpenCode Agent integration
- Settings modal keyboard navigation (fixed reactivity)
- SSE streaming for real-time job details (`src/solid/api/job-details-stream.ts`)
- LocalAPI file scanning (`src/solid/utils/localapi-scanner.ts`)
- File path utilities (`src/solid/utils/files.ts`)
- **Full Create Job wizard** (`src/solid/modals/CreateJobModal.tsx`)
  - LocalAPI file selection
  - File creation with template (`src/solid/templates/localapi.ts`)
  - Job type selection
  - **Deployment with NDJSON parsing** (`src/solid/services/deployment.ts`)
  - **Job submission** (eval, prompt_learning, learning)
- **File creation service** (`src/solid/services/file-creation.ts`)

⚠️ **Partially Complete**:
- Events pane selection (uses text-only selection, not background highlight)

🔲 **Pending**:
- (None - all major features implemented)

## Next Steps

1. **Add `[n] Create New Job` tab** to migrated version
2. **Remove "(SolidJS migration scaffold)"** from title once migration is complete
3. **Remove original `@opentui/core` code** once fully migrated
4. **Update documentation** to reflect SolidJS architecture

## Benefits of Migration

1. **Better Developer Experience**: JSX is more familiar and easier to read
2. **Reactive Updates**: Automatic UI updates when state changes
3. **Component Reusability**: Easier to extract and reuse components
4. **Type Safety**: Better TypeScript integration with SolidJS
5. **Performance**: SolidJS fine-grained reactivity is more efficient
6. **Modern Patterns**: Aligns with modern React-like patterns

### Target state (what “aligned” means)
- **Visual parity with gold TUI**: tab label style (`Description (key)`), background selection highlight + description column + scroll indicator, density/padding matches, consistent focus affordances.
- **Keep + elevate Agent pane**: `[shift+g] Agent` stays first-class and doesn’t regress.
- **SolidJS code health**: no mega-file orchestration; clear “data → view model → UI” separation; reusable primitives (theme, list panels, detail panels, modals).
- **Feature parity with RT branch (`origin/feat/job-details`)**: job details SSE streaming, modular detail/list panels, create job wizard, keyboard/focus routing improvements.

---

### Phase 0: Baseline & guardrails (small, safe)
- **Freeze UI constants**: create `src/solid/theme.ts` mirroring `src/theme.ts` (BOX + colors).
- **Define acceptance checks** (manual + minimal automated):
  - Tabs render in gold style
  - Jobs list selection uses background highlight + description column
  - Create Job flow reachable with `n` and works end-to-end
  - Job details update live while running
  - Agent pane still works

---

### Phase 1: Visual parity primitives (high leverage)
- **Tabs/key-hints parity**
  - Add a `KeyHint` component in `src/solid/components/KeyHint.tsx` that renders exactly `Description (key)` with active/inactive colors.
  - Replace current `[b] Jobs`/etc labels with KeyHint-style labels, *while keeping* `[shift+g] Agent` as an additional hint.

- **List selection parity (biggest delta)**
  - Introduce a reusable `ListPanel` component for Solid that supports:
    - **selectedBackgroundColor / selectedTextColor**
    - **focusedBackgroundColor / focusedTextColor**
    - **description column**
    - **scroll indicator**
  - Wire Jobs/Logs/Events to this component so we match gold visuals without duplicating logic.

Deliverable: the migrated UI *looks* like gold for left sidebar + tabs.

---

### Phase 2: Code structure parity (modularization like RT branch)
- **Split `src/solid/app.tsx`**
  - Extract “renderers” to folders analogous to RT branch:
    - `src/solid/ui/list-panels/*` (jobs list, logs list)
    - `src/solid/ui/detail-panels/*` (jobs detail, logs detail)
    - `src/solid/ui/render.ts` (single “sync UI from state” entry)
  - Keep `SolidShell` focused on layout + routing only.

- **Centralize formatting**
  - Port `job-card` formatting (`status | date`) into `src/solid/formatters/job-card.ts`
  - Port `job-details` formatting improvements (status-aware timestamps)

Deliverable: SolidJS TUI becomes maintainable + parity work becomes “swap implementations” not “edit the mega-file”.

---

### Phase 3: Feature parity from `origin/feat/job-details` (core behavior)
- **Job details SSE streaming**
  - Port `api/job-details-stream.ts` into `src/solid/api/job-details-stream.ts`
  - Hook into selection changes:
    - when job changes: disconnect old stream, connect new stream, use `since_seq`
    - update events/details/results/metrics view model incrementally
  - Ensure disconnect on app shutdown/modal transitions if needed.

- **Keyboard + focus routing parity**
  - Bring over the “list panel id” idea (jobs vs logs) so `g` can switch to logs while keeping left list focus.
  - Ensure `n` opens Create Job (global shortcut), and doesn’t conflict with Agent pane.

Deliverable: live-updating job details + smoother navigation behavior matching gold.

---

### Phase 4: Create New Job parity (wizard + deploy)
- **Port the wizard UX**
  - Bring over the step engine + LocalAPI scanning + manual path entry.
  - Integrate deploy workflow (NDJSON parsing, status updates).
  - Integrate with logs/task-apps panels so deploy feels “native”.

- **Polish**
  - Ensure errors are visible + copyable (error box pattern).
  - Ensure “active deployment” affects logs list/detail.

Deliverable: gold Create Job flow in SolidJS, reachable from tabs/keyboard, integrated into the rest of the UI.

---

### Phase 5: Cleanup & hardening
- **Remove duplicate/legacy paths**
  - Consolidate old imperative UI-only helpers that are superseded.
- **Tests**
  - Add a small set of deterministic tests for:
    - job card formatting
    - job details formatting
    - SSE parser behavior (feed sample SSE chunks → events)
- **Performance**
  - Ensure streaming/events don’t cause full-screen rerenders unnecessarily (only update affected panes).

---

### Suggested PR slicing (so merges are easy)
1. Theme + KeyHint tabs parity
2. ListPanel component + migrate Jobs sidebar
3. Migrate Logs sidebar + focus routing
4. Detail panels modularization
5. SSE job-details stream integration
6. Create Job wizard + deploy integration
7. Cleanup + tests

If you tell me what you want to prioritize first (pure visuals vs job-details SSE vs create-job), I can reorder the phases/PRs accordingly.