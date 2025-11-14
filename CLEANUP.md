# Synth-AI Cleanup Plan

## Overview
This document tracks cleanup tasks for the synth-ai codebase, including code quality improvements, test fixes, and removal of unnecessary files.

## Tasks

### 1. Diff Against Nightly
- [ ] Identify key changes made in this branch
- [ ] Review for dropped critical functionality
- [ ] Check for regressions

### 2. Code Quality Issues

#### Critical Functionality Review
- [ ] Check for dropped critical functionality
- [ ] Verify all core features still work

#### Code Duplication
- [ ] Identify egregious duplication
- [ ] Refactor duplicated code

#### Bad Abstractions & Sloppy Implementation
- [ ] Review abstractions for clarity and correctness
- [ ] Fix sloppy implementations

#### Data Structures & Naming
- [ ] Replace dicts with dataclasses where appropriate
- [ ] Fix bad naming conventions
- [ ] Remove bug-prone practices

#### Test Coverage
- [ ] Identify untested logic
- [ ] Add tests for critical paths

### 3. Test Suite Review
- [ ] Run unit tests
- [ ] Identify failing tests
- [ ] Determine which need updating/removing
- [ ] Identify real regressions

### 4. File Cleanup
- [ ] Identify loose .txt files to remove
- [ ] Identify loose .md files to remove
- [ ] Remove other junk files

### 5. Linting & Type Checking
- [ ] Run `uvx ruff check` and fix violations
- [ ] Run `uvx ty check` and fix violations
- [ ] Get both to 0 violations

## Findings

### Critical Functionality Issues
- **Experiment Queue Tests**: Tests fail due to missing `EXPERIMENT_QUEUE_DB_PATH` env var - need to mock or set env var in tests

### Code Duplication
- **Langprobe Examples**: Extensive duplication in `examples/blog_posts/langprobe/task_specific/` - banking77, heartdisease, hotpotqa all have similar adapters, TUI, plotting code
- **Note**: Result files are intentionally kept - they're data, not code duplication

### Untested Logic
- **Artifacts CLI**: New code has good test coverage (59 tests passing)
- **Experiment Queue**: Integration tests fail due to env var requirements
- Need to check coverage for other new features

### Test Failures
- **✅ FIXED**: Experiment queue tests - Fixed lazy celery app to allow decorator evaluation without env var
- **42 tests passing** in experiment_queue test suite
- Need to run full test suite to identify other failures

### Files to Archive (Move to `old/` folder, gitignored)
**Planning .txt files to archive:**
- `queue_notes.txt` - planning notes
- `experiment_queue.txt` - planning notes  
- `experiment_queue_plan.txt` - planning notes
- `scan_plan.txt` - planning notes
- `issues.txt` - issue tracking (should be in GitHub)
- `./temp/integration_failures/*.txt` - temporary failure logs (25+ files)
- `synth_ai/demos/core/cmd_deploy.txt` - demo commands
- `synth_ai/demos/core/demo_commands.txt` - demo commands
- `examples/blog_posts/gepa/in-process-implementation-plan.txt` - planning notes
- `examples/blog_posts/gepa/in-process-task-app.txt` - planning notes

**Planning .md files to archive:**
- `CLEANUP_SUMMARY.md` - duplicate of this file?
- `REMAINING_WORK.md` - planning notes
- `LANGPROBE_BRANCH_REVIEW.md` - branch review notes
- `sdk_refactor_plan.md` - planning notes
- `langprobe.md` - planning notes
- `V0_REMOVAL_PLAN.md` - completed work?
- `DEPENDENCY_ANALYSIS.md` - analysis notes
- Many `*_ROAST.md`, `*_REVIEW.md`, `*_PLAN.md` files in examples

**Note**: Result files in `examples/blog_posts/langprobe/task_specific/*/results/*.txt` should be KEPT - they're data, not planning docs

### Linting Violations
**Ruff found 132 errors initially:**
- **✅ FIXED**: 80 auto-fixed violations
- **✅ FIXED**: 2 manual fixes (zip strict, unused variable)
- **Remaining**: 51 errors (down from 132)
- 11 B904: raise-without-from-inside-except
- 8 F841: unused-variable
- 7 UP015: redundant-open-modes (fixable)
- 6 UP037: quoted-annotation (fixable)
- 5 E402: module-import-not-at-top-of-file
- 5 SIM102: collapsible-if
- 5 SIM108: if-else-block-instead-of-if-exp
- 3 B007: unused-loop-control-variable
- 3 SIM105: suppressible-exception
- 2 C414: unnecessary-double-cast-or-process
- 2 F811: redefined-while-unused (fixable)
- 2 SIM115: open-file-with-context-handler
- 2 SIM118: in-dict-keys
- 2 UP017: datetime-timezone-utc (fixable)
- 1 B028: no-explicit-stacklevel
- 1 B905: zip-without-explicit-strict
- 1 F823: undefined-local
- 1 N806: non-lowercase-variable-in-function

**80 fixable** with `--fix` option
**24 hidden fixes** can be enabled with `--unsafe-fixes`

### Key Changes vs Nightly
- **New Artifacts CLI**: Complete new command suite (`synth-ai artifacts`)
- **Langprobe Examples**: Large addition of example code and results
- **Experiment Queue**: New queue management features
- **Session Pricing**: New pricing features

## Action Items

### Priority 1: Critical Fixes
1. **✅ Fix Experiment Queue Tests**: Fixed lazy celery app to allow decorator evaluation
2. **✅ Fix Ruff Violations**: Auto-fixed 80 violations, manually fixed 2 more
3. **Fix Remaining Ruff Violations**: 51 errors remaining (mostly B904, F841, E402, SIM rules)
4. **Fix Type Check Violations**: Address type errors found by `ty check`

### Priority 2: Code Quality
1. **✅ Experiment Queue**: Fixed lazy celery app decorator support - all tests pass

### Priority 3: Cleanup
1. **`old/` folder exists**: Already created and gitignored (line 99 in `.gitignore`)
2. **Move Planning Files**: Move all planning `.txt` and `.md` files to `old/` folder:
   - Root-level planning files (queue_notes.txt, experiment_queue.txt, etc.)
   - `temp/integration_failures/*.txt` files
   - Planning docs in examples directories (`*_ROAST.md`, `*_REVIEW.md`, `*_PLAN.md`)
3. **Review Documentation**: Consolidate duplicate README files (keep, just organize)

### Priority 4: Testing
1. **Fix Failing Tests**: Address experiment queue test failures
2. **Add Missing Tests**: Ensure all new features have test coverage
3. **Update Test Fixtures**: Fix env var requirements in tests

