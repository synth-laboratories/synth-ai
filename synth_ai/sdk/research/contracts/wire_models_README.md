# Models

This subtree owns the canonical public Python data types for `managed_research`.

What belongs here:
- public enums
- durable request/response dataclasses
- wire-to-model parsing helpers

What does not belong here:
- MCP request parsing
- transport logic
- endpoint-specific SDK calling code

Current guidance:
- keep typed models authoritative for as long as possible
- serialize to plain dicts only at the final transport edge
- prefer explicit dataclasses/enums over open-ended `dict[str, Any]` for primary concepts

New run-control models live in [`run_timeline.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/models/run_timeline.py):
- `SmrLogicalTimeline`
- `SmrLogicalTimelineNode`
- `SmrBranchMode`
- `SmrRunBranchRequest`
- `SmrRunBranchResponse`

High-signal typed response models currently live in [`types.py`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/models/types.py):
- `ProjectSetupAuthority`
- `LaunchPreflight`
- `RunProgress`
- `SemanticProgressSnapshot`
- `WorkspaceInputsState`
- `WorkspaceUploadResult`

Generated compatibility exports remain under [`generated/v1`](/Users/joshpurtell/Documents/GitHub/managed-research/managed_research/models/generated/v1/__init__.py).
