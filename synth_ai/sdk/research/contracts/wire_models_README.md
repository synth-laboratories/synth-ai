# Models

This subtree owns the canonical public Python data types for Managed Research.

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

Run-control models live in [`run_timeline.py`](run_timeline.py):
- `SmrLogicalTimeline`
- `SmrLogicalTimelineNode`
- `SmrBranchMode`
- `SmrRunBranchRequest`
- `SmrRunBranchResponse`

High-signal typed response models live in [`types.py`](types.py):
- `ProjectSetupAuthority`
- `LaunchPreflight`
- `RunProgress`
- `SemanticProgressSnapshot`
- `WorkspaceInputsState`
- `WorkspaceUploadResult`
