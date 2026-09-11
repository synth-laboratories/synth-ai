"""The agent-facing Intern grant matrix: what a running Intern may call.

This is the *agent* MCP surface, which is a different thing from the public
``synth-research`` MCP server in :mod:`synth_ai.mcp.research`. The public server
is an operator holding an API key. This module describes what the Intern itself
is granted against the SMR plane while it runs, so callers that build or audit
grants -- eval drivers, onboarding, grant review -- can do so without importing
the backend.

Three rules are structural, not advisory, and are enforced by
:func:`validate_intern_tool_grant`:

1. **Effort-first.** The Intern's own objectives, milestones, tasks, claims, and
   links are Effort-bound and live in the Intern store. They cross no authority,
   so they ride the generic ``call_tool`` action kind. The specialized action
   kinds exist to gate authority-crossing launches.
2. **Kickoff and poll, never ownership.** The Intern starts a swarm run and
   polls it using the SMR tools it is already granted. It never plans a swarm's
   task graph. Any tool whose name reads as swarm task-graph planning is refused
   with :data:`INTERN_SWARM_PLAN_TASKS_OWNERSHIP_FORBIDDEN` *before* the
   registry is consulted, so the refusal names the boundary instead of looking
   like a missing registration.
3. **Never the swarm run task tables.** The Intern planner writes its own six
   tables and nothing else.

This is a vocabulary mirror, which is why it lives beside the other backend
vocabulary mirrors in ``contracts/`` rather than on the hero API surface.

# See: backend packages/intern/smr_mcp.py + packages/intern/core/capabilities.py
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

#: MCP server name the Intern addresses SMR under.
INTERN_SMR_MCP_SERVER_NAME = "smr"

# -- Error codes ------------------------------------------------------------

#: Raised when the Intern proposes a swarm task-graph planning tool as owner.
INTERN_SWARM_PLAN_TASKS_OWNERSHIP_FORBIDDEN = "intern_swarm_plan_tasks_ownership_forbidden"
#: Raised when a proposed tool name is not in the grant matrix at all.
INTERN_SMR_MCP_TOOL_NOT_REGISTERED = "intern_smr_mcp_tool_not_registered"
#: Raised when a registered tool is proposed under the wrong action kind.
INTERN_SMR_MCP_ACTION_TOOL_MISMATCH = "intern_smr_mcp_action_tool_mismatch"
#: Raised when the requested capability is not the one the tool is registered to.
INTERN_SMR_MCP_TOOL_CAPABILITY_MISMATCH = "intern_smr_mcp_tool_capability_mismatch"


class InternGrantError(ValueError):
    """A grant-matrix refusal, carrying the wire error code as ``code``."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class InternCapabilityOperation(StrEnum):
    """One authority the Intern may be granted. Values match the backend."""

    AGENT_INVOKE = "agent.invoke"
    MCP_DISCOVER = "mcp.discover"
    MCP_RESOURCE_READ = "mcp.resource.read"
    FACTORY_READ = "factory.read"
    FACTORY_CREATE = "factory.create"
    FACTORY_ATTACH = "factory.attach"
    FACTORY_ARCHIVE = "factory.archive"
    PROJECT_CREATE = "project.create"
    FILE_CREATE = "file.create"
    FILE_READ = "file.read"
    FILE_UPDATE = "file.update"
    FILE_DELETE = "file.delete"
    PROJECT_FILE_READ = "project.file.read"
    PROJECT_FILE_WRITE = "project.file.write"
    PROJECT_GIT_BIND = "project.git.bind"
    HARNESS_DOWNLOAD = "harness.download"
    EFFORT_CREATE = "effort.create"
    EFFORT_WAKE = "effort.wake"
    RUN_DEPLOY = "run.deploy"
    RUN_READ = "run.read"
    RUN_PAUSE = "run.pause"
    RUN_RESUME = "run.resume"
    RUN_INTERVENE = "run.intervene"
    RUN_CANCEL = "run.cancel"
    SWARM_DEPLOY = "swarm.deploy"
    SWARM_READ = "swarm.read"
    SWARM_STOP = "swarm.stop"
    MANDERQUEUE_PUBLISH = "manderqueue.publish"
    EVIDENCE_READ = "evidence.read"
    EVIDENCE_PUBLISH = "evidence.publish"
    EXPERIMENT_READ = "experiment.read"
    EXPERIMENT_PROPOSE = "experiment.propose"
    CANDIDATE_PUBLISH = "candidate.publish"
    REPORT_PUBLISH = "report.publish"
    VISUAL_READ = "visual.read"
    VISUAL_CREATE = "visual.create"
    VISUAL_UPDATE = "visual.update"
    VISUAL_PUBLISH = "visual.publish"
    VISUAL_REPAIR = "visual.repair"
    TRACE_READ = "trace.read"
    TRACE_INGEST = "trace.ingest"
    POOL_READ = "pool.read"
    POOL_ASSIGN = "pool.assign"
    POOL_RELEASE = "pool.release"
    POOL_REMOUNT = "pool.remount"
    POOL_ROLLOUT = "pool.rollout"
    GRADING_REQUEST = "grading.request"
    SPINE_HANDOFF = "intern.spine.handoff"
    MEMORY_READ = "intern.memory.read"
    # Synth-wiki project memory. Distinct from intern.memory.* (the spine).
    WIKI_READ = "wiki.read"
    WIKI_PROPOSE = "wiki.propose"
    WIKI_LIBRARIAN = "wiki.librarian"
    # Effort-primary Async plus the subordinate Intern objective/milestone/task
    # store. Every row these write is Effort-bound and lives in the Intern
    # store; none of them crosses into SMR write authority.
    EFFORT_UPDATE = "effort.update"
    INTERN_OBJECTIVE_READ = "intern.objective.read"
    INTERN_OBJECTIVE_WRITE = "intern.objective.write"
    INTERN_MILESTONE_READ = "intern.milestone.read"
    INTERN_MILESTONE_WRITE = "intern.milestone.write"
    INTERN_TASK_READ = "intern.task.read"
    INTERN_TASK_WRITE = "intern.task.write"


class InternMcpActionKind(StrEnum):
    """How a proposed tool call is classified for gating."""

    DISCOVER_TOOLS = "discover_tools"
    CALL_TOOL = "call_tool"
    READ_RESOURCE = "read_resource"
    CREATE_FACTORY = "create_factory"
    ATTACH_FACTORY = "attach_factory"
    DETACH_FACTORY = "detach_factory"
    CREATE_PROJECT = "create_project"
    CREATE_EFFORT = "create_effort"
    START_FACTORY_RUN = "start_factory_run"
    INSPECT_FACTORY_RUN = "inspect_factory_run"
    DEPLOY_SWARM = "deploy_swarm"
    INSPECT_SWARM = "inspect_swarm"
    STOP_SWARM = "stop_swarm"
    COLLECT_EVIDENCE = "collect_evidence"


# -- The Effort-first program surface --------------------------------------

#: E1/E2 plus I1-I6: the Effort-first tools the running Intern gained for the
#: Effort-primary cutover. Effort listing reuses ``FACTORY_READ`` -- listing a
#: Factory's Efforts is a Factory read, not a new authority.
INTERN_EFFORT_PROGRAM_TOOL_CAPABILITIES: Mapping[str, InternCapabilityOperation] = MappingProxyType(
    {
        "smr_effort_list": InternCapabilityOperation.FACTORY_READ,
        "smr_effort_update": InternCapabilityOperation.EFFORT_UPDATE,
        "smr_intern_objective_create": InternCapabilityOperation.INTERN_OBJECTIVE_WRITE,
        "smr_intern_objective_list": InternCapabilityOperation.INTERN_OBJECTIVE_READ,
        "smr_intern_objective_get": InternCapabilityOperation.INTERN_OBJECTIVE_READ,
        "smr_intern_objective_update": InternCapabilityOperation.INTERN_OBJECTIVE_WRITE,
        "smr_intern_progress_claim_create": InternCapabilityOperation.INTERN_OBJECTIVE_WRITE,
        "smr_intern_effort_progress": InternCapabilityOperation.INTERN_OBJECTIVE_READ,
        "smr_intern_milestone_create": InternCapabilityOperation.INTERN_MILESTONE_WRITE,
        "smr_intern_milestone_list": InternCapabilityOperation.INTERN_MILESTONE_READ,
        "smr_intern_milestone_transition": InternCapabilityOperation.INTERN_MILESTONE_WRITE,
        "smr_intern_task_create": InternCapabilityOperation.INTERN_TASK_WRITE,
        "smr_intern_task_list": InternCapabilityOperation.INTERN_TASK_READ,
        "smr_intern_task_update": InternCapabilityOperation.INTERN_TASK_WRITE,
        "smr_intern_objective_link_create": InternCapabilityOperation.INTERN_OBJECTIVE_WRITE,
        "smr_intern_objective_link_list": InternCapabilityOperation.INTERN_OBJECTIVE_READ,
    }
)

#: The capabilities the Effort-first program surface introduced.
INTERN_EFFORT_PROGRAM_CAPABILITIES: frozenset[InternCapabilityOperation] = frozenset(
    {
        InternCapabilityOperation.EFFORT_UPDATE,
        InternCapabilityOperation.INTERN_OBJECTIVE_READ,
        InternCapabilityOperation.INTERN_OBJECTIVE_WRITE,
        InternCapabilityOperation.INTERN_MILESTONE_READ,
        InternCapabilityOperation.INTERN_MILESTONE_WRITE,
        InternCapabilityOperation.INTERN_TASK_READ,
        InternCapabilityOperation.INTERN_TASK_WRITE,
    }
)

# -- Kickoff and poll (I7): existing tools, no new ones --------------------

#: The Intern is an MCP *client* of the swarm plane. Kicking a run off and
#: polling it reuses tools the Intern already had; the cutover added none.
#: The Intern-side fold is ``smr_intern_progress_claim_create``.
INTERN_KICKOFF_AND_POLL_TOOLS: Mapping[str, InternCapabilityOperation] = MappingProxyType(
    {
        "smr_project_trigger_run": InternCapabilityOperation.SWARM_DEPLOY,
        "smr_factory_wake_due": InternCapabilityOperation.EFFORT_WAKE,
        "smr_run_get": InternCapabilityOperation.RUN_READ,
        "smr_swarm_status": InternCapabilityOperation.SWARM_READ,
        "smr_swarm_activity": InternCapabilityOperation.SWARM_READ,
        "smr_swarm_evidence": InternCapabilityOperation.EVIDENCE_READ,
        "smr_work_product_list": InternCapabilityOperation.EVIDENCE_READ,
        "smr_work_product_get": InternCapabilityOperation.EVIDENCE_READ,
        "smr_run_stop": InternCapabilityOperation.SWARM_STOP,
    }
)

#: Action kind each kickoff/poll tool must be proposed under.
_KICKOFF_ACTION_KINDS: Mapping[str, InternMcpActionKind] = MappingProxyType(
    {
        "smr_project_trigger_run": InternMcpActionKind.DEPLOY_SWARM,
        "smr_factory_wake_due": InternMcpActionKind.START_FACTORY_RUN,
        "smr_run_get": InternMcpActionKind.INSPECT_FACTORY_RUN,
        "smr_swarm_status": InternMcpActionKind.INSPECT_SWARM,
        "smr_swarm_activity": InternMcpActionKind.INSPECT_SWARM,
        "smr_swarm_evidence": InternMcpActionKind.COLLECT_EVIDENCE,
        "smr_work_product_list": InternMcpActionKind.CALL_TOOL,
        "smr_work_product_get": InternMcpActionKind.CALL_TOOL,
        "smr_run_stop": InternMcpActionKind.STOP_SWARM,
    }
)

#: Every Effort-program tool rides the generic call kind -- Intern-store CRUD
#: crosses no authority, so it needs no specialized gate.
INTERN_TOOL_ACTION_KINDS: Mapping[str, InternMcpActionKind] = MappingProxyType(
    {
        **dict.fromkeys(INTERN_EFFORT_PROGRAM_TOOL_CAPABILITIES, InternMcpActionKind.CALL_TOOL),
        **_KICKOFF_ACTION_KINDS,
    }
)

# -- The ownership boundary ------------------------------------------------

#: Any tool name containing this substring is swarm task-graph planning.
INTERN_OWNERSHIP_DENIED_TOOL_SUBSTRING = "plan_tasks"

#: Names refused outright. The substring rule already covers the planners; the
#: run-task writers are listed because their names do not contain it.
INTERN_OWNERSHIP_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "plan_tasks",
        "smr_plan_tasks",
        "smr_run_plan_tasks",
        "smr_swarm_plan_tasks",
        "smr_run_task_create",
        "smr_run_task_update",
    }
)


def reject_intern_ownership_denied_tool(tool_name: str) -> None:
    """Refuse swarm task-graph planning proposed by the Intern as owner.

    Args:
        tool_name: Proposed tool name.

    Raises:
        InternGrantError: With ``intern_swarm_plan_tasks_ownership_forbidden``
            when the name is a swarm planner or a swarm run-task writer.
    """
    normalized = str(tool_name or "").strip().lower()
    if not normalized:
        return
    if (
        normalized in INTERN_OWNERSHIP_DENIED_TOOLS
        or INTERN_OWNERSHIP_DENIED_TOOL_SUBSTRING in normalized
    ):
        raise InternGrantError(INTERN_SWARM_PLAN_TASKS_OWNERSHIP_FORBIDDEN)


def validate_intern_tool_grant(
    *,
    tool_name: str,
    action_kind: InternMcpActionKind | str | None = None,
    requested_capability: InternCapabilityOperation | str | None = None,
) -> InternCapabilityOperation:
    """Resolve the capability one Intern tool call requires, or refuse it.

    The ownership check runs first, so proposing a swarm planner names the
    boundary rather than degrading into "tool not registered".

    Args:
        tool_name: Proposed tool name.
        action_kind: Action kind the caller classified the call under. When
            given, it must match the kind the tool is registered to.
        requested_capability: Capability the caller believes it needs. When
            given, it must be the registered one.

    Returns:
        The registered ``InternCapabilityOperation``.

    Raises:
        InternGrantError: On an ownership violation, an unregistered tool, an
            action-kind mismatch, or a capability mismatch.
    """
    reject_intern_ownership_denied_tool(tool_name)
    operation = str(tool_name or "").strip()
    registered = INTERN_EFFORT_PROGRAM_TOOL_CAPABILITIES.get(
        operation
    ) or INTERN_KICKOFF_AND_POLL_TOOLS.get(operation)
    if registered is None:
        raise InternGrantError(INTERN_SMR_MCP_TOOL_NOT_REGISTERED)
    if action_kind is not None:
        expected_kind = INTERN_TOOL_ACTION_KINDS[operation]
        if InternMcpActionKind(str(action_kind)) is not expected_kind:
            raise InternGrantError(INTERN_SMR_MCP_ACTION_TOOL_MISMATCH)
    if (
        requested_capability is not None
        and InternCapabilityOperation(str(requested_capability)) is not registered
    ):
        raise InternGrantError(INTERN_SMR_MCP_TOOL_CAPABILITY_MISMATCH)
    return registered


def intern_effort_program_grant(
    *,
    include_kickoff_and_poll: bool = True,
) -> frozenset[InternCapabilityOperation]:
    """The capability set an Effort-first Intern needs.

    Args:
        include_kickoff_and_poll: Also grant the read/deploy/stop capabilities
            the Intern uses to kick a swarm run off and poll it. This does not
            grant swarm task-graph ownership -- no capability does.

    Returns:
        A frozen capability set suitable for building or auditing a grant.
    """
    operations = set(INTERN_EFFORT_PROGRAM_CAPABILITIES)
    operations.add(InternCapabilityOperation.FACTORY_READ)
    if include_kickoff_and_poll:
        operations.update(INTERN_KICKOFF_AND_POLL_TOOLS.values())
    return frozenset(operations)


__all__ = [
    "INTERN_EFFORT_PROGRAM_CAPABILITIES",
    "INTERN_EFFORT_PROGRAM_TOOL_CAPABILITIES",
    "INTERN_KICKOFF_AND_POLL_TOOLS",
    "INTERN_OWNERSHIP_DENIED_TOOLS",
    "INTERN_OWNERSHIP_DENIED_TOOL_SUBSTRING",
    "INTERN_SMR_MCP_ACTION_TOOL_MISMATCH",
    "INTERN_SMR_MCP_SERVER_NAME",
    "INTERN_SMR_MCP_TOOL_CAPABILITY_MISMATCH",
    "INTERN_SMR_MCP_TOOL_NOT_REGISTERED",
    "INTERN_SWARM_PLAN_TASKS_OWNERSHIP_FORBIDDEN",
    "INTERN_TOOL_ACTION_KINDS",
    "InternCapabilityOperation",
    "InternGrantError",
    "InternMcpActionKind",
    "intern_effort_program_grant",
    "reject_intern_ownership_denied_tool",
    "validate_intern_tool_grant",
]
