--------------------------- MODULE RunControl ---------------------------
EXTENDS Naturals, Sequences

(*
 * TLA+ specification for Swarms public run control, idempotency, monotonic
 * limits_revision, and absorbing terminal / finality invariants (SYN-6050).
 *
 * Compatible with `everystate conform -trace <run.jsonl> RunControl.tla`.
 *)

VARIABLES
    run_state,
    limits_revision,
    execution_closed,
    accounting_final,
    cleanup_complete,
    message_count,
    idempotent_replays,
    idempotency_conflicts

vars == <<
    run_state,
    limits_revision,
    execution_closed,
    accounting_final,
    cleanup_complete,
    message_count,
    idempotent_replays,
    idempotency_conflicts
>>

ActiveStates == {"queued", "running", "pausing", "paused", "cancelling"}
TerminalStates == {"completed", "partial", "failed", "cancelled"}
AllStates == {"absent"} \cup ActiveStates \cup TerminalStates

TypeOK ==
    /\ run_state \in AllStates
    /\ limits_revision \in Nat
    /\ execution_closed \in BOOLEAN
    /\ accounting_final \in BOOLEAN
    /\ cleanup_complete \in BOOLEAN
    /\ message_count \in Nat
    /\ idempotent_replays \in Nat
    /\ idempotency_conflicts \in Nat

SafetyInvariant ==
    /\ TypeOK
    /\ (cleanup_complete => execution_closed)
    /\ (accounting_final => execution_closed)
    /\ (run_state \in TerminalStates => execution_closed)
    /\ (run_state = "absent" =>
            /\ limits_revision = 0
            /\ ~execution_closed
            /\ ~accounting_final
            /\ ~cleanup_complete
            /\ message_count = 0)

Init ==
    /\ run_state = "absent"
    /\ limits_revision = 0
    /\ execution_closed = FALSE
    /\ accounting_final = FALSE
    /\ cleanup_complete = FALSE
    /\ message_count = 0
    /\ idempotent_replays = 0
    /\ idempotency_conflicts = 0

CreateRun ==
    /\ run_state = "absent"
    /\ run_state' = "queued"
    /\ limits_revision' = 1
    /\ UNCHANGED <<execution_closed, accounting_final, cleanup_complete,
                   message_count, idempotent_replays, idempotency_conflicts>>

StartRun ==
    /\ run_state = "queued"
    /\ run_state' = "running"
    /\ UNCHANGED <<limits_revision, execution_closed, accounting_final,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

ReplaySameKey ==
    /\ run_state # "absent"
    /\ idempotent_replays' = idempotent_replays + 1
    /\ UNCHANGED <<run_state, limits_revision, execution_closed,
                   accounting_final, cleanup_complete, message_count,
                   idempotency_conflicts>>

ReplayMutatedKeyRefused ==
    /\ run_state # "absent"
    /\ idempotency_conflicts' = idempotency_conflicts + 1
    /\ UNCHANGED <<run_state, limits_revision, execution_closed,
                   accounting_final, cleanup_complete, message_count,
                   idempotent_replays>>

ObserveSnapshot ==
    /\ run_state # "absent"
    /\ UNCHANGED vars

PauseRun ==
    /\ run_state \in {"queued", "running", "pausing"}
    /\ run_state' = "paused"
    /\ UNCHANGED <<limits_revision, execution_closed, accounting_final,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

ResumeRun ==
    /\ run_state \in {"pausing", "paused"}
    /\ run_state' = "running"
    /\ UNCHANGED <<limits_revision, execution_closed, accounting_final,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

UpdateLimits ==
    /\ run_state \in ActiveStates
    /\ limits_revision' = limits_revision + 1
    /\ UNCHANGED <<run_state, execution_closed, accounting_final,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

UpdateLimitsAndResume ==
    /\ run_state \in {"pausing", "paused"}
    /\ run_state' = "running"
    /\ limits_revision' = limits_revision + 1
    /\ UNCHANGED <<execution_closed, accounting_final, cleanup_complete,
                   message_count, idempotent_replays, idempotency_conflicts>>

RefuseStaleLimits ==
    /\ run_state # "absent"
    /\ UNCHANGED vars

SendMessage ==
    /\ run_state \in ActiveStates
    /\ message_count' = message_count + 1
    /\ UNCHANGED <<run_state, limits_revision, execution_closed,
                   accounting_final, cleanup_complete, idempotent_replays,
                   idempotency_conflicts>>

CancelRun ==
    /\ run_state \in ActiveStates
    /\ run_state' = "cancelling"
    /\ UNCHANGED <<limits_revision, execution_closed, accounting_final,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

CloseExecutionCancelled ==
    /\ run_state = "cancelling"
    /\ run_state' = "cancelled"
    /\ execution_closed' = TRUE
    /\ UNCHANGED <<limits_revision, accounting_final, cleanup_complete,
                   message_count, idempotent_replays, idempotency_conflicts>>

CloseExecutionCompleted ==
    /\ run_state = "running"
    /\ run_state' = "completed"
    /\ execution_closed' = TRUE
    /\ UNCHANGED <<limits_revision, accounting_final, cleanup_complete,
                   message_count, idempotent_replays, idempotency_conflicts>>

FinalizeAccounting ==
    /\ execution_closed
    /\ ~accounting_final
    /\ accounting_final' = TRUE
    /\ UNCHANGED <<run_state, limits_revision, execution_closed,
                   cleanup_complete, message_count, idempotent_replays,
                   idempotency_conflicts>>

CompleteCleanup ==
    /\ execution_closed
    /\ ~cleanup_complete
    /\ cleanup_complete' = TRUE
    /\ UNCHANGED <<run_state, limits_revision, execution_closed,
                   accounting_final, message_count, idempotent_replays,
                   idempotency_conflicts>>

RefuseTerminalMutation ==
    /\ run_state \in TerminalStates
    /\ UNCHANGED vars

Next ==
    \/ CreateRun
    \/ StartRun
    \/ ReplaySameKey
    \/ ReplayMutatedKeyRefused
    \/ ObserveSnapshot
    \/ PauseRun
    \/ ResumeRun
    \/ UpdateLimits
    \/ UpdateLimitsAndResume
    \/ RefuseStaleLimits
    \/ SendMessage
    \/ CancelRun
    \/ CloseExecutionCancelled
    \/ CloseExecutionCompleted
    \/ FinalizeAccounting
    \/ CompleteCleanup
    \/ RefuseTerminalMutation

Spec == Init /\ [][Next]_vars

=============================================================================
