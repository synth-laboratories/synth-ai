-------------------------- MODULE HostedOptions ---------------------------
EXTENDS Naturals

(*
 * TLA+ specification for hosted options consistency and supplier-neutrality
 * across /v1/launch-choices -> /v1/runs/preflight -> POST /v1/runs (SYN-6050).
 *
 * Compatible with `everystate conform -trace <run.jsonl> HostedOptions.tla`.
 *)

VARIABLES
    menu_queries,
    preflight_checks,
    launch_attempts,
    vendor_leaks_detected,
    agreement_violations,
    last_menu_admissible,
    last_preflight_admissible,
    last_launch_accepted

vars == <<
    menu_queries,
    preflight_checks,
    launch_attempts,
    vendor_leaks_detected,
    agreement_violations,
    last_menu_admissible,
    last_preflight_admissible,
    last_launch_accepted
>>

TypeOK ==
    /\ menu_queries \in Nat
    /\ preflight_checks \in Nat
    /\ launch_attempts \in Nat
    /\ vendor_leaks_detected \in Nat
    /\ agreement_violations \in Nat
    /\ last_menu_admissible \in BOOLEAN
    /\ last_preflight_admissible \in BOOLEAN
    /\ last_launch_accepted \in BOOLEAN

SafetyInvariant ==
    /\ TypeOK
    /\ vendor_leaks_detected = 0
    /\ agreement_violations = 0

Init ==
    /\ menu_queries = 0
    /\ preflight_checks = 0
    /\ launch_attempts = 0
    /\ vendor_leaks_detected = 0
    /\ agreement_violations = 0
    /\ last_menu_admissible = FALSE
    /\ last_preflight_admissible = FALSE
    /\ last_launch_accepted = FALSE

QueryLaunchChoices ==
    /\ menu_queries' = menu_queries + 1
    /\ last_menu_admissible' \in BOOLEAN
    /\ UNCHANGED <<preflight_checks, launch_attempts, vendor_leaks_detected,
                   agreement_violations, last_preflight_admissible,
                   last_launch_accepted>>

PreflightAdmissible ==
    /\ menu_queries > 0
    /\ preflight_checks' = preflight_checks + 1
    /\ last_preflight_admissible' = TRUE
    /\ UNCHANGED <<menu_queries, launch_attempts, vendor_leaks_detected,
                   agreement_violations, last_menu_admissible,
                   last_launch_accepted>>

PreflightRefused ==
    /\ menu_queries > 0
    /\ preflight_checks' = preflight_checks + 1
    /\ last_preflight_admissible' = FALSE
    /\ UNCHANGED <<menu_queries, launch_attempts, vendor_leaks_detected,
                   agreement_violations, last_menu_admissible,
                   last_launch_accepted>>

LaunchAccepted ==
    /\ preflight_checks > 0
    /\ last_preflight_admissible = TRUE
    /\ launch_attempts' = launch_attempts + 1
    /\ last_launch_accepted' = TRUE
    /\ UNCHANGED <<menu_queries, preflight_checks, vendor_leaks_detected,
                   agreement_violations, last_menu_admissible,
                   last_preflight_admissible>>

LaunchRefused ==
    /\ menu_queries > 0
    /\ launch_attempts' = launch_attempts + 1
    /\ last_launch_accepted' = FALSE
    /\ UNCHANGED <<menu_queries, preflight_checks, vendor_leaks_detected,
                   agreement_violations, last_menu_admissible,
                   last_preflight_admissible>>

Next ==
    \/ QueryLaunchChoices
    \/ PreflightAdmissible
    \/ PreflightRefused
    \/ LaunchAccepted
    \/ LaunchRefused

Spec == Init /\ [][Next]_vars

=============================================================================
