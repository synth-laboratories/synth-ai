# Budget Sizing and Launch Choices Reference

Use this reference alongside `skill://swarms-launch/SKILL.md` when sizing `limits`, choosing profiles or role assignments, or recovering from admission refusals.

## Key Launch Fields

- `limits.maximum_charge`: `{amount: "<decimal>", currency: "USD"}`. Caps total run spend and funds active reservations for the planner, reviewer, and parallel workers.
- `limits.wall_time_seconds`: maximum wall-clock duration before `run_deadline`.
- `limits.parallel_workers`: number of concurrent workers. Each worker holds an equal slice of the budget.
- `launch_menu_revision`: binds the launch to the `menu_revision` snapshot returned by `swarms_launch_choices`.
- `profile` / `role_assignments`: choose a profile from `swarms_profiles_list` or `swarms_launch_choices`; override per-role inference/sandbox profiles only with entries currently advertised in `roles`, `inference_profiles`, and `sandbox_profiles`.
- `domicile_requirement`: pass to `swarms_launch_choices` first so ineligible profiles appear in `exclusions` rather than failing at launch.

## Common Launch Admission Refusals

| Code | Cause | Recovery |
| --- | --- | --- |
| `launch_menu_revision_stale` | Offered models/pricing changed since `swarms_launch_choices`. | Re-call `swarms_launch_choices`, update `launch_menu_revision`, and launch with a new `idempotency_key`. |
| `launch_choice_unaffordable` | Selected profile or role assignments exceed `limits.maximum_charge`. | Raise `limits.maximum_charge` or pick a lower-cost choice from `swarms_launch_choices`, then retry with a new key. |
| `launch_budget_below_actor_floor` | `maximum_charge` cannot fund planner + reviewer + all `parallel_workers`. | Raise `maximum_charge` above the micro-USD floor in `detail` or reduce `parallel_workers`, then retry with a new key. |
| `run_budget_below_call_bound` | `maximum_charge` is below the single-call minimum (`minimum_budget_micros`). | Raise `limits.maximum_charge` to at least `minimum_budget_micros` and launch with a new key. |
| `insufficient_credits` / `budget_exceeded` | Account headroom cannot cover the reservation. | Consult `skill://swarms-billing/SKILL.md` to inspect balance/holds or top up credits. |
| `provider_model_unavailable` / `provider_preset_unavailable` | Model or preset is no longer offered. | Refresh `swarms_launch_choices` and select an offered option. |
| `launch_request_too_large` | Serialized body exceeds the launch payload limit. | Upload long context via `swarms_upload_file` or `swarms_upload_create` and pass `file_id` in `inputs`. |
| `run_queue_full` (429) / `overloaded` (503) | Transient capacity pressure. | Retry only if the Problem sets `retryable: true`, honoring `Retry-After`. |
