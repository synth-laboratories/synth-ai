# Forge discrepancy test index — 2026-10-06

Scope: the ranked audit and its tests-only handoff. Product code is unchanged.

Source pins: backend `7860732a2c1904c08d1a9eafe34a6c6f1eb50b6b`, synth-ai `67bc4c9ff7e4385dbf1b4fb759430bbe07b67ea9`, Forge `69815acd7f4e7781b12e041034c69b37cfd566e7`.

Verified: **191 known-red assertions**, **235 passing checks**, **1 slot journey blocked**. No xfail, CI wiring, paid effects, or provider calls. All 35 Tier 1/2 finding IDs have at least one red assertion for the stated reason. Finding IDs are assertions, not expected-failure marks.

The fresh bounded/full OpenAPI, hidden-route and enum fixtures are committed in synth-ai. FastAPI 0.128.0 was used; only the handoff’s generated visual-body and ValidationError noise is excluded. Backend and SDK source snapshots are the authority. See fixture provenance and README.

## Findings and selectors

| Finding | Owning repo / test selector | Verified status |
|---|---|---|
| MX-01 | synth-ai: `tests/contract/test_discrepancy_law.py::test_all_raw_sdk_calls_resolve_to_backend__MX09` | red |
| MX-01 | synth-ai: `tests/contract/test_discrepancy_law.py::test_sdk_call_resolves_to_backend__MX01_MX02` | red |
| MX-02 | synth-ai: `tests/contract/test_discrepancy_law.py::test_all_raw_sdk_calls_resolve_to_backend__MX09` | red |
| MX-02 | synth-ai: `tests/contract/test_discrepancy_law.py::test_sdk_call_resolves_to_backend__MX01_MX02` | red |
| MX-03 | synth-ai: `tests/contract/test_discrepancy_contracts.py::test_registry_matches_backend__MX03` | green |
| MX-04 | synth-ai: `tests/contract/test_discrepancy_law.py::test_full_vendored_snapshot_matches_backend__MX04` | red |
| MX-04 | synth-ai: `tests/contract/test_discrepancy_law.py::test_vendored_schema_matches_generated__MX04` | red |
| MX-05 | backend: `tests/units/test_forge_discrepancy_law.py::test_committed_backend_request_schema_matches_source__MX05` | red |
| MX-06 | synth-ai: `tests/contract/test_discrepancy_law.py::test_request_body_satisfies_backend__PR18_MX06` | red |
| MX-07b | synth-ai: `tests/contract/test_discrepancy_law.py::test_request_body_satisfies_backend__PR18_MX06` | red |
| MX-08 | synth-ai: `tests/contract/test_discrepancy_law.py::test_visual_backend_logical_response_parses__MX08` | red |
| MX-09 | synth-ai: `tests/contract/test_discrepancy_law.py::test_all_raw_sdk_calls_resolve_to_backend__MX09` | red |
| MX-10 | synth-ai: `tests/mcp/test_serialization_discrepancy_law.py::test_public_call_tool_returns_json__MX10_RR12` | red |
| MX-11 | synth-ai: `tests/contract/test_discrepancy_law.py::test_file_page_parses__PR01_MX11` | red |
| MX-12 | Forge: `tests/test_discrepancy_law.py::test_validation_rejection_has_machine_code__MX12` | red |
| MX-12 | backend: `tests/units/test_forge_discrepancy_law.py::test_definitive_forge_write_refusal_is_not_uncertain__RR01_MX12` | red |
| MX-12 | backend: `tests/units/test_forge_discrepancy_law.py::test_forge_page_limit_is_bounded_before_send__MX12` | red |
| PR-01 | synth-ai: `tests/contract/test_discrepancy_law.py::test_file_page_parses__PR01_MX11` | red |
| PR-01 | synth-ai: `tests/contract/test_discrepancy_law.py::test_file_pagination_cursor_exposed__PR01` | red |
| PR-02 | synth-ai: `tests/contract/test_discrepancy_law.py::test_explicit_empty_resource_inventories_round_trip__PR02` | red |
| PR-02 | testing: `synth_cloud/integration/forge/test_forge_stub_worker_journey_law.py::test_forge_stub_worker_journey__RW01_PR02_RR05` | blocked by deployed run-status projection |
| PR-03 | backend: `tests/units/test_forge_discrepancy_law.py::test_project_upload_exposes_launch_inventory_identity__PR03` | red |
| PR-04 | backend: `tests/units/test_forge_discrepancy_law.py::test_metadata_patch_preserves_stored_file_identity__PR04` | red |
| PR-06 | backend: `tests/units/test_forge_discrepancy_law.py::test_native_delivery_limit_has_specific_code__PR06_PR15` | red |
| PR-07 | backend: `tests/units/test_forge_discrepancy_law.py::test_resource_readiness_keeps_selected_and_missing_files__PR07` | red |
| PR-08 | synth-ai: `tests/contract/test_discrepancy_law.py::test_preflight_keeps_typed_readiness__PR08` | red |
| PR-09 | backend: `tests/units/test_forge_discrepancy_law.py::test_trigger_required_fields_match_runtime__PR09_RW13` | red |
| PR-10 | backend: `tests/units/test_forge_discrepancy_law.py::test_preflight_cannot_clear_trigger_provenance_refusal__PR10_RW06` | red |
| PR-12 | synth-ai: `tests/mcp/test_launch_discrepancy_law.py::test_launch_schema_accepts_provenance__RW01_PR12` | red |
| PR-13 | backend: `tests/units/test_forge_discrepancy_law.py::test_verifier_only_project_accepts_explicit_no_model_files__PR13` | red |
| PR-15 | backend: `tests/units/test_forge_discrepancy_law.py::test_native_delivery_limit_has_specific_code__PR06_PR15` | red |
| PR-18 | synth-ai: `tests/contract/test_discrepancy_law.py::test_request_body_satisfies_backend__PR18_MX06` | red |
| RR-01 | Forge: `tests/test_discrepancy_law.py::test_producer_bound_result_is_specific_refusal__RR01` | red |
| RR-01 | backend: `tests/units/test_forge_discrepancy_law.py::test_definitive_forge_write_refusal_is_not_uncertain__RR01_MX12` | red |
| RR-02 | synth-ai: `tests/contract/test_discrepancy_law.py::test_comparison_retains_integrity_status_and_findings__RR02` | red |
| RR-03 | synth-ai: `tests/contract/test_discrepancy_law.py::test_research_evidence_has_typed_fields__RR03` | red |
| RR-04 | synth-ai: `tests/contract/test_discrepancy_law.py::test_native_result_identity_is_typed__RR04` | red |
| RR-05 | backend: `tests/units/test_forge_discrepancy_law.py::test_agent_result_accepts_null_outcome__RR05` | red |
| RR-05 | synth-ai: `tests/contract/test_discrepancy_law.py::test_execution_operations_client_available__RR05` | red |
| RR-05 | testing: `synth_cloud/integration/forge/test_forge_stub_worker_journey_law.py::test_forge_stub_worker_journey__RW01_PR02_RR05` | blocked by deployed run-status projection |
| RR-06 | synth-ai: `tests/contract/test_discrepancy_law.py::test_citations_client_available__RR06` | red |
| RR-06 | synth-ai: `tests/contract/test_discrepancy_law.py::test_scientific_read_has_bounded_response_contract__RR06_RR07` | red |
| RR-07 | synth-ai: `tests/contract/test_discrepancy_law.py::test_scientific_read_has_bounded_response_contract__RR06_RR07` | red |
| RR-08 | backend: `tests/units/test_forge_discrepancy_law.py::test_integrity_damage_is_service_error__RR08` | red |
| RR-09 | backend: `tests/units/test_forge_discrepancy_law.py::test_late_retry_does_not_replay_expired_admission__RR09` | red |
| RR-11 | backend: `tests/units/test_forge_discrepancy_law.py::test_bundle_truncation_is_explicit__RR11` | red |
| RR-12 | backend: `tests/units/test_forge_discrepancy_law.py::test_backend_environment_sdk_satisfies_declared_pin__RR12` | red |
| RR-12 | synth-ai: `tests/mcp/test_serialization_discrepancy_law.py::test_public_call_tool_returns_json__MX10_RR12` | red |
| RR-14 | synth-ai: `tests/contract/test_discrepancy_law.py::test_transport_failure_keeps_original_intent_and_cause__RR14` | red |
| RR-14 | synth-ai: `tests/contract/test_discrepancy_law.py::test_uncertain_write_preserves_operation_identity__RR14` | red |
| RW-01 | synth-ai: `tests/mcp/test_launch_discrepancy_law.py::test_launch_schema_accepts_provenance__RW01_PR12` | red |
| RW-01 | testing: `synth_cloud/integration/forge/test_forge_stub_worker_journey_law.py::test_forge_stub_worker_journey__RW01_PR02_RR05` | blocked by deployed run-status projection |
| RW-02 | synth-ai: `tests/contract/test_discrepancy_law.py::test_run_control_refusal_is_typed__RW02` | red |
| RW-03 | backend: `tests/integration/test_discrepancy_postgres_law.py::test_branch_refuses_persisted_forge_writer__RW03` | red |
| RW-03 | backend: `tests/units/test_forge_discrepancy_law.py::test_branch_checks_scientific_writer_before_insert__RW03` | red |
| RW-05 | backend: `tests/units/test_forge_discrepancy_law.py::test_hold_requirement_visible_before_trigger__RW05` | red |
| RW-05 | backend: `tests/units/test_forge_discrepancy_law.py::test_preflight_detects_orchestra_budget_refusal__RW05_RW06` | red |
| RW-06 | backend: `tests/integration/test_discrepancy_postgres_law.py::test_preflight_refuses_persisted_forge_writer__RW06` | red |
| RW-06 | backend: `tests/units/test_forge_discrepancy_law.py::test_preflight_cannot_clear_trigger_provenance_refusal__PR10_RW06` | red |
| RW-06 | backend: `tests/units/test_forge_discrepancy_law.py::test_preflight_detects_orchestra_budget_refusal__RW05_RW06` | red |
| RW-07 | backend: `tests/units/test_forge_discrepancy_law.py::test_refusal_response_declared__RW07` | red |
| RW-08 | synth-ai: `tests/contract/test_discrepancy_law.py::test_authority_errors_are_first_class__RW08` | red |
| RW-09 | backend: `tests/units/test_forge_discrepancy_law.py::test_intern_write_uses_execution_provenance__RW09` | red |
| RW-13 | backend: `tests/units/test_forge_discrepancy_law.py::test_trigger_required_fields_match_runtime__PR09_RW13` | red |
| RW-17 | synth-ai: `tests/contract/test_discrepancy_law.py::test_sdk_public_state_vocabulary_matches_backend__RW17` | red |

## Tier 3 and explicit exceptions

- MX-03 is green on current backend dev: both task operation IDs now belong to PUBLIC_OPERATION_IDS. It lives in the contracts suite.
- MX-04/MX-05 use semantic schema/snapshot equality; MX-09 resolves raw SDK routes beyond the bounded registry. RW-17 compares closed public state vocabularies. RR-03/RR-04/RR-06/RR-07 add typed evidence and registry checks.
- The 49 unused registry entries are retained in `unused_registry_allowlist.json` as an explicit audit inventory. Their absence of a caller is not evidence that a route is broken; do not fake execution evidence.
- Stale prose findings RW-04/RW-10/RW-11/RW-15/RW-16, PR-11 and RR-15 remain fix-ticket documentation work, as permitted by the handoff’s optional stale-doc row. Contract snapshots cover machine-readable drift; no brittle prose tests were invented.
- Lower-severity detail-report findings outside the consolidated ranked list are not all independent runtime laws (PR-05/PR-14/PR-20–25, RR-10/13, RW-12/14). These are outside the consolidated acceptance list. Their contract-bearing surface is included in the full snapshot and general route test; no narrower runtime result is claimed.

## Slot evidence and cleanup

- Freshly installed SDK 0.22.2; slot identity, stored/upload file resource preflight, and native SDK/MCP record readback pass. The readback was moved from law to contracts.
- The worker law now consumes actual retained source/image deployment pins and verifies each running image; no dummy pins, unconditional xfail, or paid worker config.
- Two trigger calls returned 200 after every installed native model endpoint was verified as model-stub. Run-status projection returned 503 before result/resource acceptance could be reached. A second run with bounded startup polling also failed. This is blocked acceptance, not proof of the final trial/result effect.
- Cleanup sent typed stop requests. PostgreSQL confirms finished_at for runs `86abc0f9-4919-45e1-85e9-31e5629dd1cb` and `95fc0664-50d3-4e53-811c-7fb160eaaf79`. Slot5 was released; the final controller projection is unclaimed/idle. Disposable PG16 was stopped and removed.
- Completing the worker acceptance requires the deployed run-status projection to serve the newly accepted native run. Fixes were deliberately excluded from this lane; no unattended work remains running.

## Commands and receipts

- synth-ai: `PYTHONPATH="$SDK_WORKTREE" backend/.venv/bin/python -m pytest "$SDK_WORKTREE/tests/contract" "$SDK_WORKTREE/tests/mcp" -q --tb=short`.
- backend: put its worktree first on PYTHONPATH; run `tests/units/test_forge_discrepancy_law.py` plus `test_discrepancy_contracts.py`.
- Postgres: supply an explicit disposable loopback PG16 URL named `discrepancy_*` via DISCREPANCY_TEST_DATABASE_URL; select `tests/integration/test_discrepancy_postgres_law.py`.
- Forge: put packages, services/forge, app on PYTHONPATH; run `tests/test_discrepancy_law.py`.
- testing: `./testctl validate`; owned slot via `slotctl eval-exec slot5 --target local-dockerized --workspace foundation-finish-20261005 --memory-bytes 536870912 -- <installed-python> -m pytest synth_cloud/integration/forge --synth-target slot5`.
- Complete node/status receipts are committed in `test_results.json`; raw junit/logs, deployment-pin witness, slot evidence and cleanup receipts are retained locally at `artifacts/discrepancy-tests-20261006/`.
- Ruff E/F/I checks pass for all new offline tests/helpers; formatter applied. First-class error laws preserve stable codes, causal chains, retryability and uncertain intent identity. Known-red assertions are not weakened.

Branches: `codex/discrepancy-tests-20261006` in synth-ai, backend, Forge and testing; pushed, unmerged.

## Additional findings EX-01 through EX-07

The extension adds 35 known-red assertions and 12 passing controls beyond the
original 104/205 results. See [additional findings](ADDITIONAL_FINDINGS_20261006.md)
and `extension_test_results.json` for precise selectors, source locations and
severity. Seven confirmed issues cover retry authority, idempotency header casing,
finite configuration, redirect/empty/non-finite JSON boundaries, and scientific
boolean-to-number coercion. No product fixes, CI, slots or paid effects.

## Additional findings EX-08 through EX-20

The next pass adds 52 known-red assertions and 18 passing controls. See
[streaming/MCP/pagination findings](STREAMING_PAGINATION_FINDINGS_20261006.md)
and `streaming_boundary_test_results.json`. These are six SSE framing/metadata
issues and seven closed-argument/idempotency/page-boundary issues. All are
reproduced offline; both SDK transports and both public MCP entry points are
covered where applicable. No product changes, CI, slots or paid effects.
