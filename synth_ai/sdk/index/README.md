# Synth Index SDK

Unreleased typed search and contents contracts. Backend
`packages/contributions/{contracts,search}.py` owns the vocabulary and validators;
the SDK mirror must remain schema- and behavior-compatible. Do not import backend
modules from the published package. Cross-repo parity checks belong in `testing`.

See `docs/drafts/synth-index-api-design-2026-09-12.md` and the frozen execution plan.
Fast search is the MVP; deep execution handles are not implemented here. No
published SDK capability or live backend route is claimed by these models.

The unreleased `SynthClient().index.contributions` and async equivalent expose
`create(idempotency_key=...)`, `prepare_upload(draft, spec)`, `finalize(draft, prepared)`, and
`submit(reference, spec)`. Request/result types are exported from
`synth_ai.sdk.index`, including `ContributionPackage`, `ContributionUploadSpec`,
and `ContributionSubmitSpec`. Creation returns a private server-issued draft;
upload preparation returns signed transfer instructions but sends no file bytes.
Submission requires an already finalized Artifact publication and does not grant
approval, publish content, or award credits. Preserve the publication ID for
submission retries after uncertain failures.

For small artifacts, async `synth_ai.sdk.index.transfer.upload_bytes(prepared,
content)` accepts an explicit mapping from declared logical paths to immutable
bytes, bounded to 64 MiB total asset content. It verifies all sizes/digests and
targets before creating a credential-free storage HTTP client. It does not read
files, follow redirects, finalize, submit, or retry. On partial failure, prepare
again with the same publication ID; already-present objects may be omitted from
transfer targets. Finalization verifies the complete Artifact publication.

The managed PostgreSQL/MinIO integration test covers SDK prepare → byte transfer
→ SDK finalize → SDK submit, with in-process HTTP and injected principals. It is
not evidence of deployed authentication or a published SDK release. Streaming
datasets/files, review/publication client methods, and Workshop/MCP upload
orchestration remain missing.
