# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `research-v1.json` (Research / SMR contract)
- Version: OpenAPI 3.1

## Index readiness integration

`index-v1.json` mirrors backend
`contracts/synth_index_openapi.json` at
`2bdf3bd7ceda6fa53a2f50e11fa8ed46919a79cb` (SHA-256
`c866daddce3d1ddba6b978f41f184fb8c87b841d9397958c922c17dafd0904b8`).
Anonymous browsing has no search route; public-scope search uses the
authenticated, funded `/api/v1/index/search` contract. Rebind this vendor file
to the final frozen backend commit before publication.
Customer accounting adds funding and terminal outcome to charges, and price,
funding, terminal outcome, settlement state, reservation, release, and refund
fields to summary rows while retaining consumption and pagination. Customer
charges and summary rows also include signed `adjustment_microcents`; settlement
totals already include those corrections and outstanding refunds. Customer
receipt/summary infrastructure-cost fields are removed. Funding uses
`deep_beta`, not `index_deep_beta`, and outcomes reference
`SearchSettlementOutcome`.

Summary rows also expose `measurement_state` and `unmeasured_search_count`.
Consumption counters are nullable for missing measurements. Settlement-only
receipts carry pending measurement state and empty observation arrays. Operator
physical costs use separate backend diagnostics DTOs, outside customer contracts.

The Index file above is a backend export rather than a hand-edited schema.
Final integration still owns the exact deployed backend/OpenAPI/SDK parity,
customer redaction, and corrected receipt/summary/CSV amount agreement.

## v0.18.2 release contract refresh

The bounded Research registry includes the existing backend meta-thread and
async-handoff routes as well as Visuals, claims, objectives, and run-limit
operations. Its exported method/path/operation IDs must match the registry
exactly; do not relax the contract gate to permit missing entries. Three older
SDK event/acceptance-receipt names remain lookup aliases to canonical backend IDs.
New mutation entries do not assume retry safety unless explicitly established.

The full `schemas/smr_openapi.yaml` is vendored from the v0.10 backend release
candidate (`19e46ac56952e3c745e50dd9f2ed14de1a9c39fe`), not an arbitrary sibling
checkout. For release verification, pass `BACKEND=` to `make test-unit` and set
`BACKEND_OPENAPI_REF` to the reviewed backend commit.

Hosted-container and related infra OpenAPI fragments may live under `old/openapi/`
while those Python clients are archived.

## Canonical Product APIs (Research-first SDK)

The live Python SDK composes Research via `SynthClient().research`. Backend
routes for tunnels/pools/containers may still appear in `synth-api-v1.yaml`;
the corresponding clients are archived under `old/sdk/` until restored.

## Internal (de-exposed from user docs)

- SynthTunnel leases (`/api/v1/synthtunnel/leases*`) -- marked `x-internal`.

## Codegen examples

```bash
# TypeScript
npx @openapitools/openapi-generator-cli generate \
  -i openapi/synth-api-v1.yaml \
  -g typescript-fetch \
  -o generated/synth-ts

# Python
openapi-generator-cli generate \
  -i openapi/synth-api-v1.yaml \
  -g python \
  -o generated/synth-py
```
