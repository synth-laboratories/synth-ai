# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `research-v1.json` (Research / SMR contract)
- Version: OpenAPI 3.1

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
