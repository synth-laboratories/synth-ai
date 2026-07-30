# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `research-v1.json` (Research / SMR contract)
- Version: OpenAPI 3.1

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
