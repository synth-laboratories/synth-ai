# Synth OpenAPI Contract

- Spec file: `synth-api-v1.yaml`
- Version: OpenAPI 3.1
- Scope: Canonical SDK HTTP transport and OpenAI-compatible chat completions.

## Includes

- Optimization APIs (`/v1/policy-optimization/systems`, `/v1/offline/jobs`, `/v1/online/sessions`)
- Inference APIs (`/api/inference/v1/chat/completions`, inference jobs)
- Graph APIs (`/api/graphs/completions`, `/graph-evolve/graphs`)
- Hosted containers (`/api/v1/containers`)
- Container pools and rollouts (`/v1/pools/*`, `/v1/rollouts/*`)
- SynthTunnel leases (`/api/v1/synthtunnel/leases`)

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
