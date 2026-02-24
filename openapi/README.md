# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport + OpenAI-compatible chat completions)
  - `container-contract-v1.yaml` (container rollout/health/info contract)
- Version: OpenAPI 3.1

## Includes

- Optimization APIs (`/v1/policy-optimization/systems`, `/v1/offline/jobs`, `/v1/online/sessions`)
- Inference APIs (`/api/inference/v1/chat/completions`, inference jobs)
- Graph APIs (`/api/graphs/completions`, `/graph-evolve/graphs`)
- Hosted containers (`/api/v1/containers`)
- Container pools and rollouts (`/v1/pools/*`, `/v1/rollouts/*`)
- SynthTunnel leases (`/api/v1/synthtunnel/leases`)
- Container runtime endpoints (`/health`, `/rollout`, `/info`, `/validate-candidate`)

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
