# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `container-contract-v1.yaml` (container rollout/health/info contract)
- Version: OpenAPI 3.1

## Canonical Product APIs

- Optimization systems (`/v1/systems/*`)
- Offline optimization / GEPA / MIPRO (`/v1/offline/jobs/*`)
- Online optimization / continual learning (`/v1/online/sessions/*`)
- Hosted containers (`/api/v1/containers/*`)
- Container pools and pool-scoped rollouts (`/v1/pools/*`)
- Container runtime endpoints (`/health`, `/rollout`, `/info`, `/validate-candidate`)

## Research Track (2026-02-28)

The following surfaces have been moved to `research/feb28_2026/` for ownership and documentation. Routes remain live for compatibility but are not primary user-facing APIs. See `research/feb28_2026/` for full contracts.

- Inference APIs (`/api/inference/v1/chat/completions`, `/api/inference/jobs*`) -- marked `x-research-track`
- Graph APIs (`/api/graphs/completions`, `/graph-evolve/graphs`) -- marked `x-research-track`
- Global rollouts (`/v1/rollouts/*`) -- marked `x-research-track`, prefer pool-scoped rollouts

## Internal (de-exposed from user docs)

- SynthTunnel leases (`/api/v1/synthtunnel/leases*`) -- marked `x-internal`. Use SDK tunnel abstractions (`SynthClient().tunnels.*`) instead of calling directly.

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
