# Synth OpenAPI Contracts

- Spec files:
  - `synth-api-v1.yaml` (canonical SDK HTTP transport)
  - `container-contract-v1.yaml` (container rollout/health/info contract)
- Version: OpenAPI 3.1

## Canonical Product APIs

- Hosted containers (`/v1/containers/*`)
- Managed tunnels (`/v1/tunnels/*`)
- Container pools and pool-scoped rollouts (`/v1/pools/*`)
- Global rollouts (`/v1/rollouts/*`)
- Container runtime endpoints:
  - Core: `/health`, `/task_info`, `/rollouts` (with `/rollout` compatibility alias)
  - Long-horizon/multi-agent: `/rollouts/{rollout_id}/checkpoints*`, `/rollouts/{rollout_id}/resume`, `/rollouts/{rollout_id}/actors*`
  - Validation: `/candidates/validate` (with `/validate-candidate` compatibility alias)

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
