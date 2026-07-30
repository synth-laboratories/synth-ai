# Vendored backend contract

Snapshots of the backend's Research contract, kept here so CI can detect drift
between this SDK and the backend without a backend checkout.

| File | Source of truth |
|------|-----------------|
| `smr_openapi.yaml` | the backend repo's `smr_openapi.yaml` |
| `public_models.json` | the backend repo's `config/smr_public_models.json` |

These are **not** shipped. No module in `synth_ai/` reads them, and until 0.18.0
they sat in `synth_ai/sdk/research/schemas/` and added ~2.1MB to every wheel —
including, in a Research-only package, the full `/v1/tunnels` and `/v1/pools`
route definitions. They are build- and test-time inputs only.

Refreshed and checked by the sibling `testing/` repo:
`scripts/sync_smr_schemas.py` writes them; `scripts/validate_synth_ai_contract.py`
compares them against the backend. Do not hand-edit.
