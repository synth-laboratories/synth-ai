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

The October 9 OP candidate snapshot is copied from backend source
`fd096cbdceb5f6139ea91ceecaad03e0709c83d7` with
`testing/scripts/sync_smr_schemas.py::sync_smr_openapi_snapshot`. Its SHA256 is
`1b011d0bc3eb992b6719afc4829096963d131c6553c53745fda0ae30784c30ed`.
The curated Research registry separately matches all 287 operations in
`openapi/research-v1.json`, including backend registry additions.
