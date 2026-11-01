# SYN-5351: Index live fingerprint drift. Evidence packet (2026-11-01)

Status: **the schema change was intentional; the stale shipped Index identity is a release-process omission.**
This packet does not change any digest. A contract owner must approve the atomic update below.

## Symptom

At SDK `41bdeee04e41501b8d60cfa3065470b4de36bca4`, with pydantic 2.12.5:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" \
  /Users/joshuapurtell/GitHub/workspaces/mnemosyne-sdk-wheel-pin-20261031/.venv/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_fingerprints_current.py --basetemp=$PWD/.test-tmp/fp
# 1 failed, 2 passed
# test_pinned_live_exports_match_the_shipped_table:
#   shipped index.v1 75563ddda18894c5cf10c18d3ce20a828a779772288c7f08b2007db7050b0f93
#   live    index.v1 728161599799b93eb2fc6e0f1b58e2ed5b6d9370afdcea062c70af4ce28f4ef3
```

`core.v1` and `swarms.v1` live exports match the shipped table.

## Bisection: live exporter digest per commit

Each commit's `synth_ai/` was exported with `git archive` into a scratch directory and the same interpreter.

| SDK commit | live index.v1 | shipped index.v1 |
|---|---|---|
| `77699d58` (#595, last commit before the change) | `75563ddda188…` | `75563ddda188…` |
| `8e24187b` (#596) | `728161599799…` | `75563ddda188…` |
| `41bdeee0` (cohort head) | `728161599799…` | `75563ddda188…` |

`75563ddd…` was recorded in `e5caec17` (#592, SYN-5220), with `contracts/index-v1.json`. The test pin in
`tests/test_error_contract_fingerprint.py` was aligned in `77699d58` (#595).

## What changed in the export

The live export was compared with retained `contracts/index-v1.json` model by model. Operations are identical.
All 60 models but one are byte-identical. Only **`AccountEntitlements`** changed, through the shared
`synth_ai/types/account.py`:

| JSON path under `models/AccountEntitlements` | before | after |
|---|---|---|
| `$defs/AccountEntitlement/properties/product/enum` | index, swarms, monitor | + `inference` |
| `$defs/AccountEntitlement/properties/subproduct/enum` | 11 values | + `inference.glm-5.3-flash`, `inference.deepseek-v4.1-flash-1017` |
| `$defs/AccountPolicyLimit/properties/product/enum` | index, swarms | + `inference` |
| `$defs/AccountPolicyLimit/properties/dimension/enum` | requests, concurrent, spend_cents | + `tokens` |
| `$defs/AccountPolicyLimit/properties/window/enum` | token_bucket, concurrent, day, month | + `minute` |
| `properties/subproducts/maxItems` | 11 | 13 |

The `AccountEntitlements.complete` validator also accepts subproduct sets without the two inference
subproducts. Validator behavior is not included in the JSON schema, so it has no fingerprint effect.

## Cause and intent

- Commit `8e24187be63478da753f8fd7b690b3d403be6fae`, PR #596 "Support inference account observations in SDK",
  merged into `dev` at 2026-10-31T22:10:38Z. It is an intentional product change: inference account entitlements,
  usage filters and events. It added `tests/test_account_inference.py` and refreshed `contracts/account-mcp-v1.json`.
- The PR body says: "The current dev Index fingerprint and package version remain intact." The authors deliberately
  kept the Index identity. They overlooked that `AccountEntitlements` is exported in `index.v1`
  (`synth_ai/contracts/schema.py` `MODELS`). So the Index wire schema changed while its shipped identity did not.
- Independent confirmation: sibling commit `81e730fa` ("fix(account): expose managed inference policy products and
  token limits", branch `codex/shoal-serverless-launch-20261030`, **not** an ancestor of 41bdeee0) made the same
  `account.py` change. It moved `index.v1` to exactly `728161599799…` together with a version bump and `uv.lock`.
  The new digest is the expected identity of this schema change, not nondeterminism.

Classification: **intentional schema change, with an incomplete contract release (stale Index identity)**.
The drift is not accidental source drift, so reverting `account.py` is the wrong fix. A revert would drop shipped
inference account support from #596.

## Consumer impact

- SDK negotiation requires exact equality (`synth_ai/core/capabilities.py:110-111`): the backend-advertised
  `index.v1` schema must equal the SDK shipped table. Backend (`mnemosyne-traces-integration-20261031` @ 74d9b838dc) serves
  `packages/customer_api/contracts/fingerprints.json` index.v1 = `75563ddd…`. Its own live export also computes
  `75563ddd…` because `packages/customer_api/types/account.py` still has `max_length=11` and no inference values.
  **The current shipped pair (SDK 75563ddd / backend 75563ddd) negotiates.** An SDK-only rebaseline to `728161…`
  would make every Index call against this backend refuse with `schema_mismatch`.
- Wire-level today: the SDK models are a strict superset of the backend's. Backend responses still validate in the
  SDK. The backend cannot emit inference entitlements: its own `AccountEntitlements` would refuse them.
- Backend Index consumers of the table: `packages/customer_api/contracts/fingerprint.py` (shipped table),
  `app/api/v1/customer_api/negotiation.py` (`X-Synth-Schema-SHA256` pins), `app/api/app.py` and
  `app/api/local_app.py` (warm-up). Tests that pin subproduct cardinality and may need review:
  `tests/units/test_entitlements_admin.py`, `tests/units/test_entitlements_admin_cli.py`,
  `tests/customer_api/test_mcp_tool_visibility.py`.
- Provenance records still claiming `75563ddd…`: `requirements/20261030-index-public-content-refs-provenance.json`,
  `requirements/20261030-index-public-content-refs-1015-provenance.json` (historical; leave them). The cohort's
  `requirements/20261031-mnemosyne-trace-sdk-provenance.json` lists this drift under `remaining_gates`.

## Mnemosyne

The change does not affect Mnemosyne. Mnemosyne operations and `EvaluationEvidenceResponse` are exported under `core.v1`
(`packages/customer_api/contracts/core_schema.py`), and `core.v1` live == shipped in both repos. The Index export
contains no trace or Mnemosyne models. `services/mnemosyne`, `services/smr/context_engine`, `packages/mnemosyne` and
`app/api/v1/*mnemosyne*.py` import nothing from `synth_ai` and do not reference `AccountEntitlements` or `index.v1`.
The two strict receipt xfails (publication/event-batch receipts) are unrelated to this drift.

## Proposed atomic update (for owner approval; not applied)

Two options. They must not be mixed:

**A. Adopt inference entitlements in Index (recommended, matches #596 and 81e730fa intent).** All of these land together:

SDK (`synth-ai`):
1. `synth_ai/contracts/fingerprints.json`: index.v1 → `728161599799b93eb2fc6e0f1b58e2ed5b6d9370afdcea062c70af4ce28f4ef3`
   (via `python -m synth_ai.contracts.fingerprint --write` under pydantic 2.12.5).
2. `contracts/index-v1.json`: regenerate from `synth_ai.contracts.schema.export_contract()`.
3. `tests/test_error_contract_fingerprint.py`: `INDEX_V1_SCHEMA_SHA256` plus a history comment citing #596/SYN-5351.
4. `pyproject.toml` version bump and `uv.lock` (package version line).

Backend:
5. `packages/customer_api/types/account.py`: mirror the six schema deltas above and the validator's extra allowed sets.
6. `packages/customer_api/contracts/fingerprints.json`: index.v1 → `728161…`. Confirm that the backend live export equals it.
7. New wheel `requirements/synth_ai-<new version>-py3-none-any.whl` plus a new provenance JSON. Drop the
   "Index live exporter differs" gate only after a fresh run of `tests/test_fingerprints_current.py` passes 3/3.
8. Repin the same version/sha in `requirements/api.txt`, `requirements/worker.txt`, `requirements/runtime-common.txt`,
   `infra/requirements.txt`, `infra/Dockerfile.python_services`, `pyproject.toml`, `uv.lock`, `BUCK`,
   `scripts/buck2_image_seeds.py`. Then rerun the backend SDK pin/seed checks (33) and the customer_api contract tests.
9. Update the handoff contract table (`docs/operations/mnemosyne-engineer-handoff-20261101.md` §SDK package and contracts).

**B. Keep Index at `75563ddd…`.** Remove `AccountEntitlements` deltas from the Index export. For example, export a
pre-inference `AccountEntitlements` for `index.v1`, or move inference account types to a separate account surface.
This also needs a rebuilt wheel and repins (steps 7-8). It contradicts the stated intent of #596 and 81e730fa.

Deployment order for A: backend and SDK identities change at once, so old SDK clients (75563ddd) will refuse against
a new backend (728161…), because the SDK requires exact equality. This requires a coordinated cohort
release, unless a deliberately designed and reviewed transition window exists. Do not add an ad hoc dual-accept list.
