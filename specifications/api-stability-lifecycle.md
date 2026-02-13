# API Stability Lifecycle

Every public SDK module and API endpoint goes through three stability stages: **Alpha**, **Beta**, and **Stable**. The stage determines access, compatibility guarantees, and how the feature appears in documentation.

## Stages

### Alpha

- **Access**: Feature-gated. Enabled per-account on request.
- **Compatibility**: No guarantees. Breaking changes can ship without notice.
- **Docs tag**: `ALPHA` (yellow badge in sidebar).
- **When to use**: Internal testing, design partners, early validation.

### Beta

- **Access**: Generally available. Any authenticated user can call it.
- **Compatibility**: Best-effort backwards compatibility. Breaking changes are announced at least one release in advance, but may still occur.
- **Docs tag**: `BETA` (blue badge in sidebar).
- **When to use**: Feature is functional and usable in production, but the API surface may still change based on feedback.

### Stable

- **Access**: Generally available.
- **Compatibility**: Semantic versioning. Breaking changes require a major version bump.
- **Docs tag**: None (no badge). Stable is the default; absence of a tag means stable.
- **When to use**: API surface is locked. Changes are additive only.

## Marking a module

Add a `**Status:**` line to the module docstring, immediately after the summary line:

```python
"""Environment Pools — run coding agents in managed sandboxes.

**Status:** Alpha

This module provides the ``EnvironmentPoolsClient`` ...
"""
```

Valid values: `Alpha`, `Beta`. Omit the line entirely for Stable modules.

The docs generator (`generate_sdk_reference.py`) reads this marker and:
1. Adds a `tag:` field to the MDX frontmatter (renders as a sidebar badge).
2. Inserts a `<Badge>` component at the top of the page body.

## Runtime enforcement (feature flags)

Alpha features are gated at runtime using the `feature_flags` table in PlanetScale.
Only orgs with an explicit enabled row can call Alpha endpoints; all other callers
receive a `403` with error code `feature_not_available`.

### Database schema

```
feature_flags (
    org_id      uuid NOT NULL,
    user_id     uuid,              -- NULL means org-wide
    feature     varchar(100),      -- e.g. "environment_pools", "managed_pools"
    enabled     boolean DEFAULT true,
    expires_at  timestamptz,       -- NULL = never expires
    UNIQUE (org_id, user_id, feature)
)
```

Migration: `backend/app/data/db/migrations/create_feature_flags_table.py`

### Enforcement layers

| Layer | File | Mechanism |
|-------|------|-----------|
| Python API gateway | `backend/app/core/feature_gates.py` | `FeatureGate` class used as a router-level FastAPI dependency. Queries `feature_flags` with a 30 s L1 cache. |
| Rust backend (defense-in-depth) | `rust_backend/src/managed_env_pool/api/routes.rs` | `require_feature_flag()` called after auth in each handler. Uses `DbClient::check_feature_flag()`. |

### Error response

```json
{
  "error": {
    "code": "feature_not_available",
    "feature": "environment_pools",
    "message": "This feature is in Alpha. Contact support to request access.",
    "upgrade_url": "https://usesynth.ai/contact"
  }
}
```

The SDK's `_raise_for_status_with_plan_check()` converts this 403 into a
`PlanGatingError` so callers get a structured Python exception.

### Granting access

```sql
INSERT INTO feature_flags (org_id, feature, enabled, reason, created_by)
VALUES ('<org-uuid>', 'environment_pools', true, 'alpha tester', 'manual');
```

### Currently gated features

| Feature key | Routes | Stage |
|------------|--------|-------|
| `environment_pools` | `/v1/environment-pools/*`, `/v1/rollouts/*`, `/v1/pools/*` | Alpha |
| `managed_pools` | `/v1/managed-pools/*` | Alpha |

## Promotion checklist

### Alpha to Beta

- [ ] Feature gate removed (or switched to open-by-default).
- [ ] Public documentation written (not just auto-generated reference).
- [ ] At least one cookbook or quickstart exists.
- [ ] Error messages are user-facing quality (no raw tracebacks or internal jargon).
- [ ] Rate limiting and input validation in place.

### Beta to Stable

- [ ] API surface reviewed and locked (no planned breaking changes).
- [ ] Migration guide written for any breaking changes since Beta launch.
- [ ] Integration tests cover the public contract.
- [ ] At least 2 weeks in Beta with external usage and no critical bugs.
- [ ] `**Status:** Beta` line removed from docstring (absence = Stable).

## Generator implementation

The pipeline lives in `docs/scripts/generate_sdk_reference.py` in the `_extract_and_add_status_tags()` function. It supports:

| Docstring marker   | Sidebar tag | Badge color | Badge icon              |
|--------------------|-------------|-------------|-------------------------|
| `**Status:** Alpha`| ALPHA       | yellow      | triangle-exclamation    |
| `**Status:** Beta` | BETA        | blue        | circle-check            |
| *(none)*           | *(none)*    | *(none)*    | *(none — Stable)*       |

The `Experimental` value is deprecated. Use `Alpha` instead.
