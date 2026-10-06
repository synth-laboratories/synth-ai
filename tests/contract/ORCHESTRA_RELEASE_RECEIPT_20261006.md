# Orchestra discrepancy release — 2026-10-06 18:19 UTC

SYN-4001–4004 are released to the supported staging/prerelease lane. This supersedes the deployment/publication blockers in the earlier source-fix receipt. Production was not promoted.

- Backend dev PR1842 merged at b0b88b1; staging PR1843 source `fa0be484930587267d1381fd859b2260e83fa97f` (tree-equal to stage-qualified035ec54f4).
- [Backend run37508979719](https://github.com/synth-laboratories/backend/actions/runs/37508979719) succeeded. One digest `ghcr.io/synth-laboratories/backend@sha256:b33f56d66ff231e6f2422c3aff4e5244dba191bb31740f655edb341619315405`, pinned runner00e0b304, unchanged schema `20261116_settlement_usage_unknown_aborted`. DEEP first, then API/worker/SMR. Four provider IDs verified SUCCESS through the pinned read-only release runner. Both version routes, health and migration-current readback pass.
- Deployments: DEEP912acd7c-32e6-47ce-9a35-00b594f86ebc; API f28a6e34-c0be-468d-92d3-378db1323e89; worker7994e49a-0a73-4721-ab51-c1cf812002b1; SMR f6995333-e566-4dcf-95fb-9f0262609708.
- SDK423 merged into422;422 merged to dev at `4e77cfb5b529a87637684e150fe8ab4806832056` only after backend success. Source tree equals qualified86e3fd88.
- [Published SDK0.22.2.dev682](https://pypi.org/project/synth-ai/0.22.2.dev682/) / [release](https://github.com/synth-laboratories/synth-ai/releases/tag/v0.22.2.dev682). [Publication run37509915460](https://github.com/synth-laboratories/synth-ai/actions/runs/37509915460) succeeded. Public wheel SHA256 `6e754e0f4ddf3120d3c7c544fec31839d06b993b41ee0dfa3e01e31f31edeffc`.
- Fresh venv installed the exact public package with PYTHONPATH unset. All317 committed Python files byte-equal;18 real MockTransport control paths and4 MCP launch forwarding paths pass. Factory remains an optional module, outside run/Swarm types and default MCP discovery; independent Intern planner efforts remain available.

Source proofs:551 backend regression/launch/HTTP checks +198 stage compatibility;69 SDK contract/launch +31 run-read/control +13 optional Factory traces +11 MCP discovery. Counts overlap where suites intentionally requalify the same code. Original failing-to-passing laws remain retained; no assertions were weakened. Live full OpenAPI and pinned release-contract checks pass. Static legacy snapshot extraction misses dynamic native/Intern routes; the real producer app was the authority.

Raw receipts in `/Users/joshuapurtell/GitHub/artifacts/orchestra-discrepancy-unblock-20261006/`: completion.json, manifest and all original checkpoints, live provider status, both version/health readbacks, public PyPI metadata/hash, public install logs, public_smoke.py and public-installed-receipt.json, and all before/after qualification logs. The first index install saw a stale index view; explicit public-index refresh/reinstall passed. The direct public wheel was independently SHA-verified.

Reproduce installed acceptance (no provider access):

```sh
env -u PYTHONPATH artifacts/orchestra-discrepancy-unblock-20261006/public-venv/bin/python artifacts/orchestra-discrepancy-unblock-20261006/public_smoke.py
```

No slot2 mutation, no new CI wiring, no production/Monitor deployment, no provider/model call. Model spend$0. Existing release/publication infrastructure allowance$5; Actions timing reports0 billable milliseconds, invoice unknown until reconciliation. Both owned workflows are completed; nothing runs unattended.
