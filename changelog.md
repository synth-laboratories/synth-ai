# Release v[0;34m[INFO][0m Current version: 0.2.1.dev1
[0;34m[INFO][0m Bumping version type: dev
[0;34m[INFO][0m New version: 0.2.1.dev2
[0;32m[SUCCESS][0m Version bumped to 0.2.1.dev2
0.2.1.dev2 - 2025-07-29

## What's New

- **Environment Registration API**: Third-party packages can now register custom environments dynamically via REST API, CLI, or entry points (e.g., `curl -X POST localhost:8901/registry/environments -d '{"name":"MyEnv-v1","module_path":"my_env","class_name":"MyEnv"}'`)
- **Turso Database Integration**: Added Turso/sqld daemon support with local-first database replication via `uvx synth-ai serve` (replicas sync every 2 seconds by default)
- **Environment Service Daemon**: The `uvx synth-ai serve` command now starts both the Turso database daemon (port 8080) and environment service API (port 8901) for complete local development setup

## Breaking Changes

- [List any breaking changes]

## Bug Fixes

- [List bug fixes]

## Documentation

- [List documentation updates]

---

# Release v[0;34m[INFO][0m Current version: 0.2.1.dev0
[0;34m[INFO][0m Bumping version type: dev
[0;34m[INFO][0m New version: 0.2.1.dev1
[0;32m[SUCCESS][0m Version bumped to 0.2.1.dev1
0.2.1.dev1 - 2025-07-29

## What's New

- [Add your changes here]

## Breaking Changes

- [List any breaking changes]

## Bug Fixes

- [List bug fixes]

## Documentation

- [List documentation updates]

---
