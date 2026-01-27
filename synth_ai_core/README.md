# synth-ai-core

Rust core library for [Synth AI](https://usesynth.ai) - serverless post-training APIs.

This crate provides the low-level building blocks for Synth AI SDKs:

- **API Client** - HTTP client for Synth AI backend
- **Streaming** - Real-time job progress and event streaming
- **Tracing** - Session tracing with libsql storage
- **Tunnels** - Cloudflare tunnel management for local APIs
- **Orchestration** - Job lifecycle management

## Usage

Most users should use the high-level [`synth-ai`](https://crates.io/crates/synth-ai) crate instead.

```rust
use synth_ai_core::SynthClient;

let client = SynthClient::from_env()?;
```

## License

MIT
