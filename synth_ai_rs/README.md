# synth-ai

Rust SDK for [Synth AI](https://usesynth.ai) - serverless post-training APIs.

## Installation

```bash
cargo add synth-ai
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
synth-ai = "0.1"
```

## Status

This crate is under active development. It now exposes the shared Rust core for URLs and
event polling. For the full-featured SDK, see the [Python package](https://pypi.org/project/synth-ai/).

## Quick Example

```rust
use synth_ai::{Client, CoreConfig, EventKind};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let config = CoreConfig::default();
let client = Client::new(config);
let resp = client
    .poll_events(EventKind::PromptLearning, "pl_123", Some(0), Some(200))
    .await?;
println!("events: {}", resp.events.len());
# Ok(())
# }
```

## Links

- [Documentation](https://docs.usesynth.ai)
- [GitHub](https://github.com/synth-laboratories/synth-ai)
- [Discord](https://discord.gg/VKxZqUhZ)
