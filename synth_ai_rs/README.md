# synth-ai (Rust SDK)

Canonical Rust SDK for [Synth](https://usesynth.ai), aligned to the same namespace model as the Python SDK.

## Installation

```bash
cargo add synth-ai
```

Or in `Cargo.toml`:

```toml
[dependencies]
synth-ai = "0.9.1"
```

## Quick Start

```rust
use synth_ai::SynthClient;

#[tokio::main]
async fn main() -> Result<(), synth_ai::Error> {
    let client = SynthClient::from_env()?;

    let systems = client.optimization().systems().list(None).await?;
    println!("systems={}", systems.items.len().max(systems.data.len()));

    Ok(())
}
```

## Canonical Namespaces

- `client.optimization()`
- `client.inference()`
- `client.graphs()`
- `client.verifiers()`
- `client.pools()`
- `client.containers()`
- `client.tunnels()`

## Contract Source of Truth

- OpenAPI: `openapi/synth-api-v1.yaml`
- Container contract: `openapi/container-contract-v1.yaml`

## Links

- [Documentation](https://docs.usesynth.ai)
- [GitHub](https://github.com/synth-laboratories/synth-ai)
- [Discord](https://discord.gg/VKxZqUhZ)
