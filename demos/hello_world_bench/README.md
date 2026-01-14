## hello_world_bench

Minimal sanity-check harness to answer one question:

**Can OpenCode successfully perform a single file write/edit in a sandbox when routed through the Synth interceptor?**

It creates a temp sandbox with a `README.md` containing a TODO, asks OpenCode to write `Hello, world!`,
and then prints the resulting `README.md`.

### Run

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
LOG_LEVEL=INFO uv run python demos/hello_world_bench/run_demo.py --local --model gpt-5-nano --timeout 120
```

