Prod-only API integration tests

Setup:
- export SYNTH_API_KEY_PROD (or SYNTH_API_KEY) in your shell

Run:
- uv run pytest -q tests/api

Notes:
- Tests hit https://agent-learning.onrender.com/api/v1
- Tests allow 401/429 to avoid flakiness without valid credentials or under rate limits.

