#!/usr/bin/env bash
set -euo pipefail

cd /Users/joshpurtell/Documents/GitHub/synth-ai
source /Users/joshpurtell/Documents/GitHub/monorepo/.venv/bin/activate
set -a
source /Users/joshpurtell/Documents/GitHub/synth-ai/.env
set +a

python -m synth_ai.cli eval \
  --config /Users/joshpurtell/Documents/GitHub/synth-ai/tests/local_api/banking77_eval.toml \
  --url http://localhost:8103 \
  --output-txt /Users/joshpurtell/Documents/GitHub/synth-ai/tests/local_api/banking77_eval_output.txt \
  --output-json /Users/joshpurtell/Documents/GitHub/synth-ai/tests/local_api/banking77_eval_output.json
