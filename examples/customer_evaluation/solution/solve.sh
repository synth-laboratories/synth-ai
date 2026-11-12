#!/bin/sh
set -eu
python - <<'PYTHON'
import json
from pathlib import Path
values = json.loads(Path('/app/input.json').read_text())['values']
Path('/app/result.json').write_text(json.dumps({'values': sorted(set(values))}))
PYTHON
