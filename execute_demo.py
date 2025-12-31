#!/usr/bin/env python3
"""Execute the demo script."""
import subprocess
import os
import sys
from pathlib import Path

# Change to the synth-ai directory
os.chdir('/Users/joshpurtell/Documents/GitHub/synth-ai')

# Set environment variable
env = os.environ.copy()
env['LOCAL_BACKEND'] = 'true'

# Use the venv Python
venv_python = Path('/Users/joshpurtell/Documents/GitHub/synth-ai/.venv/bin/python')
if not venv_python.exists():
    venv_python = sys.executable

# Run the script
script_path = Path('/Users/joshpurtell/Documents/GitHub/synth-ai/demos/image_style_matching/run_notebook.py')

print(f"Running: {venv_python} {script_path}")
print("=" * 80)

process = subprocess.Popen(
    [str(venv_python), str(script_path)],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Stream output
for line in process.stdout:
    if 'UserWarning' not in line and 'celery_app' not in line:
        print(line.rstrip())
    sys.stdout.flush()

process.wait()
sys.exit(process.returncode)


