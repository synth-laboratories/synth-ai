#!/bin/bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
export LOCAL_BACKEND=true
.venv/bin/python demos/image_style_matching/run_notebook.py 2>&1 | grep -v "UserWarning\|celery_app"


