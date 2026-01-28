#!/usr/bin/env python3
"""
Run the image style matching notebook end-to-end using papermill.

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    uv run python demos/image_style_matching/run_notebook.py
"""

import os
import sys
from pathlib import Path


def main():
    """Execute the demo notebook using papermill."""
    try:
        import papermill as pm
    except ImportError:
        print("papermill not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "papermill"])
        import papermill as pm

    notebook_dir = Path(__file__).parent
    input_notebook = notebook_dir / "graphgen_image_style_matching.ipynb"
    output_notebook = notebook_dir / "demo_prod_executed.ipynb"

    # Parameters to pass to the notebook
    parameters = {}

    # Use local backend if LOCAL_BACKEND is set
    if os.environ.get("LOCAL_BACKEND", "").lower() in ("1", "true", "yes"):
        parameters["BACKEND_URL"] = "http://127.0.0.1:8000"

    # Pass API key if set
    if os.environ.get("SYNTH_API_KEY"):
        parameters["API_KEY"] = os.environ["SYNTH_API_KEY"]

    print(f"Executing notebook: {input_notebook}")
    print(f"Output will be saved to: {output_notebook}")
    if parameters:
        print(f"Parameters: {parameters}")
    print()

    try:
        pm.execute_notebook(
            str(input_notebook),
            str(output_notebook),
            parameters=parameters,
            cwd=str(notebook_dir),
        )
        print("\n✅ Notebook executed successfully!")
        print(f"Results saved to: {output_notebook}")
    except pm.PapermillExecutionError as e:
        print(f"\n❌ Notebook execution failed at cell {e.cell_index}:")
        print(e.ename, e.evalue)
        sys.exit(1)


if __name__ == "__main__":
    main()
