#!/usr/bin/env python3
"""
Allow running synth_ai as a module: python -m synth_ai
"""

import sys
import traceback

# Log CLI invocation immediately
if "train" in sys.argv:
    sys.stderr.write(f"[SYNTH_AI_MAIN] Module invoked with args: {sys.argv}\n")
    sys.stderr.flush()

try:
    from .cli import cli
    
    if "train" in sys.argv:
        sys.stderr.write("[SYNTH_AI_MAIN] CLI imported successfully\n")
        sys.stderr.flush()
    
    if __name__ == "__main__":
        if "train" in sys.argv:
            sys.stderr.write("[SYNTH_AI_MAIN] About to call cli()\n")
            sys.stderr.flush()
        try:
            cli()
        except Exception as e:
            sys.stderr.write(f"[SYNTH_AI_MAIN] CLI call failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            raise
except Exception as e:
    sys.stderr.write(f"[SYNTH_AI_MAIN] Import failed: {type(e).__name__}: {e}\n")
    sys.stderr.flush()
    traceback.print_exc(file=sys.stderr)
    raise
