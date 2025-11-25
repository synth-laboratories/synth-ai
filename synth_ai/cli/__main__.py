#!/usr/bin/env python3
"""
Allow running synth_ai.cli as a module: python -m synth_ai.cli
"""

import sys
import traceback

# Log CLI invocation immediately
if "train" in sys.argv:
    sys.stderr.write(f"[CLI_MAIN] Module synth_ai.cli invoked with args: {sys.argv}\n")
    sys.stderr.flush()

try:
    from . import cli
    
    if "train" in sys.argv:
        sys.stderr.write("[CLI_MAIN] CLI imported successfully\n")
        sys.stderr.flush()
    
    if __name__ == "__main__":
        if "train" in sys.argv:
            sys.stderr.write("[CLI_MAIN] About to call cli()\n")
            sys.stderr.flush()
        try:
            cli()
        except Exception as e:
            sys.stderr.write(f"[CLI_MAIN] CLI call failed: {type(e).__name__}: {e}\n")
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            raise
except Exception as e:
    sys.stderr.write(f"[CLI_MAIN] Import failed: {type(e).__name__}: {e}\n")
    sys.stderr.flush()
    traceback.print_exc(file=sys.stderr)
    raise






