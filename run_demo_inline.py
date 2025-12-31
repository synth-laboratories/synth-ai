#!/usr/bin/env python3
"""Run the demo by directly importing and executing."""
import os
import sys
from pathlib import Path

# Set environment before any imports
os.environ['LOCAL_BACKEND'] = 'true'

# Change to synth-ai directory
os.chdir('/Users/joshpurtell/Documents/GitHub/synth-ai')

# Add to Python path
sys.path.insert(0, '/Users/joshpurtell/Documents/GitHub/synth-ai')

# Now import and run
try:
    from demos.image_style_matching.run_notebook import main
    print("Starting demo execution...")
    print("=" * 80)
    main()
    print("=" * 80)
    print("Demo completed successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


