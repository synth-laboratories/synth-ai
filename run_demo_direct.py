#!/usr/bin/env python3
"""Direct runner for the image style matching demo."""
import os
import sys
from pathlib import Path

# Set environment
os.environ['LOCAL_BACKEND'] = 'true'

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
from demos.image_style_matching.run_notebook import main

if __name__ == "__main__":
    main()


