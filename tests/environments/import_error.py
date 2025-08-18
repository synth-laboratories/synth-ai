#!/usr/bin/env python3
"""
Test synth_ai.environments import functionality
"""

print("=== Testing synth_ai.environments import issue ===\n")

print("1. Attempting to import synth_ai.environments:")
try:
    import synth_ai.environments

    print("✓ Success: synth_ai.environments imported")
except ImportError as e:
    print(f"✗ Failed: {e}")

print("\n2. Checking what's in the synth_ai.environments package directory:")
import os
import sys

print(f"   Python path: {sys.path}")

# Try to find where synth_ai.environments is installed
for path in sys.path:
    potential_path = os.path.join(path, "synth_ai.environments")
    if os.path.exists(potential_path):
        print(f"   Found synth_ai.environments at: {potential_path}")
        contents = os.listdir(potential_path)
        print(f"   Contents: {contents}")

        print("\n3. Looking at synth_ai.environments/__init__.py:")
        init_file = os.path.join(potential_path, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file) as f:
                print("   " + "\n   ".join(f.read().splitlines()[:10]))  # First 10 lines
        break
else:
    print("   synth_ai.environments not found in any Python path")

print("\n4. The issue:")
print("   - synth_ai.environments/__init__.py tries to import 'service' module")
print("   - But there's no 'service' directory in the package")
print("   - This causes: ImportError: cannot import name 'service'")

print("\n5. What we need from synth_ai.environments for our tests:")
print("   - synth_ai.environments.schema (for ToolCall, StepResult, etc.)")
print("   - synth_ai.environments.environment (for Environment base classes)")
print("   - But these imports fail due to the 'service' import error")
