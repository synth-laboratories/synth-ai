#!/usr/bin/env python3
"""Retired: internal CloudDevSlot operations moved to synth-dev."""

import sys

print(
    "ERROR: scripts/cloud_slot.py was retired because cdev slots are not "
    "Managed Research CloudDeployments. Use synth-dev/scripts/cloud_dev_slot.py "
    "with CLOUD_DEV_CONTROL_URL and CLOUD_DEV_CONTROL_TOKEN. Product backend "
    "authorities (including api-dev) are forbidden.",
    file=sys.stderr,
)
raise SystemExit(2)
