#!/usr/bin/env python3
"""Run pyright and only fail on synth_ai SDK usage errors.

This script runs pyright and filters the output to only fail when there are
errors related to synth_ai SDK usage (imports, parameter names, types).
External library type errors (openai, datasets, etc.) are reported but don't
cause the script to fail.

Usage:
    python scripts/check_synth_ai_types.py [files...]
"""

import subprocess
import sys

# Error messages that indicate synth_ai SDK misuse
SYNTH_SDK_ERROR_PATTERNS = [
    "synth_ai.sdk",
    "synth_ai.data",
    "synth_ai.core",
    "TaskDescriptor",
    "DatasetInfo",
    "InferenceInfo",
    "LimitsInfo",
    "TaskInfo",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutMetrics",
    "LocalAPIConfig",
    "EvalJobConfig",
    "PromptLearningJob",
]


def main():
    # Run pyright and capture output
    args = [sys.executable, "-m", "pyright"] + sys.argv[1:]
    result = subprocess.run(args, capture_output=True, text=True)

    # Print all output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Check for synth_ai SDK errors
    output = result.stdout + result.stderr
    synth_errors = []

    for line in output.splitlines():
        # Check if this is an error line (contains "error:")
        if "error:" not in line.lower():
            continue

        # Check if this error relates to synth_ai SDK
        for pattern in SYNTH_SDK_ERROR_PATTERNS:
            if pattern in line:
                synth_errors.append(line)
                break

    if synth_errors:
        print("\n" + "=" * 60)
        print("SYNTH_AI SDK TYPE ERRORS FOUND:")
        print("=" * 60)
        for error in synth_errors:
            print(error)
        print("=" * 60)
        sys.exit(1)

    # If no synth_ai errors, exit successfully even if there are external library errors
    if result.returncode != 0:
        print("\n" + "-" * 60)
        print(f"Note: {result.returncode} external library type issues found.")
        print("These don't block commits as synth_ai SDK usage is correct.")
        print("-" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
