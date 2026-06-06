#!/usr/bin/env python3
"""T0: verify public Research API shape without a backend run."""

from __future__ import annotations

import os
import sys


def main() -> int:
    from synth_ai import SynthClient
    from synth_ai.research import (
        ResearchClient,
        ResearchControlClient,
        ResearchProjectsAPI,
        ResearchRunHandle,
        ResearchRunsAPI,
        ResearchWorkMode,
    )

    base_url = os.environ.get("SYNTH_BACKEND_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("SYNTH_API_KEY", "sk_test_shape_only")

    client = SynthClient(api_key=api_key, base_url=base_url)
    research = client.research
    assert isinstance(research, ResearchClient)
    assert isinstance(research.projects, ResearchProjectsAPI)
    assert isinstance(research.runs, ResearchRunsAPI)
    assert isinstance(research.control(), ResearchControlClient)

    print("T0 ok", ResearchWorkMode.GENERAL.value, research.projects, research.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
