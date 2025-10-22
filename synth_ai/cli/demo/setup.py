from __future__ import annotations

from .common import forward_to_core


def register(group):
    @group.command("setup")
    def demo_setup():
        forward_to_core(["demo.setup"])
