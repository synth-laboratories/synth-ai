from __future__ import annotations

from .common import forward_to_core


def register(group):
    @group.command("configure")
    def demo_configure():
        forward_to_core(["demo.configure"])
