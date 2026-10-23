"""Exact packaged research guidance for admission pins and MCP discovery."""

import hashlib
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class ResearchSkill:
    name: str
    revision: str
    sha256: str
    content: str

    @property
    def uri(self) -> str:
        return "synth://skills/forge-research/" + self.sha256


def research_skill() -> ResearchSkill:
    raw = files("synth_ai.sdk.forge").joinpath("skills/forge-research/SKILL.md").read_bytes()
    return ResearchSkill(
        "forge-research", "5", hashlib.sha256(raw).hexdigest(), raw.decode("utf-8")
    )
