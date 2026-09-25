"""Ship reviewed API snapshots with the installed SDK.

See specifications/tanha/synth-index: installed clients must carry the same
contract bytes as the reviewed source. Keep openapi/ as the single source and
copy only the three public contract snapshots into the wheel, never other JSON.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SCHEMAS = ("index-v1.json", "research-v1.json", "messaging-v1.json")


class BuildWithContracts(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "synth_ai" / "openapi"
        destination.mkdir(parents=True, exist_ok=True)
        for name in SCHEMAS:
            shutil.copyfile(Path(__file__).parent / "openapi" / name, destination / name)

    def get_outputs(self, include_bytecode=True):
        return [
            *super().get_outputs(include_bytecode),
            *(str(Path(self.build_lib) / "synth_ai" / "openapi" / name) for name in SCHEMAS),
        ]


setup(cmdclass={"build_py": BuildWithContracts})
