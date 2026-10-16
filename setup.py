"""Ship the provisional typed Index contract with the installed SDK.

See specifications/tanha/synth-index: installed clients must carry the same
contract bytes as the reviewed source. Keep contracts/ as the single source and
copy only the retained Index snapshot into the wheel, never retired contracts.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SCHEMAS = ("index-v1.json", "swarms-v1.json", "mcp-v1.json")


class BuildWithContracts(build_py):
    def run(self) -> None:
        # Setuptools otherwise retains modules deleted by a branch migration.
        # Only the generated package tree belongs to this build command.
        package_output = Path(self.build_lib).resolve() / "synth_ai"
        source = Path(__file__).resolve().parent / "synth_ai"
        if package_output == source or package_output in source.parents:
            raise RuntimeError("The build output must not contain the SDK source tree")
        if package_output.exists():
            shutil.rmtree(package_output)
        super().run()
        destination = Path(self.build_lib) / "synth_ai" / "contracts"
        destination.mkdir(parents=True, exist_ok=True)
        for name in SCHEMAS:
            shutil.copyfile(Path(__file__).parent / "contracts" / name, destination / name)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        return [
            *super().get_outputs(include_bytecode),
            *(str(Path(self.build_lib) / "synth_ai" / "contracts" / name) for name in SCHEMAS),
        ]


setup(cmdclass={"build_py": BuildWithContracts})
