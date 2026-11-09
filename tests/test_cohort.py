import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts/check_cohort.py"
spec = importlib.util.spec_from_file_location("check_cohort", MODULE)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def test_committed_contract_and_provenance_are_checked_independently(tmp_path):
    core, sdk = tmp_path / "core", tmp_path / "sdk"
    core.mkdir()
    (sdk / "contracts").mkdir(parents=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(core), *args], text=True).strip()

    git("init", "-q")
    source = core / "contract.yaml"
    source.write_bytes(b"openapi: 3.1.0\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-qm",
        "initial",
    )
    revision = git("rev-parse", "HEAD")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (sdk / "contracts/synth-v1.openapi.yaml").write_bytes(source.read_bytes())
    (sdk / "contracts/SOURCE.toml").write_text(
        f'commit = "{revision}"\npath = "contract.yaml"\nsha256 = "{digest}"\n'
    )
    assert cohort.check(sdk, core_checkout=core)["aligned"] is True
    source.write_bytes(b"openapi: 3.0.0\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-qm",
        "changed",
    )
    result = cohort.check(sdk, core_checkout=core)
    assert result["aligned"] is False and result["provenance_verified"] is True
    assert cohort.check(sdk, core_checkout=core, core_revision=revision)["aligned"] is True
    (sdk / "contracts/synth-v1.openapi.yaml").write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA256"):
        cohort.check(sdk)
