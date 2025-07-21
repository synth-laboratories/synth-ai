"""
test_taskset_build.py – verifies create_sokoban_taskset().
"""

import asyncio
import json
from pathlib import Path
from uuid import UUID

import pytest

from synth_ai.environments.examples.sokoban.taskset import (
    create_sokoban_taskset,
    SokobanTaskInstance,
)


@pytest.mark.asyncio
async def test_create_and_roundtrip_taskset(tmp_path: Path):
    """
    1. build the task-set
    2. ensure splits are mutually exclusive & cover all ids
    3. serialize → disk → deserialize and check equality of id sets
    """
    ts = await create_sokoban_taskset()

    # -------- split integrity -------- #
    val = set(ts.split_info.val_instance_ids)
    test = set(ts.split_info.test_instance_ids)
    all_ids = {inst.id for inst in ts.instances}

    # everything not explicitly in val or test is considered train
    train = all_ids - val - test

    # pair-wise disjointness checks
    assert train.isdisjoint(val | test)
    assert val.isdisjoint(test)
    assert (train | val | test) == all_ids

    # -------- round-trip serialisation -------- #
    outfile = tmp_path / "instances.json"
    serialised = await asyncio.gather(*(inst.serialize() for inst in ts.instances))
    outfile.write_text(json.dumps(serialised))

    loaded = json.loads(outfile.read_text())
    deser = await asyncio.gather(*(SokobanTaskInstance.deserialize(d) for d in loaded))

    deser_ids = {inst.id for inst in deser if isinstance(inst.id, UUID)}
    assert deser_ids == all_ids
