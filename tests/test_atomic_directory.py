"""Native no-overwrite publication, including a competing publisher."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from synth_ai.sdk.core.atomic import publish_directory

pytestmark = pytest.mark.unit


def test_existing_empty_destination_is_never_replaced(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_text("verified bytes")
    identity = destination.stat().st_ino
    with pytest.raises(FileExistsError):
        publish_directory(source, destination)
    assert destination.stat().st_ino == identity
    assert list(destination.iterdir()) == []
    assert (source / "file").read_text() == "verified bytes"


def test_only_one_competing_directory_can_be_published(tmp_path):
    destination = tmp_path / "destination"
    sources = [tmp_path / name for name in ("first", "second")]
    for source in sources:
        source.mkdir()
        (source / "file").write_text(source.name)
    barrier = Barrier(2)

    def publish(source):
        barrier.wait(timeout=5)
        try:
            publish_directory(source, destination)
            return source.name
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, sources))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert (destination / "file").read_text() == winners[0]
    assert sum(source.exists() for source in sources) == 1
