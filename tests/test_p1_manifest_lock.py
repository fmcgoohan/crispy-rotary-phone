"""P1-5 (ORCH-4): concurrent stages on one track keep each other's entries."""

from __future__ import annotations

import threading

import pytest

from mrw.library import Library
from mrw.models import DocumentEntry
from test_m3_smoke import _click_track, _fabricate_track

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)


def test_updates_from_stale_copies_both_survive(tmp_path):
    mix = _click_track(2.0)
    library = Library(tmp_path / "lib")
    track_id = _fabricate_track(library.root, mix, {"drums": mix}, tmp_path)
    # Two "stages" each read the manifest at start, then record at the end:
    # the old whole-file rewrite let the second write erase the first.
    stale_a = library.read_manifest(track_id)
    stale_b = library.read_manifest(track_id)
    lyrics = DocumentEntry(status="failed", error="a")
    video = DocumentEntry(status="failed", error="b")
    library.update_manifest(track_id, lambda m: setattr(m.documents, "lyrics", lyrics))
    library.update_manifest(track_id, lambda m: setattr(m.documents, "video", video))
    final = library.read_manifest(track_id)
    assert final.documents.lyrics.error == "a" and final.documents.video.error == "b"
    assert stale_a.documents.lyrics.status != "failed"  # copies untouched


def test_concurrent_updates_serialise(tmp_path):
    mix = _click_track(2.0)
    library = Library(tmp_path / "lib")
    track_id = _fabricate_track(library.root, mix, {"drums": mix}, tmp_path)
    names = ["lyrics", "video", "structure"] * 10

    def worker(i, name):
        library.update_manifest(
            track_id,
            lambda m: setattr(m.documents, name, DocumentEntry(status="failed", error=str(i))))

    threads = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = library.read_manifest(track_id)
    assert all(getattr(final.documents, n).status == "failed" for n in set(names))
