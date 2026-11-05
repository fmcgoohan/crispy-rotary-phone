"""P0-9 (SEC-5): a decode that writes lots of stderr must not deadlock."""

from __future__ import annotations

import os
import signal
import sys
import textwrap

import pytest

from mrw import ingest


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="needs SIGALRM")
def test_decode_hash_survives_large_stderr(tmp_path, monkeypatch):
    # A stand-in "ffmpeg" that writes 1 MB of errors before any PCM — the
    # pattern of a badly damaged file. With stderr piped but undrained the
    # old code blocked for ever.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "ffmpeg"
    fake.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        sys.stderr.write("x" * (1 << 20))
        sys.stderr.flush()
        sys.stdout.buffer.write(b"\\0" * 44100 * 2 * 2)
    """))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def _hung(*_):
        raise TimeoutError("decode deadlocked")

    old = signal.signal(signal.SIGALRM, _hung)
    signal.alarm(20)
    try:
        sha, duration = ingest._decode_hash_and_duration(tmp_path / "in.wav", 44100, 2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert duration == 1.0 and len(sha) == 64
