"""Pinned Whisper model revisions (CLAUDE.md: offline by default; D1).

faster-whisper resolves a size name to a Hugging Face repo and, by default,
to that repo's moving `main` branch. Every name here is pinned to the
commit that `mrw models fetch` downloads and transcription loads, so an
upstream re-upload can never change lyrics.json under an unchanged config.

Regenerate only deliberately: a new revision can change transcription
output, so it must ship with a lyrics ALGORITHM_VERSIONS bump.
Resolved 2026-11-20 from the Hugging Face API.
"""

from __future__ import annotations

# size name -> (repo id, commit sha)
WHISPER_MODELS: dict[str, tuple[str, str]] = {
    "tiny.en": ("Systran/faster-whisper-tiny.en", "0d3d19a32d3338f10357c0889762bd8d64bbdeba"),
    "tiny": ("Systran/faster-whisper-tiny", "d90ca5fe260221311c53c58e660288d3deb8d356"),
    "base.en": ("Systran/faster-whisper-base.en", "3d3d5dee26484f91867d81cb899cfcf72b96be6c"),
    "base": ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    "small.en": ("Systran/faster-whisper-small.en", "d1d751a5f8271d482d14ca55d9e2deeebbae577f"),
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "medium.en": ("Systran/faster-whisper-medium.en", "a29b04bd15381511a9af671baec01072039215e3"),
    "medium": ("Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f"),
    "large-v1": ("Systran/faster-whisper-large-v1", "b07c8d4be0be90092aa01a29c975077acb8d15c9"),
    "large-v2": ("Systran/faster-whisper-large-v2", "f0fe81560cb8b68660e564f55dd99207059c092e"),
    "large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
    "large": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478"),
    "distil-large-v2": ("Systran/faster-distil-whisper-large-v2", "fe9b404fc56de3f7c38606ef9ba6fd83526d05e4"),
    "distil-medium.en": ("Systran/faster-distil-whisper-medium.en", "80ddfce281f77766d8943d63109199fc8145dfa5"),
    "distil-small.en": ("Systran/faster-distil-whisper-small.en", "ef77d90526ccd62cde3808ee70626a01e5cf83e4"),
    "distil-large-v3": ("Systran/faster-distil-whisper-large-v3", "c3058b475261292e64a0412df1d2681c06260fab"),
    "distil-large-v3.5": ("distil-whisper/distil-large-v3.5-ct2", "9793ccc07920e0f830e1dba0343efcdf0ef8c903"),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
    "turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
}
