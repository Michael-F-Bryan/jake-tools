from pathlib import Path

from ..hermes import Hermes
from .paths import Paths


def process_obsidian_recording(hermes: Hermes, obsidian_note: Path):
    """
    Process the recording(s) in a given Obsidian note into a polished,
    chapterized transcript.
    """
    with Paths.temp() as paths:
        raise NotImplementedError("Not implemented")
