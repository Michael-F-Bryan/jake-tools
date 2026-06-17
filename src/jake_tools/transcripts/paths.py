from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self

from pydantic import BaseModel


class Paths(BaseModel):
    root: Path

    @classmethod
    @contextmanager
    def temp(cls) -> Generator[Self, Any, Any]:
        with TemporaryDirectory() as temp_dir:
            yield cls(root=Path(temp_dir))

    @property
    def merged(self) -> Path:
        return self.root / "merged.mp3"

    @property
    def transcript(self) -> Path:
        return self.root / "merged.json"

    @property
    def transcript_markdown(self) -> Path:
        return self.root / "transcript.md"

    @property
    def chapters(self) -> Path:
        return self.root / "chapters.json"

    @property
    def speaker_mapping(self) -> Path:
        return self.root / "speaker-mapping.json"

    @property
    def merge_report(self) -> Path:
        return self.root / "merge-report.json"

    @property
    def minutes(self) -> Path:
        return self.root / "minutes.json"
