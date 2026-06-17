from collections.abc import Generator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel


class Paths(BaseModel):
    root: Path

    @contextmanager
    @classmethod
    def temp(cls) -> Generator[Self, Any, Any]:
        with TemporaryDirectory() as temp_dir:
            yield cls(root=Path(temp_dir))

    @property
    def merged(self) -> Path:
        return self.root.joinpath("merged.mp3")
