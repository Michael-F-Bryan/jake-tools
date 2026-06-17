from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Self

from pydantic import BaseModel


class Paths(BaseModel):
    root: Path

    @classmethod
    def temp(cls) -> Self:
        with TemporaryDirectory() as temp_dir:
            return cls(root=Path(temp_dir))

    @property
    def merged(self) -> Path:
        return self.root.joinpath("merged.mp3")
