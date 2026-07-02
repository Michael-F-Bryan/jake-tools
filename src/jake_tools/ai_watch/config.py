from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from .models import AiWatchCommandOptions


class AiWatchConfig(BaseModel):
    scout_model: str = "gpt-5.4-mini"
    scout_provider: str = "openai-codex"
    curator_model: str = "gpt-5.5"
    curator_provider: str = "openai-codex"
    vault_path: Path = Path("/Users/work/Documents/Vault")
    discord_target: str = ""
    promote_to_curator_score: float = 3.5
    surface_score: float = 4.3
    max_candidates_per_run: int = 80

    @classmethod
    def from_options(cls, options: AiWatchCommandOptions) -> AiWatchConfig:
        discord_target = options.discord_target or os.environ.get(
            "AI_WATCH_DISCORD_TARGET", ""
        )
        return cls(
            scout_model=options.scout_model,
            scout_provider=options.scout_provider,
            curator_model=options.curator_model,
            curator_provider=options.curator_provider,
            vault_path=options.vault_path,
            discord_target=discord_target,
            max_candidates_per_run=options.max_candidates,
        )
