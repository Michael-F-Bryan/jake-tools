"""Strict envelopes for JSON artefacts persisted by Jake's tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class CacheEnvelope(BaseModel):
    """Reject unrelated JSON while retaining the supported legacy shape."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def _default_supported_legacy_version(cls, value: Any) -> Any:
        """Accept pre-versioned cache files and upgrade them on the next write."""
        if isinstance(value, dict) and "schema_version" not in value:
            return {**value, "schema_version": 1}
        return value
