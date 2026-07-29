"""Shared error taxonomy for the transcripts package.

Every domain-specific error raised anywhere in ``jake_tools.transcripts``
subclasses :class:`TranscriptError`, so recipe- and CLI-level code can catch
one type and be confident it has covered every typed failure the pipeline can
raise, without also swallowing unrelated bugs (e.g. ``pydantic.ValidationError``
escaping from code that forgot to handle it, or a plain ``ValueError`` from a
programming mistake).
"""

from __future__ import annotations


class TranscriptError(RuntimeError):
    """Base class for all typed errors raised by the transcripts package."""
