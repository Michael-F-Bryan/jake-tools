"""Clean-verbatim polishing of a ``transcribe`` result, with source accounting.

:func:`polish` is the entry point (``operations/polish_transcript.py`` is
its only adapter). ``source`` numbers and chunks the words, ``prompts``
holds the two model roles' prompts and reply contracts, ``edits`` decides
what code accepts from them, and ``render`` writes the reader's Markdown and
the word ledger.
"""

from .pipeline import (
    DEFAULT_BUDGET_USD,
    MAX_BUDGET_USD,
    PolishFailedError,
    PolishResult,
    parse_glossary,
    polish,
)
from .source import SourceError

__all__ = [
    "DEFAULT_BUDGET_USD",
    "MAX_BUDGET_USD",
    "PolishFailedError",
    "PolishResult",
    "SourceError",
    "parse_glossary",
    "polish",
]
