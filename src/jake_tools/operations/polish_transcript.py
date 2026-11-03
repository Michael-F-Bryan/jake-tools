"""``polish_transcript``: a clean-verbatim draft of a saved ``transcribe`` result.

Part of the ``claude`` capability, not ``transcription``: it sends the
transcript's words to Anthropic's models through the local Claude CLI
(tool-less ``ClaudeAgent.run_structured`` calls), and an agent granted only
local transcription must not gain that. It needs no transcription endpoint,
so it is always available. The handler validates arguments and maps errors;
:mod:`jake_tools.polishing` does the work.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..polishing import (
    DEFAULT_BUDGET_USD,
    MAX_BUDGET_USD,
    PolishFailedError,
    PolishResult,
    SourceError,
    parse_glossary,
    polish,
)
from ..polishing.pipeline import DEADLINE_SECONDS, MAX_CHUNKS
from ..polishing.source import TARGET_CHUNK_WORDS
from . import Handler, Operation
from .errors import ToolError

DESCRIPTION = (
    "Write a clean-verbatim, speaker-labelled Markdown draft of a saved "
    "`transcribe` result into a NEW directory (it must not exist; its parent "
    "must). The transcript's words are sent to Anthropic's Claude models; the "
    "audio is not. transcript_path and output_directory are absolute paths as "
    "jake-tools sees them; the source file is never modified. Per chunk of "
    f"about {TARGET_CHUNK_WORDS} words, one conservative edit pass (fillers, "
    "stutters, abandoned restarts, punctuation/case, optional glossary "
    "candidates) whose decisions code validates against the source words, then "
    "a detection-only review that flags places to check and can undo edits but "
    "never rewrites. No ordinary-word repair, no speaker changes. glossary: "
    "optional 'TERM' or 'TERM: context' strings; matches are shown as tentative "
    "[TERM?] candidates. start_seconds/end_seconds select a window of the "
    f"recording (at most {MAX_CHUNKS} chunks per call). max_budget_usd caps the "
    f"whole call (default {DEFAULT_BUDGET_USD:g}, at most {MAX_BUDGET_USD:g}). "
    f"Finishes within about {DEADLINE_SECONDS:.0f} s. Returns status "
    "(complete|partial), file paths (transcript.md, ledger.json, review.json, "
    "metadata.json, diagnostics/), per-chunk outcomes, counts and estimated cost."
)


def build(config: Config) -> Handler:
    async def polish_transcript(
        transcript_path: str,
        output_directory: str,
        glossary: list[str] | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        max_budget_usd: float | None = None,
    ) -> PolishResult:
        try:
            return await polish(
                transcript_path=Path(transcript_path).expanduser(),
                output_directory=Path(output_directory).expanduser(),
                glossary=parse_glossary(glossary),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                budget_usd=DEFAULT_BUDGET_USD
                if max_budget_usd is None
                else max_budget_usd,
            )
        except SourceError as exc:
            raise ToolError(
                "invalid_argument", str(exc), detail={"argument": exc.argument}
            ) from exc
        except PolishFailedError as exc:
            raise ToolError(
                "upstream_error",
                str(exc),
                detail={
                    "integration": "claude",
                    "output_directory": str(exc.output_directory),
                },
            ) from exc

    return polish_transcript


OPERATION = Operation(
    name="polish_transcript",
    capability="claude",
    skill="transcription",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
