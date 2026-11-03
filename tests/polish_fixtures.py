"""Small transcribe results for the polishing tests, built from plain text.

Each turn is ``(speaker, text)``; words get consecutive timings (0.3 s per
word, a 0.5 s pause between turns). Mechanics only: these are not
representative conversations and are never evidence of model quality.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage

WORD_SECONDS = 0.3


def transcript_payload(
    turns: Sequence[tuple[str | None, str]], *, overlap: Sequence[int] = ()
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    clock = 0.0
    index = 0
    for speaker, text in turns:
        words = []
        for token in text.split():
            words.append(
                {
                    "text": token,
                    "start_seconds": round(clock, 2),
                    "end_seconds": round(clock + WORD_SECONDS - 0.05, 2),
                    "confidence": 0.9,
                    "speaker": speaker,
                    "overlap": index in overlap,
                }
            )
            clock += WORD_SECONDS
            index += 1
        segments.append(
            {
                "speaker": speaker,
                "start_seconds": words[0]["start_seconds"],
                "end_seconds": words[-1]["end_seconds"],
                "text": text,
                "words": words,
            }
        )
        clock += 0.5
    speakers = sorted({s for s, _ in turns if s is not None})
    return {
        "inputs": [{"index": 0, "path": "/recordings/meeting.m4a", "bytes": 1}],
        "transcript": {
            "schema_version": 1,
            "engine": {
                "name": "test",
                "version": "0",
                "asr_model": "asr",
                "diarization_model": "diar",
            },
            "timestamp_unit": "seconds",
            "timestamp_origin": "start of the first file",
            "timing_resolution_seconds": 0.08,
            "duration_seconds": clock,
            "files": [
                {
                    "index": 0,
                    "name": "meeting.m4a",
                    "start_seconds": 0.0,
                    "end_seconds": clock,
                    "duration_seconds": clock,
                }
            ],
            "speakers": [
                {"id": s, "speaking_seconds": 1.0, "segment_count": 1} for s in speakers
            ],
            "segments": segments,
            "text": " ".join(t for _, t in turns),
            "unassigned_word_count": 0,
            "limitations": [],
            "processing": {
                "decode_seconds": 0,
                "asr_seconds": 0,
                "diarization_seconds": 0,
                "total_seconds": 0,
            },
        },
    }


def write_transcript(
    path: Path, turns: Sequence[tuple[str | None, str]], **kwargs: Any
) -> Path:
    path.write_text(json.dumps(transcript_payload(turns, **kwargs)))
    return path


Responder = Callable[[str, ClaudeAgentOptions], dict[str, Any] | Exception]


class FakeQuery:
    """A stand-in for the SDK's ``query``: the true external boundary.

    ``respond(prompt, options)`` returns the structured output to report (or
    an exception to raise). It sees which role is asking from the schema
    title. It is not a model; tests using it check mechanics only.
    """

    def __init__(self, respond: Responder, *, cost: float | None = 0.01) -> None:
        self.respond = respond
        self.cost = cost
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))
        answer = self.respond(prompt, options)

        async def stream() -> AsyncIterator[Message]:
            if isinstance(answer, Exception):
                raise answer
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="fake",
                total_cost_usd=self.cost,
                usage={"input_tokens": 10, "output_tokens": 5},
                result=None,
                structured_output=answer,
            )

        return stream()


def role(options: ClaudeAgentOptions) -> str:
    output_format = options.output_format or {}
    return output_format["schema"]["title"]


EMPTY_EDITS: dict[str, Any] = {
    "deletions": [],
    "formats": [],
    "glossary": [],
    "paragraph_breaks": [],
}
EMPTY_REVIEW: dict[str, Any] = {"issues": [], "edit_concerns": []}


def empty(prompt: str, options: ClaudeAgentOptions) -> dict[str, Any]:
    return EMPTY_EDITS if role(options) == "EditDecisions" else EMPTY_REVIEW
