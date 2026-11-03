"""Deterministic mechanics of transcript polishing.

The guards, chunking, rendering and the pipeline's file, budget, failure and
call-configuration behaviour run for real. Decisions fed to the guards are
inputs to validators, and the only fake is the SDK's ``query`` (the external
boundary), whose replies are mechanical. None of this is evidence that a
model makes good decisions: ``test_polish_transcript_live.py`` is.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from polish_fixtures import (
    EMPTY_EDITS,
    EMPTY_REVIEW,
    FakeQuery,
    empty,
    role,
    transcript_payload,
    write_transcript,
)

from jake_tools.claude import ClaudeAgent, ClaudeAgentError
from jake_tools.polishing import (
    PolishFailedError,
    SourceError,
    parse_glossary,
    polish,
)
from jake_tools.polishing import pipeline as pipeline_module
from jake_tools.polishing.edits import (
    EditState,
    apply_edit_decisions,
    apply_review,
)
from jake_tools.polishing.prompts import (
    EditDecisions,
    GlossaryEntry,
    ReviewFindings,
)
from jake_tools.polishing.render import ledger_words
from jake_tools.polishing.source import (
    MAX_CHUNK_WORDS,
    Chunk,
    SourceWord,
    Window,
    load_source,
    plan_chunks,
    select_window,
)
from jake_tools.transcription import TranscriptionResult

Turns = list[tuple[str | None, str]]


def _words(turns: Turns, tmp_path: Path, **kwargs: Any) -> tuple[SourceWord, ...]:
    return load_source(write_transcript(tmp_path / "t.json", turns, **kwargs)).words


def _whole(words: tuple[SourceWord, ...]) -> tuple[EditState, Chunk]:
    window = Window(first_word=0, last_word=len(words) - 1)
    chunk = Chunk(
        index=1,
        first_word=0,
        last_word=len(words) - 1,
        context_first=0,
        context_last=len(words) - 1,
    )
    return EditState.start(words, window), chunk


def _id(words: tuple[SourceWord, ...], text: str, nth: int = 0) -> int:
    return [w.id for w in words if w.text == text][nth]


def _decide(
    turns: Turns,
    tmp_path: Path,
    *,
    deletions: Sequence[dict[str, Any]] = (),
    formats: Sequence[dict[str, Any]] = (),
    glossary_decisions: Sequence[dict[str, Any]] = (),
    glossary: tuple[GlossaryEntry, ...] = (),
    **kwargs: Any,
) -> tuple[EditState, list[str], str]:
    """Apply decisions; returns the state, rejection reasons, reader text."""
    words = _words(turns, tmp_path, **kwargs)
    state, chunk = _whole(words)
    decisions = _decisions(deletions, formats, glossary_decisions)
    rejected = apply_edit_decisions(state, chunk, decisions, glossary)
    text = " ".join(w.display for w in ledger_words(state) if w.display is not None)
    return state, [r.reason for r in rejected], text


def _decisions(
    deletions: Sequence[dict[str, Any]] = (),
    formats: Sequence[dict[str, Any]] = (),
    glossary: Sequence[dict[str, Any]] = (),
) -> EditDecisions:
    return EditDecisions.model_validate(
        {
            "deletions": list(deletions),
            "formats": list(formats),
            "glossary": list(glossary),
            "paragraph_breaks": [],
        }
    )


def _span(
    first: int, last: int, text: str, kind: str, reason: str = "r"
) -> dict[str, Any]:
    return {
        "first_word": first,
        "last_word": last,
        "text": text,
        "kind": kind,
        "reason": reason,
    }


# --- deterministic cleanup ----------------------------------------------------------


def test_standalone_fillers_go_and_hand_on_their_sentence_end(tmp_path: Path) -> None:
    _, rejected, text = _decide(
        [("speaker_1", "Um, we went home um. Then uh-huh mm-hmm we left")], tmp_path
    )
    assert rejected == []
    assert text == "We went home. Then uh-huh mm-hmm we left"


# --- repetitions ----------------------------------------------------------------------


def test_a_repetition_never_deletes_both_copies(tmp_path: Path) -> None:
    # Observed in the prototype: a "duplicate" covering both words of
    # "you you" left "Even so, ha also have...".
    turns: Turns = [("speaker_1", "Even so, you you ha also have to do it")]
    words = _words(turns, tmp_path)
    first = _id(words, "you")
    _, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[_span(first, first + 1, "you you", "repetition")],
    )
    assert rejected == ["no identical copy by the same speaker stays next to it"]
    assert "you you" in text


def test_one_copy_of_a_stutter_goes_and_its_twin_is_pinned(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "and if you if you want it we go")]
    words = _words(turns, tmp_path)
    first = _id(words, "if")
    second = _id(words, "if", 1)
    state, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[
            _span(first, first + 1, "if you", "repetition"),
            _span(second, second + 1, "if you", "repetition"),
        ],
    )
    assert text == "And if you want it we go"
    assert rejected == [f"an accepted edit relies on word {second} staying"]
    assert [e.kind for e in state.edits.values()] == ["repetition"]


def test_several_stuttered_copies_go_when_one_stays(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "Right. I I I think so, so")]
    _, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[
            _span(1, 2, "I I", "repetition"),
            # A repeated intensifier may be emphasis, however it is cut.
            _span(5, 6, "so, so", "repetition"),
        ],
    )
    assert text == "Right. I think so, so"
    assert rejected == ["a repeated 'so' may be emphasis"]


def test_repeated_negation_may_be_emphasis(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "no no that is wrong")]
    _, rejected, text = _decide(
        turns, tmp_path, deletions=[_span(0, 0, "no", "repetition")]
    )
    assert rejected == ["a repeated 'no' may be emphasis"]
    assert text.startswith("No no")


def test_a_copy_said_by_someone_else_is_not_a_stutter(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "we need the"), ("speaker_2", "the budget")]
    _, rejected, _ = _decide(
        turns, tmp_path, deletions=[_span(2, 2, "the", "repetition")]
    )
    assert rejected == ["no identical copy by the same speaker stays next to it"]


def test_overlapping_speech_is_not_deduplicated(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "yes I I agree")]
    _, rejected, _ = _decide(
        turns, tmp_path, deletions=[_span(1, 1, "I", "repetition")], overlap=[1]
    )
    assert rejected == [
        "unattributed or overlapping speech may be another speaker's copy"
    ]


# --- restarts and self-corrections ------------------------------------------------


@pytest.mark.parametrize(
    ("source", "span", "missing"),
    [
        ("we meet on Tuesday, no, Wednesday at noon", "on Tuesday, no,", "'Tuesday,'"),
        ("it is not, it is ready now", "it is not,", "'not,'"),
        ("he said, she said it was fine", "he said,", "'he'"),
        ("we need two, three people there", "two,", "'two,'"),
    ],
)
def test_a_self_correction_is_not_a_restart(
    tmp_path: Path, source: str, span: str, missing: str
) -> None:
    words = _words([("speaker_1", source)], tmp_path)
    first = _id(words, span.split()[0])
    last = first + len(span.split()) - 1
    _, rejected, text = _decide(
        [("speaker_1", source)],
        tmp_path,
        deletions=[_span(first, last, span, "restart")],
    )
    assert len(rejected) == 1, rejected
    assert missing.strip("'") in source  # the corrected word stays
    assert text.casefold() == source.casefold()


def test_a_restart_that_repeats_its_meaning_words_is_allowed(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "I think we, I think we should go")]
    _, rejected, text = _decide(
        turns, tmp_path, deletions=[_span(0, 2, "I think we,", "restart")]
    )
    assert rejected == []
    assert text == "I think we should go"


def test_restarts_and_fillers_cannot_cross_or_empty_a_turn(tmp_path: Path) -> None:
    turns: Turns = [
        ("speaker_1", "so we should go"),
        ("speaker_2", "you know"),
        ("speaker_1", "go now"),
    ]
    _, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[
            _span(3, 4, "go you", "restart"),
            _span(4, 5, "you know", "filler"),
        ],
    )
    assert rejected == ["spans a speaker change", "would remove a whole speaker turn"]
    assert text == "So we should go You know Go now"


def test_a_filler_is_only_a_set_off_discourse_marker(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "it is like not kind of ready, you know, today")]
    words = _words(turns, tmp_path)
    _, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[
            _span(_id(words, "not"), _id(words, "not"), "not", "filler"),
            _span(_id(words, "kind"), _id(words, "of"), "kind of", "filler"),
            _span(_id(words, "you"), _id(words, "know,"), "you know,", "filler"),
        ],
    )
    only = "only a set-off 'you know' or 'I mean' is a filler (code already removes um/uh/erm)"
    assert rejected == [only, only]
    assert text == "It is like not kind of ready, today"


def test_spans_must_quote_their_words_and_stay_in_the_chunk(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "alpha bravo charlie delta echo foxtrot golf hotel")]
    words = _words(turns, tmp_path)
    state, chunk = _whole(words)
    narrow = chunk.model_copy(update={"first_word": 1, "last_word": 6})
    decisions = _decisions(
        [_span(0, 0, "alpha", "filler"), _span(5, 5, "bravo", "filler")]
    )
    reasons = [r.reason for r in apply_edit_decisions(state, narrow, decisions, ())]
    assert reasons == [
        "outside this chunk's editable words",
        "text does not match the source words of the span",
    ]


def test_a_span_one_word_off_its_quote_is_found(tmp_path: Path) -> None:
    # Observed with real models: IDs cited one word away from the quote.
    turns: Turns = [("speaker_1", "Okay. And then and then the next one, we go")]
    state, rejected, text = _decide(
        turns,
        tmp_path,
        deletions=[
            # Both copies are as near as each other: either may go.
            _span(2, 3, "and then", "repetition"),
            # One word off, and only one placement fits.
            _span(9, 9, "one,", "filler"),
        ],
    )
    assert rejected == [
        "only a set-off 'you know' or 'I mean' is a filler (code already removes um/uh/erm)"
    ]
    assert text == "Okay. And then the next one, we go"
    (edit,) = [e for e in state.edits.values() if e.origin == "edit_model"]
    assert (edit.first_word, edit.last_word, edit.cited) == (1, 2, (2, 3))


def test_an_ambiguous_quote_is_refused(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "so go and go home")]
    # "go" is one word either side of the cited "and": neither is chosen.
    _, rejected, _ = _decide(turns, tmp_path, deletions=[_span(2, 2, "go", "filler")])
    assert rejected == ["text does not match the source words of the span"]


def test_deletions_are_budgeted_per_chunk(tmp_path: Path) -> None:
    names = [f"word{chr(97 + i)}" for i in range(20)]
    turns: Turns = [("speaker_1", " ".join(f"{n} {n}" for n in names))]
    deletions = [_span(2 * i, 2 * i, n, "repetition") for i, n in enumerate(names)]
    _, rejected, _ = _decide(turns, tmp_path, deletions=deletions)
    assert rejected.count("exceeds this chunk's deletion budget of 10 words") == 10


# --- formats ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "display", "reason"),
    [
        ("going", "going.", None),
        ("dfes", "Dfes", None),
        ("going", "goin'.", "changes more than case and trailing punctuation"),
        ("ready.", "ready?", "adds or removes a question mark"),
        ("ready?", "ready.", "adds or removes a question mark"),
        ("going", "going.[R1]", "changes more than case and trailing punctuation"),
        ("Laura", "laura", "lowercases a possible name"),
    ],
)
def test_formats_change_only_case_and_trailing_punctuation(
    tmp_path: Path, source: str, display: str, reason: str | None
) -> None:
    turns: Turns = [("speaker_1", f"we are {source} home")]
    _, rejected, text = _decide(
        turns, tmp_path, formats=[{"word": 2, "display": display}]
    )
    assert rejected == ([] if reason is None else [reason])
    if reason is None:
        assert display in text


# --- glossary ---------------------------------------------------------------------------

GLOSSARY = (GlossaryEntry(term="WAERN", note="radio course"), GlossaryEntry(term="FRO"))


def test_glossary_candidates_are_occurrence_level_and_tentative(tmp_path: Path) -> None:
    turns: Turns = [
        ("speaker_1", "basic and advanced wearing, and F RO training."),
        ("speaker_2", "I was wearing a jacket. Then Brad left."),
    ]
    words = _words(turns, tmp_path)
    _, rejected, text = _decide(
        turns,
        tmp_path,
        glossary=GLOSSARY,
        glossary_decisions=[
            {
                "first_word": 3,
                "last_word": 3,
                "text": "wearing,",
                "term": "WAERN",
                "reason": "radio course context",
            },
            {
                "first_word": 5,
                "last_word": 6,
                "text": "F RO",
                "term": "FRO",
                "reason": "split acronym",
            },
            {
                "first_word": _id(words, "Brad"),
                "last_word": _id(words, "Brad"),
                "text": "Brad",
                "term": "WAERN",
                "reason": "x",
            },
            {
                "first_word": 0,
                "last_word": 0,
                "text": "basic",
                "term": "VCC",
                "reason": "x",
            },
        ],
    )
    assert "advanced [WAERN?], and [FRO?] training." in text
    assert "I was wearing a jacket." in text  # the other occurrence is untouched
    assert rejected == [
        "not spelled plausibly like the term",
        "term is not in the supplied glossary",
    ]


def test_glossary_cannot_replace_negation_or_numbers(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "we did not go to two")]
    _, rejected, _ = _decide(
        turns,
        tmp_path,
        glossary=(GlossaryEntry(term="KNOTT"), GlossaryEntry(term="TWOS")),
        glossary_decisions=[
            {
                "first_word": 2,
                "last_word": 2,
                "text": "not",
                "term": "KNOTT",
                "reason": "x",
            },
            {
                "first_word": 5,
                "last_word": 5,
                "text": "two",
                "term": "TWOS",
                "reason": "x",
            },
        ],
    )
    assert rejected == [
        "would replace a negation word ('not')",
        "would replace a number word ('two')",
    ]


def test_glossary_sentence_end_is_its_last_source_word(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "we did wearing and then left")]
    _, _, text = _decide(
        turns,
        tmp_path,
        glossary=GLOSSARY,
        glossary_decisions=[
            {
                "first_word": 2,
                "last_word": 2,
                "text": "wearing",
                "term": "WAERN",
                "reason": "x",
            },
        ],
    )
    # The candidate's "?" is doubt, not a question: "and" stays lower case.
    assert text == "We did [WAERN?] and then left"


def test_glossary_strings_parse_and_are_bounded() -> None:
    assert parse_glossary(["WAERN: WA radio course", "FRO"]) == (
        GlossaryEntry(term="WAERN", note="WA radio course"),
        GlossaryEntry(term="FRO"),
    )
    for bad in (["[x]"], ["x", "X"], [""], ["a" * 61]):
        with pytest.raises(SourceError) as excinfo:
            parse_glossary(bad)
        assert excinfo.value.argument == "glossary"


# --- review -------------------------------------------------------------------------------


def test_review_points_merges_and_restores_but_never_rewrites(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "I like flee the training side, you know, honestly")]
    words = _words(turns, tmp_path)
    state, chunk = _whole(words)
    like = _id(words, "you")
    apply_edit_decisions(
        state,
        chunk,
        _decisions([_span(like, like + 1, "you know,", "filler")]),
        (),
    )
    (edit,) = [e for e in state.edits.values() if e.origin == "edit_model"]
    findings = ReviewFindings.model_validate(
        {
            "issues": [
                {
                    "first_word": 2,
                    "last_word": 3,
                    "text": "flee the",
                    "category": "possible_misrecognition",
                    "note": "Odd verb.",
                },
                {
                    "first_word": 3,
                    "last_word": 5,
                    "text": "the training side,",
                    "category": "unclear_meaning",
                    "note": "Unclear.",
                },
                {
                    "first_word": 0,
                    "last_word": 0,
                    "text": "nonsense",
                    "category": "fragment",
                    "note": "x",
                },
            ],
            "edit_concerns": [
                {"edit_id": edit.id, "note": "'you know' may address the listener."},
                {"edit_id": "f0", "note": "not a model edit"},
            ],
        }
    )
    items, rejected = apply_review(state, chunk, findings)
    assert [(i.first_word, i.last_word, i.categories) for i in items] == [
        (2, 5, ["possible_misrecognition", "unclear_meaning"])
    ]
    assert [r.reason for r in rejected] == [
        "text does not match the source words of the span",
        "not an accepted edit of this chunk",
    ]
    assert state.edits[edit.id].status == "restored_after_review"
    text = " ".join(w.display for w in ledger_words(state) if w.display)
    assert text == "I like flee the training side, you know, honestly"


# --- chunking ---------------------------------------------------------------------------


def test_chunks_own_every_word_once_and_split_long_turns_at_sentences(
    tmp_path: Path,
) -> None:
    sentence = "this is a sentence of ten words about the plan."
    turns: Turns = [
        ("speaker_1", " ".join([sentence] * 150)),
        ("speaker_2", "short reply."),
        (None, " ".join([sentence] * 60)),
    ]
    words = _words(turns, tmp_path)
    window = select_window(words, None, None)
    chunks = plan_chunks(words, window)
    owned = [i for c in chunks for i in c.ids()]
    assert owned == list(range(len(words)))
    assert all(c.size <= MAX_CHUNK_WORDS for c in chunks)
    for chunk in chunks[:-1]:
        assert words[chunk.last_word].text.endswith(".")
    for chunk in chunks:
        assert (
            chunk.context_first
            <= chunk.first_word
            <= chunk.last_word
            <= chunk.context_last
        )


def test_a_window_selects_words_by_start_time(tmp_path: Path) -> None:
    words = _words([("speaker_1", "a b c d e f g h")], tmp_path)
    window = select_window(words, 0.6, 1.5)
    assert (window.first_word, window.last_word) == (2, 4)
    with pytest.raises(SourceError):
        select_window(words, 100.0, None)
    with pytest.raises(SourceError):
        select_window(words, 2.0, 1.0)


# --- the pipeline -----------------------------------------------------------------------

MEETING: Turns = [
    ("speaker_1", "Um, so we we need to book the room."),
    (None, "Uh"),
    ("speaker_1", "And I think the the budget is not final."),
    ("speaker_2", "Yeah, okay. Twelve fifty euro, right?"),
]


async def _polish(
    tmp_path: Path, fake: FakeQuery, *, turns: Turns = MEETING, **kwargs: Any
) -> tuple[Path, Path, Any]:
    source = write_transcript(tmp_path / "raw.json", turns)
    out = tmp_path / "polished"
    result = await polish(
        transcript_path=source,
        output_directory=out,
        agent=ClaudeAgent(run_query=fake),
        **kwargs,
    )
    return source, out, result


async def test_writes_a_fresh_directory_and_leaves_the_source_alone(
    tmp_path: Path,
) -> None:
    def respond(prompt: str, options: ClaudeAgentOptions) -> dict[str, Any]:
        if role(options) == "EditDecisions":
            return {
                **EMPTY_EDITS,
                "deletions": [_span(2, 2, "we", "repetition")],
            }
        return {
            **EMPTY_REVIEW,
            "issues": [
                {
                    "first_word": 21,
                    "last_word": 23,
                    "text": "Twelve fifty euro,",
                    "category": "unclear_meaning",
                    "note": "12.50 or 50?",
                }
            ],
        }

    raw_before = transcript_payload(MEETING)
    source, out, result = await _polish(tmp_path, FakeQuery(respond))
    assert json.loads(source.read_text()) == raw_before
    assert result.status == "complete" and result.source_unchanged
    assert sorted(p.name for p in out.iterdir()) == [
        "diagnostics",
        "ledger.json",
        "metadata.json",
        "review.json",
        "transcript.md",
    ]
    assert sorted(p.name for p in (out / "diagnostics").iterdir()) == [
        "chunk-01-edit-prompt.txt",
        "chunk-01-edit-response.json",
        "chunk-01-review-prompt.txt",
        "chunk-01-review-response.json",
    ]
    markdown = (out / "transcript.md").read_text()
    # The unassigned "Uh" turn held only a filler: one speaker_1 block remains.
    assert (
        "**[00:00] speaker_1**\n\nSo we need to book the room.\n\nAnd I think the the budget is not final."
        in markdown
    )
    assert (
        "**[00:" in markdown
        and "] speaker_2**\n\nYeah, okay. Twelve [R1] fifty euro, right?" in markdown
    )
    assert (
        "- **R1** [" in markdown
        and "unclear meaning: “Twelve fifty euro,” — 12.50 or 50?" in markdown
    )
    ledger = json.loads((out / "ledger.json").read_text())
    assert [w["id"] for w in ledger["words"]] == list(range(25))
    assert {w["status"] for w in ledger["words"]} == {"retained", "removed"}
    review = json.loads((out / "review.json").read_text())
    assert (
        review["complete"] is True
        and review["items"][0]["text"] == "Twelve fifty euro,"
    )
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["policy"]["tools"] == []
    assert metadata["result"]["source_sha256"] == result.source_sha256
    assert result.counts.removed_by_kind == {"filler": 2, "repetition": 1}
    assert result.estimated_cost_usd == pytest.approx(0.02)


async def test_every_call_is_tool_less_bounded_and_isolated(tmp_path: Path) -> None:
    fake = FakeQuery(empty)
    await _polish(tmp_path, fake)
    by_role = {role(options): options for _, options in fake.calls}
    assert by_role["EditDecisions"].model == "sonnet"
    assert by_role["ReviewFindings"].model == "opus"
    for options in by_role.values():
        assert options.tools == [] and options.allowed_tools == []
        assert options.mcp_servers == {} and options.strict_mcp_config
        assert options.setting_sources == []
        assert options.permission_mode == "dontAsk"
        assert options.max_turns == 3
        assert options.max_budget_usd is not None and options.max_budget_usd <= 1.0


async def test_an_existing_output_directory_is_refused_untouched(
    tmp_path: Path,
) -> None:
    out = tmp_path / "polished"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    with pytest.raises(SourceError) as excinfo:
        await polish(
            transcript_path=write_transcript(tmp_path / "raw.json", MEETING),
            output_directory=out,
            agent=ClaudeAgent(run_query=FakeQuery(empty)),
        )
    assert excinfo.value.argument == "output_directory"
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


@pytest.mark.parametrize(
    ("content", "argument"),
    [
        ("not json", "transcript_path"),
        (json.dumps({"transcript": {"schema_version": 1}}), "transcript_path"),
    ],
)
async def test_unreadable_sources_are_refused_before_writing(
    tmp_path: Path, content: str, argument: str
) -> None:
    source = tmp_path / "raw.json"
    source.write_text(content)
    with pytest.raises(SourceError) as excinfo:
        await polish(
            transcript_path=source,
            output_directory=tmp_path / "out",
            agent=ClaudeAgent(run_query=FakeQuery(empty)),
        )
    assert excinfo.value.argument == argument
    assert not (tmp_path / "out").exists()


async def test_a_bare_transcript_object_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "bare.json"
    source.write_text(json.dumps(transcript_payload(MEETING)["transcript"]))
    result = await polish(
        transcript_path=source,
        output_directory=tmp_path / "out",
        agent=ClaudeAgent(run_query=FakeQuery(empty)),
    )
    assert result.counts.window_words == 25
    TranscriptionResult.model_validate(transcript_payload(MEETING))


async def test_a_failed_edit_call_is_loud_and_review_still_runs(tmp_path: Path) -> None:
    sentence = "we talk about the plan for the next quarter in detail."
    turns: Turns = [
        ("speaker_1", " ".join([sentence] * 100)),
        ("speaker_2", " ".join([sentence] * 100)),
    ]

    def respond(prompt: str, options: ClaudeAgentOptions) -> dict[str, Any] | Exception:
        if role(options) == "EditDecisions" and "decide only on ids 0 " not in prompt:
            return ClaudeAgentError("agent call failed: error_max_budget_usd")
        return empty(prompt, options)

    _, out, result = await _polish(tmp_path, FakeQuery(respond), turns=turns)
    assert result.status == "partial"
    edits = [c.edit.status for c in result.chunks]
    assert edits[0] == "completed" and set(edits[1:]) == {"failed"}
    assert all(c.review.status == "completed" for c in result.chunks)
    markdown = (out / "transcript.md").read_text()
    assert "> **Partial result.**" in markdown
    assert markdown.count("> **Notice:**") == len(result.chunks) - 1
    assert "not polished (agent call failed: error_max_budget_usd)" in markdown
    assert (out / "diagnostics" / "chunk-02-edit-error.txt").is_file()


async def test_no_edited_chunk_is_an_error_with_diagnostics(tmp_path: Path) -> None:
    def respond(prompt: str, options: ClaudeAgentOptions) -> Exception:
        return ClaudeAgentError("Claude rate limit reached")

    out = tmp_path / "polished"
    with pytest.raises(PolishFailedError, match="rate limit"):
        await polish(
            transcript_path=write_transcript(tmp_path / "raw.json", MEETING),
            output_directory=out,
            agent=ClaudeAgent(run_query=FakeQuery(respond)),
        )
    assert (out / "metadata.json").is_file() and (out / "transcript.md").is_file()


async def test_the_budget_is_reserved_before_calls_and_never_exceeded(
    tmp_path: Path,
) -> None:
    fake = FakeQuery(empty, cost=None)  # unknown cost: charged at each cap
    _, out, result = await _polish(tmp_path, fake, budget_usd=1.0)
    assert result.chunks[0].edit.status == "completed"
    assert result.chunks[0].review.status == "skipped"
    assert result.chunks[0].review.error == "not started: budget exhausted"
    assert result.estimated_cost_usd <= 1.0 and result.cost_is_estimate_only
    assert len(fake.calls) == 1
    assert (
        "not reviewed (not started: budget exhausted)"
        in (out / "transcript.md").read_text()
    )


async def test_a_slow_call_times_out_into_a_reported_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    monkeypatch.setattr(pipeline_module, "CALL_TIMEOUT_SECONDS", 0.2)

    class Slow(FakeQuery):
        def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> Any:
            inner = super().__call__(prompt=prompt, options=options)

            async def stream() -> Any:
                if role(options) == "ReviewFindings":
                    await anyio.sleep(5)
                async for message in inner:
                    yield message

            return stream()

    _, _, result = await _polish(tmp_path, Slow(empty))
    assert result.chunks[0].review.status == "failed"
    assert (result.chunks[0].review.error or "").startswith("timed out")
    assert result.status == "partial"


async def test_too_long_a_window_is_refused_with_the_way_out(tmp_path: Path) -> None:
    sentence = "we talk about the plan for the next quarter in detail."
    turns: Turns = [("speaker_1", " ".join([sentence] * 2500))]
    with pytest.raises(SourceError, match="start_seconds/end_seconds"):
        await _polish(tmp_path, FakeQuery(empty), turns=turns)
    assert not (tmp_path / "polished").exists()


# --- regressions from the independent review ------------------------------------------


@pytest.mark.parametrize(
    ("source", "span", "kind"),
    [
        # A filler is a closed set of set-off discourse markers, not any word.
        ("the committee approved the budget", "approved", "filler"),
        ("I like the plan", "like", "filler"),
        ("I mean it.", "I mean", "filler"),
        ("you know the answer", "you know", "filler"),
        # A restart must be said again, not followed by a different claim.
        ("the committee approved the budget, the board rejected it", "the committee approved the budget,", "restart"),
        ("we approved, we rejected it", "we approved,", "restart"),
        ("there's, for example, there seems to be appetite", "there's, for example,", "restart"),
        # A repeated number may be a year or dictated digits.
        ("it was in twenty twenty", "twenty", "repetition"),
        ("call four four four two", "four four", "repetition"),
        ("in twenty, twenty one", "twenty,", "restart"),
    ],
)
def test_meaning_bearing_deletions_are_refused(
    tmp_path: Path, source: str, span: str, kind: str
) -> None:
    words = _words([("speaker_1", source)], tmp_path)
    first = _id(words, span.split()[0])
    last = first + len(span.split()) - 1
    _, rejected, text = _decide(
        [("speaker_1", source)],
        tmp_path,
        deletions=[_span(first, last, span, kind)],
    )
    assert len(rejected) == 1, rejected
    assert text.casefold() == source.casefold()


@pytest.mark.parametrize(
    ("source", "span", "kind", "expected"),
    [
        ("it's, you know, fine", "you know,", "filler", "It's, fine"),
        ("I mean, we could go", "I mean,", "filler", "We could go"),
        ("I'm already pr produced a deck", "pr", "restart", "I'm already produced a deck"),
        ("I was go- I was going home", "I was go-", "restart", "I was going home"),
        ("so we, so we start now", "so we,", "restart", "So we start now"),
    ],
)
def test_set_off_markers_cut_off_fragments_and_echoes_go(
    tmp_path: Path, source: str, span: str, kind: str, expected: str
) -> None:
    words = _words([("speaker_1", source)], tmp_path)
    first = _id(words, span.split()[0])
    last = first + len(span.split()) - 1
    _, rejected, text = _decide(
        [("speaker_1", source)],
        tmp_path,
        deletions=[_span(first, last, span, kind)],
    )
    assert rejected == []
    assert text == expected


def test_glossary_cannot_replace_modality_or_pronouns(tmp_path: Path) -> None:
    turns: Turns = [("speaker_1", "we might send it to us today")]
    _, rejected, _ = _decide(
        turns,
        tmp_path,
        glossary=(GlossaryEntry(term="MIGHTY"), GlossaryEntry(term="USS")),
        glossary_decisions=[
            {"first_word": 1, "last_word": 1, "text": "might", "term": "MIGHTY", "reason": "x"},
            {"first_word": 5, "last_word": 5, "text": "us", "term": "USS", "reason": "x"},
        ],
    )
    assert rejected == [
        "would replace a modality word ('might')",
        "would replace a pronoun word ('us')",
    ]


@pytest.mark.parametrize(
    ("source", "display"),
    [
        ("US", "us"),
        ("US", "uS"),
        ("dfes", "DFES"),
        ("iPhone", "Iphone"),
        ("DFES.", "Dfes."),
    ],
)
def test_formats_never_recase_beyond_a_first_letter(
    tmp_path: Path, source: str, display: str
) -> None:
    turns: Turns = [("speaker_1", f"we said {source} so")]
    _, rejected, text = _decide(
        turns, tmp_path, formats=[{"word": 2, "display": display}]
    )
    assert len(rejected) == 1
    assert f" {source} " in f" {text} "


def test_sentence_starts_keep_internal_capitals(tmp_path: Path) -> None:
    _, _, text = _decide([("speaker_1", "iPhone sales. eBay too")], tmp_path)
    assert text == "iPhone sales. eBay too"


def test_model_edits_inside_a_flagged_passage_are_undone(tmp_path: Path) -> None:
    # Observed: the review flagged a passage as possibly not a plain stutter
    # but named no edit concern, so the deletion inside it stood.
    turns: Turns = [("speaker_1", "Um so this is red red hat, okay. Then we we go home")]
    words = _words(turns, tmp_path)
    state, chunk = _whole(words)
    red, we = _id(words, "red"), _id(words, "we")
    apply_edit_decisions(
        state,
        chunk,
        _decisions(
            [_span(red, red, "red", "repetition"), _span(we, we, "we", "repetition")],
            [{"word": _id(words, "okay."), "display": "okay!"}],
        ),
        (),
    )
    assert len([e for e in state.edits.values() if e.origin == "edit_model"]) == 3
    findings = ReviewFindings.model_validate(
        {
            "issues": [
                {"first_word": 1, "last_word": 7, "text": "so this is red red hat, okay.", "category": "unclear_meaning", "note": "May be a name, not a stutter."}
            ],
            "edit_concerns": [],
        }
    )
    items, rejected = apply_review(state, chunk, findings)
    assert len(items) == 1 and rejected == []
    restored = [e for e in state.edits.values() if e.status == "restored_after_review"]
    assert sorted(e.kind for e in restored) == ["format", "repetition"]
    assert all("May be a name, not a stutter." in (e.review_note or "") for e in restored)
    # Outside the passage, and the code's own filler removal, still apply.
    assert state.edits["f0"].status == "applied"
    text = " ".join(w.display for w in ledger_words(state) if w.display)
    assert text == "So this is red red hat, okay. Then we go home"
