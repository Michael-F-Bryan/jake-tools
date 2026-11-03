"""Token-level facts the edit guards rely on.

Deliberately small and categorical: what makes a word *meaning-bearing* for
the purpose of refusing an unrequested deletion (negation, quantity,
modality/hedging, condition/scope, person reference, names), not a list of
corrections or exceptions. A protected word is never forbidden outright; a
restart may drop it only when the speaker repeats it straight away, and a
repetition only when an identical copy stays.
"""

from __future__ import annotations

import re

DETERMINISTIC_FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm", "erm"})
"""Removed in code wherever they stand alone. Not acknowledgements
(``mm-hmm``, ``uh-huh``) and not ``er``/``ah``, which recognisers also emit
for real words."""

NEGATION = frozenset(
    [
        "no",
        "not",
        "never",
        "nor",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        "cannot",
        "without",
    ]
)
NUMBER_WORDS = frozenset(
    [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "half",
        "quarter",
        "dozen",
        "once",
        "twice",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    ]
)
MODALITY = frozenset(
    [
        "may",
        "might",
        "must",
        "should",
        "could",
        "would",
        "will",
        "shall",
        "can",
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "likely",
        "unlikely",
        "think",
        "guess",
        "suppose",
        "reckon",
        "believe",
        "doubt",
        "sure",
        "unsure",
        "apparently",
        "sort",
        "kind",
    ]
)
SCOPE = frozenset(
    [
        "if",
        "unless",
        "except",
        "only",
        "until",
        "before",
        "after",
        "all",
        "every",
        "always",
        "some",
        "any",
        "more",
        "most",
        "less",
        "least",
        "few",
        "many",
        "much",
        "very",
        "really",
    ]
)
PRONOUNS = frozenset(
    [
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
    ]
)
EMPHASIS = NEGATION | frozenset(
    ["very", "really", "so", "too", "all", "every", "always"]
)
"""A single word in here, said twice, may be emphasis ("no, no"), not a stutter."""

_TOKEN = re.compile(r"^(?P<lead>\W*)(?P<core>.*?)(?P<trail>\W*)$", re.DOTALL)
_CONTRACTION = re.compile(r"(?:n't|n’t)$")
_MODAL_CONTRACTION = re.compile(r"(?:'ll|'d|’ll|’d)$")
_SENTENCE_END = frozenset(".?!")


def split_token(text: str) -> tuple[str, str, str]:
    """``(leading punctuation, core, trailing punctuation)``."""
    match = _TOKEN.match(text)
    assert match is not None  # the pattern matches every string
    return match["lead"], match["core"], match["trail"]


def lexical(text: str) -> str:
    """The comparable form of a token: its core, case-folded, ``’`` as ``'``."""
    return split_token(text)[1].casefold().replace("’", "'")


def ends_sentence(text: str) -> bool:
    return bool(set(split_token(text)[2]) & _SENTENCE_END)


def is_filler(text: str) -> bool:
    return lexical(text) in DETERMINISTIC_FILLERS


def is_number(token: str) -> bool:
    return any(c.isdigit() for c in token) or token in NUMBER_WORDS


def is_negation(token: str) -> bool:
    return token in NEGATION or bool(_CONTRACTION.search(token))


def protected_kind(token: str, *, name: bool) -> str | None:
    """Why ``token`` (already :func:`lexical`) carries meaning, if it does.

    ``name`` is whether the source word looks like a name: capitalised where
    a sentence does not start (see :func:`looks_like_name`).
    """
    if not token:
        return None
    if is_negation(token):
        return "negation"
    if is_number(token):
        return "number"
    if name:
        return "name"
    if token in MODALITY or _MODAL_CONTRACTION.search(token):
        return "modality"
    if token in SCOPE:
        return "scope"
    if token in PRONOUNS or token.split("'")[0] in PRONOUNS:
        return "pronoun"
    return None


def looks_like_name(text: str, *, sentence_initial: bool) -> bool:
    """Capitalised mid-sentence, other than the pronoun ``I`` and its forms.

    The recogniser capitalises names and acronyms; a sentence's first word
    is capitalised anyway, so it tells us nothing.
    """
    core = split_token(text)[1]
    if not core or not core[0].isupper() or sentence_initial:
        return False
    return core.split("'")[0].split("’")[0] != "I"
