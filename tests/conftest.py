"""Repo-wide pytest configuration.

Registers the ``corpus`` marker (Phase 2 canary runs, see
``test_transcript_bundle_corpus.py``) and keeps it deselected by
default -- without touching ``pyproject.toml``'s ``[tool.pytest.
ini_options]`` (forbidden to this slice; see CONTRACTS.md Phase 2
scope fence). ``addopts`` there is ``-m "not live"``, which only ever
excludes ``live``-marked tests; a marker with no matching ``addopts``
entry runs by default unless something else opts it out, which is
exactly what the hook below does for ``corpus``.

Unlike ``live`` (tests hitting real external services), ``corpus``
tests are local and deterministic but depend on the evaluation corpus
living at a fixed path under ``_working/`` (gitignored, not guaranteed
present in every environment) and write real evidence files under
``_working/.../phase2-canaries/`` -- opt-in via ``pytest -m corpus``,
matching the Phase 2 brief's "deselected by default".
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "corpus: Phase 2 evaluation-corpus canary runs (opt-in: pytest -m corpus).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    markexpr = config.getoption("markexpr", default="") or ""
    if "corpus" in markexpr:
        # The operator explicitly asked for corpus tests (e.g. `-m corpus`
        # or `-m "corpus and teams"`) -- pytest's own `-m` filtering
        # already handles selecting/deselecting from here.
        return
    skip_corpus = pytest.mark.skip(
        reason="corpus canary tests are opt-in: run with `pytest -m corpus`"
    )
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip_corpus)
