from __future__ import annotations

from datetime import date

from jake_tools.ai_watch.obsidian import render_obsidian_note


def test_render_obsidian_note_has_callout() -> None:
    note = render_obsidian_note(
        title="Harness design",
        url="https://example.com",
        digest_summary="Concrete harness pattern.",
        body="Article body",
        curator_reason="Transferable harness design.",
        candidate_id="sha256:test",
        run_id="2026-07-02",
        tags=["agent-harnesses"],
        target_date=date(2026, 7, 2),
    )
    assert "> [!Summary] TL;DR:" in note
    assert "note/capture" in note
    assert "source/article/clipping" in note
    assert "Concrete harness pattern." in note
    assert 'Link: "https://example.com"' in note
