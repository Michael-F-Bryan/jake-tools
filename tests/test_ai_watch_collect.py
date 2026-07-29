from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_watch.collect import run_collect
from jake_tools.ai_watch.models import AiWatchCommandOptions, SearchResult
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.sources import DEFAULT_SOURCE_QUERIES, SourceQuery
from jake_tools.ai_watch.state import SeenIndex
from jake_tools.ai_watch.web_tools import FakeWebTools

_QUERY = SourceQuery("test", "site:example.com agent harness", 5)


def test_collect_skips_seen_url_twice(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    state.record_seen(
        candidate_id="sha256:seen",
        url="https://example.com/post",
        title="Post",
        source="test",
    )
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    fake = FakeWebTools(
        search_results={
            _QUERY.query: [
                SearchResult(
                    url="https://example.com/post",
                    title="Post",
                    description="seen",
                    source="test",
                )
            ]
        }
    )
    result = run_collect(
        options=options,
        paths=paths,
        state=state,
        web_tools=fake,
        source_queries=(_QUERY,),
    )
    assert result.discovered == 0
    assert result.skipped_seen == 1


def test_collect_discovers_new_candidates(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    fake = FakeWebTools(
        search_results={
            _QUERY.query: [
                SearchResult(
                    url="https://www.anthropic.com/engineering/harness-design-long-running-apps",
                    title="Harness design",
                    description="harness",
                    source="anthropic_engineering",
                )
            ]
        }
    )
    result = run_collect(
        options=options,
        paths=paths,
        state=state,
        web_tools=fake,
        source_queries=(_QUERY,),
    )
    assert result.discovered == 1
    assert paths.candidates.exists()


def test_collect_flushes_seen_index_to_disk(tmp_path: Path) -> None:
    """A standalone `collect` CLI invocation is a separate process from the
    next `fetch` invocation, so newly-seen URLs must be durable on disk by
    the time run_collect returns, not just held in memory."""
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    fake = FakeWebTools(
        search_results={
            _QUERY.query: [
                SearchResult(
                    url="https://example.com/new-article",
                    title="New article",
                    description="fresh",
                    source="test",
                )
            ]
        }
    )
    run_collect(
        options=options,
        paths=paths,
        state=state,
        web_tools=fake,
        source_queries=(_QUERY,),
    )

    assert state.url_index_path.exists()
    reloaded = SeenIndex(paths.state_root)
    assert reloaded.check_seen(url="https://example.com/new-article") is not None


def test_default_queries_cover_anthropic_sdk_and_builder_sources() -> None:
    queries = {query.source_id: query.query for query in DEFAULT_SOURCE_QUERIES}
    all_queries = "\n".join(query.query for query in DEFAULT_SOURCE_QUERIES)

    assert "anthropic_engineering" in queries
    assert "anthropic_news" in queries
    assert "anthropic_cookbook" in queries
    assert "anthropic_sdk_python" in queries
    assert "anthropic_sdk_typescript" in queries
    assert "claude_code_sdk" in queries
    assert "Claude SDK" in all_queries
