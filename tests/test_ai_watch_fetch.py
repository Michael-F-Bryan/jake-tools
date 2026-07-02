from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jake_tools.ai_watch.audit import append_model
from jake_tools.ai_watch.audit_models import DiscoveredRecord, FetchRecord, FetchStatus
from jake_tools.ai_watch.fetch import run_fetch
from jake_tools.ai_watch.models import AiWatchCommandOptions, ExtractResult
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.state import SeenIndex
from jake_tools.ai_watch.web_tools import FakeWebTools


def _discovered_record(
    candidate_id: str, url: str, *, title: str = "Article"
) -> DiscoveredRecord:
    return DiscoveredRecord(
        run_id="2026-07-02",
        candidate_id=candidate_id,
        timestamp="2026-07-02T00:00:00+00:00",
        source="test",
        url=url,
        title=title,
    )


def test_fetch_honours_max_candidates_cap(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    urls = [
        "https://example.com/one",
        "https://example.com/two",
        "https://example.com/three",
    ]
    for index, url in enumerate(urls, start=1):
        append_model(
            paths.candidates,
            _discovered_record(f"sha256:{index}", url, title=f"Article {index}"),
        )
    fake = FakeWebTools(
        extract_results={
            url: ExtractResult(
                url=url, title=f"Article {index}", content=f"Body {index}"
            )
            for index, url in enumerate(urls, start=1)
        }
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        max_candidates=2,
    )
    result = run_fetch(options=options, paths=paths, state=state, web_tools=fake)
    assert result.fetched == 2
    assert len(fake.extract_calls) == 2


def test_fetch_max_candidates_ignores_skipped_archives(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    archived_url = "https://example.com/archived"
    state.record_seen(
        candidate_id="sha256:archived",
        url=archived_url,
        title="Archived",
        source="test",
        content_hash="sha256:abc",
        content_path="articles/sha256:archived.md",
    )
    append_model(
        paths.candidates,
        _discovered_record("sha256:archived", archived_url, title="Archived"),
    )
    append_model(
        paths.candidates,
        _discovered_record("sha256:new", "https://example.com/new", title="New"),
    )
    fake = FakeWebTools(
        extract_results={
            "https://example.com/new": ExtractResult(
                url="https://example.com/new",
                title="New",
                content="Fresh article body.",
            )
        }
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        max_candidates=1,
    )
    result = run_fetch(options=options, paths=paths, state=state, web_tools=fake)
    assert result.skipped == 1
    assert result.fetched == 1
    assert len(fake.extract_calls) == 1


def test_fetch_archives_metadata_with_content_hash(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp="2026-07-02T00:00:00+00:00",
            source="anthropic_engineering",
            url="https://www.anthropic.com/engineering/harness-design-long-running-apps",
            title="Harness design",
        ),
    )
    fake = FakeWebTools(
        extract_results={
            "https://www.anthropic.com/engineering/harness-design-long-running-apps": ExtractResult(
                url="https://www.anthropic.com/engineering/harness-design-long-running-apps",
                title="Harness design",
                content="Harness design for long-running application development with evaluator loops.",
            )
        }
    )
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    result = run_fetch(options=options, paths=paths, state=state, web_tools=fake)
    assert result.fetched == 1
    metadata = json.loads(
        paths.article_metadata("sha256:test").read_text(encoding="utf-8")
    )
    assert metadata["content_hash"].startswith("sha256:")


def test_fetch_records_truncated_extract_full_text_path(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:trunc",
            timestamp="2026-07-02T00:00:00+00:00",
            source="test",
            url="https://example.com/long",
            title="Long article",
        ),
    )
    fake = FakeWebTools(
        extract_results={
            "https://example.com/long": ExtractResult(
                url="https://example.com/long",
                title="Long article",
                content="Truncated head and tail preview.",
                full_text_path="/tmp/hermes/full-long-article.md",
            )
        }
    )
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    run_fetch(options=options, paths=paths, state=state, web_tools=fake)
    metadata = json.loads(
        paths.article_metadata("sha256:trunc").read_text(encoding="utf-8")
    )
    assert metadata["full_text_path"] == "/tmp/hermes/full-long-article.md"
    fetch_rows = [
        row
        for row in paths.fetch_results.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    assert fetch_rows
    record = FetchRecord.model_validate_json(fetch_rows[0])
    assert record.status == FetchStatus.OK
