from __future__ import annotations

import json

from jake_tools.ai_watch.web_tools import (
    FakeWebTools,
    _parse_extract_results,
    _parse_search_results,
)


def test_parse_search_results() -> None:
    raw = json.dumps(
        {
            "success": True,
            "data": {
                "web": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "description": "desc",
                    }
                ]
            },
        }
    )
    results = _parse_search_results(raw)
    assert len(results) == 1
    assert results[0].url == "https://example.com"


def test_parse_extract_results() -> None:
    raw = json.dumps(
        {
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "content": "Body text",
                }
            ]
        }
    )
    results = _parse_extract_results(raw)
    assert results[0].status == "ok"
    assert "Body" in results[0].content


def test_fake_web_tools_records_calls() -> None:
    fake = FakeWebTools()
    fake.search("query", limit=2)
    assert fake.search_calls == [("query", 2)]
