from __future__ import annotations

import json
from typing import Any, Protocol

from run_agent import handle_function_call

from .models import ExtractResult, SearchResult


class WebTools(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...

    def extract(
        self, urls: list[str], *, char_limit: int = 15000
    ) -> list[ExtractResult]: ...


class HermesWebTools:
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        raw = handle_function_call(
            "web_search",
            {"query": query, "limit": limit},
            enabled_toolsets=["web"],
        )
        return _parse_search_results(raw)

    def extract(
        self, urls: list[str], *, char_limit: int = 15000
    ) -> list[ExtractResult]:
        if not urls:
            return []
        raw = handle_function_call(
            "web_extract",
            {"urls": urls[:5], "char_limit": char_limit},
            enabled_toolsets=["web"],
        )
        return _parse_extract_results(raw)


class FakeWebTools:
    def __init__(
        self,
        *,
        search_results: dict[str, list[SearchResult]] | None = None,
        extract_results: dict[str, ExtractResult] | None = None,
    ) -> None:
        self.search_results = search_results or {}
        self.extract_results = extract_results or {}
        self.search_calls: list[tuple[str, int]] = []
        self.extract_calls: list[tuple[list[str], int]] = []

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        self.search_calls.append((query, limit))
        return self.search_results.get(query, [])[:limit]

    def extract(
        self, urls: list[str], *, char_limit: int = 15000
    ) -> list[ExtractResult]:
        self.extract_calls.append((urls, char_limit))
        results: list[ExtractResult] = []
        for url in urls:
            if url in self.extract_results:
                results.append(self.extract_results[url])
            else:
                results.append(
                    ExtractResult(
                        url=url,
                        title="",
                        content="",
                        status="fail",
                        error="not found in fake",
                    )
                )
        return results


def _parse_search_results(raw: str) -> list[SearchResult]:
    payload = json.loads(raw)
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        data = payload.get("data", {})
        if isinstance(data, dict):
            web = data.get("web", [])
            if isinstance(web, list):
                items = [item for item in web if isinstance(item, dict)]
    results: list[SearchResult] = []
    for item in items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        results.append(
            SearchResult(
                url=url,
                title=str(item.get("title", "")).strip(),
                description=str(item.get("description", "")).strip(),
                source="web_search",
            )
        )
    return results


def _parse_extract_results(raw: str) -> list[ExtractResult]:
    payload = json.loads(raw)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        results = payload.get("results", [])
        if isinstance(results, list):
            rows = [row for row in results if isinstance(row, dict)]
    parsed: list[ExtractResult] = []
    for row in rows:
        url = str(row.get("url", "")).strip()
        content = str(row.get("content", ""))
        error = row.get("error")
        status = "ok" if content and not error else "fail"
        parsed.append(
            ExtractResult(
                url=url,
                title=str(row.get("title", "")).strip(),
                content=content,
                full_text_path=(
                    str(row["full_text_path"]) if row.get("full_text_path") else None
                ),
                status=status,  # type: ignore[arg-type]
                error=str(error) if error else None,
            )
        )
    return parsed
