from __future__ import annotations

from pathlib import Path

from .audit_models import ArticleMetadata
from .models import ExtractResult, content_hash_for
from .records import write_model


def archive_extract(
    *,
    paths_articles: Path,
    candidate_id: str,
    extracted: ExtractResult,
    source: str,
) -> tuple[Path, Path, str]:
    markdown_path = paths_articles / f"{candidate_id}.md"
    metadata_path = paths_articles / f"{candidate_id}.metadata.json"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(extracted.content, encoding="utf-8")
    content_hash = content_hash_for(extracted.content)
    metadata = ArticleMetadata(
        candidate_id=candidate_id,
        url=extracted.url,
        title=extracted.title,
        source=source,
        content_hash=content_hash,
        full_text_path=extracted.full_text_path,
        status=extracted.status,
        error=extracted.error,
    )
    write_model(metadata_path, metadata)
    return markdown_path, metadata_path, content_hash


def load_article_metadata(path: Path) -> ArticleMetadata:
    from .records import load_model

    return load_model(path, ArticleMetadata)
