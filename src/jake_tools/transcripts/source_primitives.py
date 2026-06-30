from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from .models import SourceArtifact
from .obsidian import load_source_note

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class SourcePrimitiveError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def source_from_obsidian_note(note_path: Path) -> SourceArtifact:
    source_note = load_source_note(note_path)
    artifact = source_note.to_source_artifact()
    artifact.source_path = source_note.path.resolve()
    artifact.attachments = [
        recording.resolved_path.resolve() for recording in source_note.recordings
    ]
    return artifact


def source_from_gemini_text(text_path: Path) -> SourceArtifact:
    resolved = text_path.resolve()
    return SourceArtifact(
        kind="gemini-text",
        source_path=resolved,
        title=resolved.stem,
        raw_text_path=resolved,
    )


def _pdf_extraction_warnings(stderr: str) -> list[str]:
    return [line.strip() for line in stderr.splitlines() if line.strip()]


def source_from_gemini_pdf(
    pdf_path: Path,
    *,
    source_output_path: Path,
    run_command: CommandRunner = _run,
) -> SourceArtifact:
    resolved_pdf = pdf_path.resolve()
    raw_text_path = source_output_path.resolve().with_suffix(".raw.txt")
    try:
        result = run_command(
            [
                "pdftotext",
                "-layout",
                str(resolved_pdf),
                str(raw_text_path),
            ]
        )
    except FileNotFoundError as exc:
        raise SourcePrimitiveError(
            "pdftotext is required for `transcript source gemini-pdf` but was not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise SourcePrimitiveError(f"pdftotext failed: {message}") from exc

    if not raw_text_path.exists():
        raise SourcePrimitiveError(
            f"pdftotext completed but did not create extracted text at {raw_text_path}"
        )

    return SourceArtifact(
        kind="gemini-pdf",
        source_path=resolved_pdf,
        title=resolved_pdf.stem,
        raw_text_path=raw_text_path,
        warnings=_pdf_extraction_warnings(result.stderr),
    )
