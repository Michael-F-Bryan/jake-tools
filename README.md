# Jake's Tools

Internal tools used by Jake.

## Installation

```bash
uv tool add -e .
```

## Transcribing Obsidian recordings

```bash
jake-tools transcribe obsidian-recording NOTE.md
jake-tools transcribe obsidian-recording --dry-run --json NOTE.md
```

The Obsidian recording pipeline always writes the same shape:

- `## Meeting Notes` with high-level dot points
- `## Chapters` with timestamps
- `## Transcript` with polished transcript text grouped by chapter

This command expects local `ffmpeg` and `scribe` executables to be available.
