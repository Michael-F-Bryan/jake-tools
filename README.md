# Jake's Tools

Internal tools used by Jake.

## Installation

```bash
uv tool add -e .
```

## Transcribing Obsidian recordings

```bash
jake-tools transcribe obsidian-recording NOTE.md
jake-tools transcribe obsidian-recording --mode transcript NOTE.md
jake-tools transcribe obsidian-recording --mode chaptered-transcript NOTE.md
jake-tools transcribe obsidian-recording --dry-run --json NOTE.md
```

This command expects local `ffmpeg` and `scribe` executables to be available.
