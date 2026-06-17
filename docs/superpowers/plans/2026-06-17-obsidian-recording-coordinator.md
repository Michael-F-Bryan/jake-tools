# Obsidian Recording Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `jake-tools transcribe obsidian-recording` as a Python coordinator harness that runs Hermes under the hood to execute the `obsidian-recording-to-meeting-minutes` workflow against a local Obsidian note.

**Architecture:** Keep the Click CLI thin, move orchestration into a dedicated coordinator module, and separate deterministic local work from Hermes-driven reasoning. Deterministic steps such as note parsing, recording discovery, concatenation, temporary artifact-path management, and final verification should live in plain Python. Hermes should be used as the bounded reasoning engine for stage outputs that benefit from agentic judgement: speaker mapping, chaptering, polishing, minutes extraction, and final signoff. Use the existing `Paths.temp()` context manager for the run scratch space and let the working directory disappear once the note is safely updated.

**Tech Stack:** Python 3.14, Click, Pydantic, Hermes Agent (`run_agent.AIAgent`), Jinja2, local filesystem tools (`ffmpeg`/`ffprobe`, `scribe`, `obsidian` CLI), pytest.

---

## Current repo state

**Existing entrypoints and stubs**
- `src/jake_tools/cli/transcribe.py:32-39` already exposes `jake-tools transcribe obsidian-recording OBSIDIAN_NOTE` and forwards to `process_obsidian_recording()`.
- `src/jake_tools/transcripts/__init__.py:8-13` is a `NotImplementedError` stub.
- `src/jake_tools/hermes.py` now exposes `Hermes.oneshot_structured()`, which should be the default path for structured stage outputs.
- `src/jake_tools/hermes.py` now creates `AIAgent(..., quiet_mode=True)`, so coordinator runs will stay quiet by default.
- `src/jake_tools/transcripts/paths.py:10-21` is now a context-managed temporary artifact root. That matches the desired lifecycle for this command.
- `src/jake_tools/transcripts/polish.py:5-26` is a useful precedent for Jinja-based prompts and Hermes invocation, but it is too thin for a multi-stage pipeline.

**Gaps to fill**
- no coordinator runtime;
- no structured stage/artifact models;
- no local note / recording ingestion;
- no Hermes prompt library for specialist stages;
- no verification layer for note preservation and stage handoffs;
- no tests at all;
- no documented tool prerequisites or failure modes.

## Proposed file structure

### Modify
- `pyproject.toml`
  - add test dependencies and any runtime helpers we decide to use.
- `README.md`
  - document the new command, prerequisites, and output layout.
- `src/jake_tools/transcripts/__init__.py`
  - replace the stub with a thin public orchestration entrypoint.
- `src/jake_tools/transcripts/paths.py`
  - extend the temporary artifact-root helper with the hard-coded paths the workflow needs.
- `src/jake_tools/hermes.py`
  - use `oneshot_structured()` for typed stage boundaries, and only extend the wrapper if implementation reveals a real gap.
- `src/jake_tools/cli/transcribe.py`
  - add options for mode, output root, keep-working-dir, quiet/json output, and dry-run validation.

### Create
- `src/jake_tools/transcripts/coordinator.py`
  - main coordinator class and top-level workflow.
- `src/jake_tools/transcripts/models.py`
  - Pydantic models for provenance, chapters, ledgers, reports, and run state.
- `src/jake_tools/transcripts/obsidian.py`
  - parse note content, extract embed links, resolve vault-relative attachment paths.
- `src/jake_tools/transcripts/audio.py`
  - recording ordering, concatenation, `scribe` execution, provenance capture.
- `src/jake_tools/transcripts/prompts.py`
  - Jinja templates for stage-specific Hermes prompts.
- `src/jake_tools/transcripts/stages.py`
  - Python wrappers around individual Hermes reasoning stages.
- `src/jake_tools/transcripts/merge.py`
  - deterministic note-update logic for `## Meeting Minutes`, `## Chapters`, and `## Transcript`.
- `src/jake_tools/transcripts/verify.py`
  - fidelity gates and final signoff checks.
- `tests/test_transcribe_obsidian_recording_cli.py`
  - CLI coverage.
- `tests/test_transcript_paths.py`
  - artifact-root and keep/delete semantics.
- `tests/test_obsidian_resolution.py`
  - note parsing and recording resolution.
- `tests/test_audio_pipeline.py`
  - concatenation / `scribe` invocation plumbing.
- `tests/test_merge.py`
  - note merge preservation checks.
- `tests/test_verify.py`
  - fidelity gate logic.
- `tests/fixtures/obsidian/`
  - sample notes, attachments, transcript JSON, expected merged-note outputs.

## Design decisions

### 1. Coordinator owns sequencing; Hermes does not self-route the whole pipeline
Do **not** hand the whole skill body to Hermes and hope it drives itself correctly. The Python coordinator should own:
- stage order;
- retry budgets;
- artifact paths;
- pass/fail gating;
- whether to continue or stop.

Hermes should be called for bounded subproblems, with explicit input artifacts and explicit required JSON-shaped outputs.

### 2. Deterministic local steps stay in Python
These should not go through the model:
- reading the source note;
- resolving recording embeds;
- sorting recordings by creation time;
- concatenating recordings;
- running `scribe`;
- writing artifacts to disk;
- verifying headings, counts, anchors, and original-content preservation.

### 3. Hermes stage prompts should be specialist-specific
Model prompts should mirror the skill’s specialist roster, but the Python layer should expose each one as a dedicated function. Example stage functions:
- `run_speaker_mapping_stage()`
- `run_chaptering_stage()`
- `run_chapter_polish_stage()`
- `run_fidelity_audit_stage()`
- `run_proper_noun_correction_stage()`
- `run_final_signoff_stage()`

### 4. All stage boundaries should use typed models + JSON artifacts
Use Pydantic models that map closely to the skill’s schemas:
- `ProvenanceReport`
- `ScribeRunReport`
- `SpeakerMapping`
- `Chapter`
- `ChapterPolishLedger`
- `TurnManifest`
- `FidelityReport`
- `CorrectionLedgerEntry`
- `MergeReport`
- `SignoffReport`

Treat model output as untrusted until parsed and revalidated. Prefer `Hermes.oneshot_structured()` over hand-rolled JSON-repair loops unless a stage genuinely needs multi-turn agent behaviour.

### 5. Working directory should be temporary and implementation-local
Use the existing `Paths.temp()` context manager and hard-coded artifact names such as:
- `merged.mp3`
- `merged.json`
- `chapters.json`
- `speaker-mapping.json`

The command does not need to preserve artefacts after the note is written successfully, so avoid slug logic and durable working-directory policy unless a later requirement proves it necessary.

### 6. Start with the transcript + chaptered-transcript path, then add full minutes merge
Implement in layers:
1. robust ingestion + transcript artifacting;
2. chaptering and chaptered transcript rendering;
3. full meeting-minutes insertion and fidelity pipeline;
4. proper-noun review and final signoff.

That gives a working command earlier and keeps the hard parts isolated.

---

## Task 1: Harden the project scaffold for a real pipeline

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/jake_tools/hermes.py`
- Create: `tests/test_hermes_wrapper.py`

- [ ] **Step 1: Add test tooling and any missing runtime libraries**

Update `pyproject.toml` so the dev group includes pytest. Keep it minimal.

```toml
[dependency-groups]
dev = [
    "pyright>=1.1.410",
    "pytest>=8.4.0",
]
```

If later tasks need a helper like `python-slugify`, add it then; don’t pre-load libraries just in case.

- [ ] **Step 2: Verify the existing structured Hermes helper is sufficient**

The repo now has `Hermes.oneshot_structured()`, so do not introduce a broader runtime wrapper unless a concrete stage requires it. The likely work here is small fixes only:
- make sure `oneshot_structured()` really parses model output the way stage models expect;
- make sure it fails clearly on empty responses;
- remove any leftover debug `print()` calls elsewhere in the transcript pipeline.

- [ ] **Step 3: Add a parsing test for `HermesResult` and wrapper behaviour**

Write a small unit test that validates `HermesResult.model_validate()` still accepts the dict shape returned by Hermes, and that `oneshot_structured()` can parse a structured payload without a follow-up repair turn when the response is already valid.

```python
def test_hermes_result_aliases_final_response() -> None:
    result = HermesResult.model_validate({"final_response": "ok", "completed": True})
    assert result.response == "ok"
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_hermes_wrapper.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/jake_tools/hermes.py tests/test_hermes_wrapper.py
git commit -m "test: cover Hermes structured helper"
```

## Task 2: Extend the temporary path helper with workflow-specific artifact names

**Files:**
- Modify: `src/jake_tools/transcripts/paths.py`
- Create: `tests/test_transcript_paths.py`

- [ ] **Step 1: Add the hard-coded artifact paths the workflow needs**

Keep `Paths.temp()` as-is conceptually. The main work is to add explicit path properties for the files the coordinator writes during a run.

Target shape:

```python
class Paths(BaseModel):
    root: Path

    @property
    def merged(self) -> Path: ...

    @property
    def transcript(self) -> Path: ...

    @property
    def chapters(self) -> Path: ...

    @property
    def speaker_mapping(self) -> Path: ...

    @property
    def merge_report(self) -> Path: ...
```

Keep names fixed and boring.

- [ ] **Step 2: Test path creation and context-manager semantics**

Write tests that assert:
- the root exists after creation;
- artifact paths live under the root;
- the context manager cleans up after exit.

- [ ] **Step 4: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_transcript_paths.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/jake_tools/transcripts/paths.py tests/test_transcript_paths.py
git commit -m "feat: add transcript artifact temp paths"
```

## Task 3: Implement note parsing and recording resolution

**Files:**
- Create: `src/jake_tools/transcripts/obsidian.py`
- Create: `src/jake_tools/transcripts/models.py`
- Create: `tests/test_obsidian_resolution.py`

- [ ] **Step 1: Define typed ingestion models**

Add Pydantic models for the deterministic ingestion surface.

```python
class RecordingRef(BaseModel):
    raw_link: str
    resolved_path: Path
    created_at: datetime

class SourceNote(BaseModel):
    path: Path
    body: str
    recordings: list[RecordingRef]
```

- [ ] **Step 2: Parse Obsidian embeds from note content**

Support embeds like `![[file.m4a]]` and markdown links when present.

```python
EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
```

Resolve attachment paths relative to the note first, then the vault-level `Attachments/` directory if needed.

- [ ] **Step 3: Capture recording creation times for ordering**

Use `path.stat()` and store a timestamp used for oldest→newest ordering.

```python
def created_at(path: Path) -> datetime:
    stat = path.stat()
    return datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime, tz=timezone.utc)
```

- [ ] **Step 4: Test embed parsing and path resolution**

Use fixture notes plus fake attachment files to verify:
- one recording resolves correctly;
- multiple recordings resolve correctly;
- recordings sort by creation time;
- a missing recording raises a clear exception with the note path in the message.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_obsidian_resolution.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/models.py src/jake_tools/transcripts/obsidian.py tests/test_obsidian_resolution.py
 git commit -m "feat: resolve Obsidian recording embeds"
```

## Task 4: Build the deterministic audio ingestion pipeline

**Files:**
- Create: `src/jake_tools/transcripts/audio.py`
- Create: `tests/test_audio_pipeline.py`

- [ ] **Step 1: Implement ordered concatenation planning**

Create a function that takes resolved recordings and returns a concat plan matching the skill’s ingestion contract.

```python
class ConcatPlan(BaseModel):
    inputs_in_creation_order: list[Path]
    output_merged_audio: Path
```

- [ ] **Step 2: Implement `ffmpeg` concatenation via a generated concat file**

Use a temporary concat manifest under the run root and call `ffmpeg` with explicit arguments.

```python
def concatenate_recordings(plan: ConcatPlan) -> None:
    ...
```

Command shape:

```bash
ffmpeg -f concat -safe 0 -i inputs.txt -c copy merged.m4a
```

If stream-copy fails on mixed codecs, decide whether to fall back to re-encoding in this task or defer that fallback to a later task. Document the choice.

- [ ] **Step 3: Implement `scribe` JSON execution and provenance capture**

Wrap `scribe` so it always writes JSON to the expected artifact path.

Command shape:

```bash
scribe -o /path/to/merged.json --format json "/path/to/merged.m4a"
```

Capture warnings/stdout/stderr into a `ScribeRunReport`.

- [ ] **Step 4: Test command construction and failure handling**

Mock subprocess execution so tests verify:
- correct input ordering;
- quoted path handling via subprocess argument lists, not shell strings;
- `scribe` always uses `--format json` and `-o`;
- subprocess failures raise stage-specific exceptions.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_audio_pipeline.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/audio.py tests/test_audio_pipeline.py
git commit -m "feat: add recording concat and scribe ingestion"
```

## Task 5: Add typed stage schemas and Hermes prompt templates

**Files:**
- Create: `src/jake_tools/transcripts/prompts.py`
- Modify: `src/jake_tools/transcripts/models.py`
- Create: `tests/test_stage_parsing.py`

- [ ] **Step 1: Add models for the Hermes-driven stages**

Mirror the skill schemas closely.

```python
class SpeakerIdentity(BaseModel):
    name: str
    confidence: float
    reason: str

class SpeakerMapping(BaseModel):
    mapping: dict[str, SpeakerIdentity]
    unresolved: list[str] = Field(default_factory=list)
    notes: str = ""

class Chapter(BaseModel):
    title: str
    start: float
    end: float
    summary: str
```

Do the same for chapter polish ledgers, manifests, fidelity reports, corrections, merge reports, and signoff reports.

- [ ] **Step 2: Create stage-specific prompt renderers**

Do not store giant inline f-strings in the coordinator. Use Jinja templates with explicit response contracts.

Example shape:

```python
SPEAKER_MAPPING_PROMPT = Template("""
You are the speaker-mapping specialist for the obsidian-recording-to-meeting-minutes workflow.

Input transcript JSON:
```json
{{ transcript_json }}
```

Return valid JSON matching this schema:
{{ schema_json }}
""")
```

Each stage prompt should include:
- the specialist role;
- the relevant non-negotiables from the skill;
- the exact JSON schema it must satisfy;
- a warning that unparseable output is treated as stage failure.

- [ ] **Step 3: Add tests that parse representative model outputs into Pydantic models**

Use canned JSON strings to ensure the models are a good fit before wiring Hermes into the coordinator.

- [ ] **Step 4: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_stage_parsing.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/jake_tools/transcripts/models.py src/jake_tools/transcripts/prompts.py tests/test_stage_parsing.py
git commit -m "feat: add transcript stage schemas and prompts"
```

## Task 6: Implement a bounded stage runner around Hermes

**Files:**
- Create: `src/jake_tools/transcripts/stages.py`
- Modify: `src/jake_tools/hermes.py`
- Create: `tests/test_stage_runner.py`

- [ ] **Step 1: Add a generic JSON stage runner**

The coordinator needs one place that:
- renders the prompt;
- runs Hermes;
- extracts the response;
- parses it into a target model;
- raises a stage-specific exception on failure.

Target shape:

```python
T = TypeVar("T", bound=BaseModel)

def run_json_stage(
    hermes: Hermes,
    *,
    prompt: str,
    model_type: type[T],
    max_iterations: int = 8,
    enabled_toolsets: list[str] | None = None,
) -> T:
    ...
```

- [ ] **Step 2: Add explicit stage wrappers**

Wrap the generic runner with named functions:

```python
def run_speaker_mapping_stage(...)-> SpeakerMapping: ...
def run_chaptering_stage(...)-> list[Chapter]: ...
def run_fidelity_audit_stage(...)-> FidelityReport: ...
```

That keeps the coordinator legible.

- [ ] **Step 3: Decide the toolset policy for each stage**

Default to the narrowest toolset that makes sense. Most stages should probably run with no external tools and rely only on the provided artifacts. If a stage needs vault search or web evidence later, make that explicit at the wrapper boundary.

- [ ] **Step 4: Test parse failure and empty-response handling**

Unit tests should confirm that:
- empty Hermes responses fail fast;
- invalid JSON fails with a stage-specific error;
- valid JSON parses into the expected model.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_stage_runner.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/stages.py src/jake_tools/hermes.py tests/test_stage_runner.py
git commit -m "feat: add bounded Hermes stage runner"
```

## Task 7: Implement the Python coordinator skeleton and transcript-only flow

**Files:**
- Create: `src/jake_tools/transcripts/coordinator.py`
- Modify: `src/jake_tools/transcripts/__init__.py`
- Create: `tests/test_coordinator_transcript_only.py`

- [ ] **Step 1: Define the coordinator state object**

Use a Pydantic model or dataclass to hold the run state.

```python
class CoordinatorState(BaseModel):
    source_note: Path
    paths: TranscriptPaths
    provenance: ProvenanceReport | None = None
    scribe_run: ScribeRunReport | None = None
    speaker_map: SpeakerMapping | None = None
    chapters: list[Chapter] = Field(default_factory=list)
```

- [ ] **Step 2: Implement deterministic transcript-only flow first**

The first working path should do:
1. read note;
2. resolve recordings;
3. concatenate if needed;
4. run `scribe`;
5. render markdown transcript;
6. merge `## Transcript` into the source note;
7. verify the note on disk.

This gives you an end-to-end vertical slice before the full coordinator logic lands.

- [ ] **Step 3: Replace the public stub with the coordinator entrypoint**

In `src/jake_tools/transcripts/__init__.py`:

```python
def process_obsidian_recording(hermes: Hermes, obsidian_note: Path) -> None:
    coordinator = ObsidianRecordingCoordinator(hermes=hermes, source_note=obsidian_note)
    coordinator.run()
```

- [ ] **Step 4: Test the transcript-only happy path**

The test should create a fake note, fake artifact outputs, stub `scribe`, and assert:
- the note now contains one `## Transcript` section;
- the recording embed still exists;
- the raw JSON path is recorded;
- the updated note is written back to disk.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_coordinator_transcript_only.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/__init__.py src/jake_tools/transcripts/coordinator.py tests/test_coordinator_transcript_only.py
 git commit -m "feat: add obsidian recording coordinator skeleton"
```

## Task 8: Implement chaptering and chaptered transcript rendering

**Files:**
- Modify: `src/jake_tools/transcripts/coordinator.py`
- Create: `src/jake_tools/transcripts/merge.py`
- Create: `tests/test_merge.py`

- [ ] **Step 1: Add chaptering stage invocation after successful `scribe`**

Feed the transcript JSON into `run_chaptering_stage()` and persist `chapters.json`.

- [ ] **Step 2: Split transcript turns into chapter ranges deterministically**

This should be pure Python. The model decides chapter ranges, but Python assigns turns to chapters.

- [ ] **Step 3: Render `## Chapters` and chapter headings in `## Transcript`**

The merge logic should preserve the original note body and embed while inserting generated sections.

Target note shape:

```md
<original content>

## Chapters
- [00:00 — Kickoff](#...)

## Transcript
### 00:00 — Kickoff
Speaker: turn text
```

- [ ] **Step 4: Test note preservation and heading alignment**

Assert that:
- original content still exists;
- `## Chapters` count matches transcript chapter heading count;
- headings match the chapter index labels exactly.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_merge.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/coordinator.py src/jake_tools/transcripts/merge.py tests/test_merge.py
git commit -m "feat: add chaptered transcript merge"
```

## Task 9: Implement fidelity gates in Python

**Files:**
- Create: `src/jake_tools/transcripts/verify.py`
- Create: `tests/test_verify.py`
- Modify: `src/jake_tools/transcripts/coordinator.py`

- [ ] **Step 1: Encode the binary gates from the skill as Python checks**

At minimum implement Gates A, B, C, E, and G in Python before reporting success.

Example shapes:

```python
def check_chapter_coverage(chapters: list[Chapter], transcript_end: float) -> GateResult: ...
def check_merge_integrity(note_body: str, chapters: list[Chapter]) -> GateResult: ...
```

Where possible, prefer deterministic checks over another model pass.

- [ ] **Step 2: Define a `FidelityReport` builder that aggregates gate results**

Use the same gate letters as the skill so failures are readable.

- [ ] **Step 3: Wire the verification stage into the coordinator**

Do not write “done” or return cleanly until the gates pass.

- [ ] **Step 4: Test good and bad cases**

Include fixtures for:
- gapped chapters;
- mismatched chapter index and heading counts;
- missing original embed/content;
- clean happy path.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_verify.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/verify.py src/jake_tools/transcripts/coordinator.py tests/test_verify.py
git commit -m "feat: add fidelity verification gates"
```

## Task 10: Add full meeting-minutes generation and merge

**Files:**
- Modify: `src/jake_tools/transcripts/prompts.py`
- Modify: `src/jake_tools/transcripts/stages.py`
- Modify: `src/jake_tools/transcripts/merge.py`
- Modify: `src/jake_tools/transcripts/coordinator.py`
- Create: `tests/test_minutes_merge.py`

- [ ] **Step 1: Add a dedicated minutes-extraction stage**

Keep this separate from chaptering and transcript polishing.

```python
def run_minutes_stage(...)-> MeetingMinutes: ...
```

The prompt should emphasise structured minutes above the transcript, not note replacement.

- [ ] **Step 2: Extend the merge logic to insert `## Meeting Minutes` above transcript sections**

Preserve everything else.

- [ ] **Step 3: Persist minutes artifacts and include them in final reports**

Store the raw JSON and the rendered markdown fragment so debugging is easy.

- [ ] **Step 4: Test that the merge preserves authored note content and embed**

This is a regression magnet. Explicitly assert on it.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_minutes_merge.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/transcripts/prompts.py src/jake_tools/transcripts/stages.py src/jake_tools/transcripts/merge.py src/jake_tools/transcripts/coordinator.py tests/test_minutes_merge.py
 git commit -m "feat: add meeting minutes generation"
```

## Task 11: Add CLI options and user-facing reporting

**Files:**
- Modify: `src/jake_tools/cli/transcribe.py`
- Create: `tests/test_transcribe_obsidian_recording_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Add explicit CLI options for mode and artifact handling**

Recommended options:
- `--mode [transcript|chaptered-transcript|minutes]`
- `--output-root PATH`
- `--keep-working-dir/--cleanup-working-dir`
- `--json`
- `--dry-run`

- [ ] **Step 2: Emit concise operator output**

The command should print paths to:
- updated note;
- merged audio;
- raw transcript JSON;
- chapters/fidelity reports if present.

If `--json` is set, print a machine-readable summary object.

- [ ] **Step 3: Add CLI tests for mode routing and error messaging**

Cover:
- missing note;
- dry-run success;
- transcript-only mode;
- invalid mode;
- JSON output path summary.

- [ ] **Step 4: Document prerequisites and examples in README**

Document that the command expects local tooling such as `scribe` and likely `ffmpeg`.

- [ ] **Step 5: Run the focused tests**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest tests/test_transcribe_obsidian_recording_cli.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/jake_tools/cli/transcribe.py README.md tests/test_transcribe_obsidian_recording_cli.py
git commit -m "feat: add obsidian recording CLI options"
```

## Task 12: Final integration pass

**Files:**
- Modify: any touched files from prior tasks

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/work/Documents/jake-tools
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run static type checking**

```bash
cd /Users/work/Documents/jake-tools
uv run pyright
```

Expected: 0 errors.

- [ ] **Step 3: Smoke test the CLI help and dry-run path**

```bash
cd /Users/work/Documents/jake-tools
uv run jake-tools transcribe obsidian-recording --help
```

Then run a dry-run against a fixture note.

- [ ] **Step 4: Run a real end-to-end local transcript-only check against a disposable fixture note**

This should exercise the deterministic path and confirm artifact writing and note updates.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: implement obsidian recording coordinator"
```

---

## Expected coordinator API shape

The implementation should converge on something close to this:

```python
class ObsidianRecordingCoordinator:
    def __init__(
        self,
        *,
        hermes: Hermes,
        source_note: Path,
        mode: Literal["transcript", "chaptered-transcript", "minutes"] = "minutes",
        output_root: Path | None = None,
        keep_working_dir: bool = True,
    ) -> None: ...

    def run(self) -> CoordinatorResult: ...

    def ingest(self) -> CoordinatorState: ...
    def transcribe(self, state: CoordinatorState) -> CoordinatorState: ...
    def chapter(self, state: CoordinatorState) -> CoordinatorState: ...
    def generate_minutes(self, state: CoordinatorState) -> CoordinatorState: ...
    def merge(self, state: CoordinatorState) -> CoordinatorState: ...
    def verify(self, state: CoordinatorState) -> CoordinatorResult: ...
```

## Risks and watchpoints

- **Broken path lifetime**: fix `TemporaryDirectory()` usage first or every later step will write into deleted paths.
- **Provider/model resolution**: `AIAgent()` with empty provider/model can take surprising paths. Keep the wrapper explicit enough that the coordinator gets a predictable runtime.
- **Model-output fragility**: every Hermes stage must parse into a typed model or fail fast.
- **Over-agentic merge logic**: do not let Hermes rewrite the whole note. Keep final note writes deterministic.
- **Tool prerequisites**: `scribe`, `ffmpeg`, and probably `obsidian` need clear preflight checks and crisp failures.
- **No tests in the repo yet**: add the test harness early so the coordinator doesn’t become an untestable blob.
- **Parallel chapter polishing**: defer true parallelism until the single-threaded chapter pipeline is solid. The skill permits fan-out, but the first implementation does not need concurrent worker orchestration.

## Suggested implementation order

If you want the shortest path to a useful command:
1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 7
6. Task 8
7. Task 9
8. Task 10
9. Task 11
10. Task 12

Tasks 5 and 6 slot in before chaptering if you want the Hermes stage surface clean from the start.

## Definition of done

This feature is done when:
- `jake-tools transcribe obsidian-recording NOTE.md` executes end-to-end;
- the source note is updated in place without losing authored content or recording embeds;
- artifacts are written to a durable working directory;
- transcript/chapter/minutes modes all behave distinctly;
- fidelity gates block bad merges;
- the command has tests, docs, and clear operator output.
