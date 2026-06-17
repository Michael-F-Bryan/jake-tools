from pathlib import Path

from jake_tools.transcripts.paths import Paths


def test_paths_temp_creates_artifact_root() -> None:
    with Paths.temp() as paths:
        assert paths.root.exists()
        assert paths.root.is_dir()
        assert paths.merged == paths.root / "merged.mp3"
        assert paths.transcript == paths.root / "merged.json"
        assert paths.chapters == paths.root / "chapters.json"
        assert paths.speaker_mapping == paths.root / "speaker-mapping.json"
        assert paths.merge_report == paths.root / "merge-report.json"


def test_paths_temp_cleans_up_after_exit() -> None:
    root: Path

    with Paths.temp() as paths:
        root = paths.root
        paths.transcript.write_text("{}")
        assert root.exists()

    assert not root.exists()
