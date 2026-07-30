"""CLI entrypoint: ``python -m inference_worker run REQUEST.json --out-dir DIR``.

Exit code 0 whenever a valid ``response.json`` was written, even if
individual stages failed — their status travels in the response. Nonzero
exit is reserved for contract-level failures where no valid response.json
could be produced at all: an unreadable/invalid request, a declared model
name that isn't the one this worker pins (M4), an audio hash mismatch
against the request's declared hash, or an unwritable out-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from inference_worker import asr, diarise
from inference_worker.atomic_io import atomic_write_text
from inference_worker.models import InferenceRequest, InferenceResponse
from inference_worker.orchestrator import (
    RunInferenceFn,
    run_inference,
    worker_internal_error_response,
)
from inference_worker.provenance import sha256_file

_LOCKFILE_NAME = "uv.lock"
_RESPONSE_FILENAME = "response.json"


class ContractError(Exception):
    """A failure that means no valid response.json can be produced."""


def main(
    argv: list[str] | None = None, *, run_inference_fn: RunInferenceFn = run_inference
) -> int:
    args = _parse_args(argv)
    try:
        request = _load_request(args.request)
        _refuse_unpinned_models(request)
        _verify_audio_hash(request)
        # m5: only create the out-dir once every contract-level check
        # upstream has passed — a rejected request shouldn't leave an
        # empty directory behind.
        out_dir = _prepare_out_dir(args.out_dir)
    except ContractError as exc:
        print(f"inference-worker: {exc}", file=sys.stderr)
        return 1

    lockfile_path = _lockfile_path()
    try:
        response = run_inference_fn(request, out_dir, lockfile_path)
    except Exception as exc:
        # B3: every stage already converts its own known failure modes
        # into a clean typed result; this is the residual safety net so a
        # genuinely unexpected error can't lose the whole run to an
        # unhandled traceback and a missing response.json.
        try:
            response = worker_internal_error_response(
                request, out_dir, lockfile_path, exc
            )
        except OSError as write_exc:
            print(
                f"inference-worker: cannot write stage artefacts to {out_dir}: {write_exc}",
                file=sys.stderr,
            )
            return 1

    if not _write_response(out_dir, response):
        return 1
    return 0


def _write_response(out_dir: Path, response: InferenceResponse) -> bool:
    # B2: temp file + fsync + atomic rename, never a truncate-in-place
    # write — a mid-write failure must never destroy a previously good
    # response.json.
    try:
        atomic_write_text(
            out_dir / _RESPONSE_FILENAME, response.model_dump_json(indent=2)
        )
    except OSError as exc:
        print(
            f"inference-worker: cannot write {_RESPONSE_FILENAME}: {exc}",
            file=sys.stderr,
        )
        return False
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="inference-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="run one inference request against one audio artefact, then exit"
    )
    run_parser.add_argument("request", type=Path, help="path to a request JSON file")
    run_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for response.json and stage artefacts",
    )
    return parser.parse_args(argv)


def _load_request(path: Path) -> InferenceRequest:
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ContractError(f"cannot read request file {path}: {exc}") from exc
    try:
        return InferenceRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise ContractError(f"invalid request {path}: {exc}") from exc


def _refuse_unpinned_models(request: InferenceRequest) -> None:
    """M4: the worker runs exactly one pinned ASR model and one pinned
    diarisation model (no backend-selection flags). Silently satisfying a
    request that declares a different model with the pinned one is
    misleading — refuse at the contract boundary instead."""
    if request.asr_model.name != asr.MODEL_ID:
        raise ContractError(
            f"request declares asr_model.name={request.asr_model.name!r}, but this "
            f"worker only runs {asr.MODEL_ID!r} (no backend-selection flags)"
        )
    if request.diarisation_model.name != diarise.PIPELINE_ID:
        raise ContractError(
            f"request declares diarisation_model.name={request.diarisation_model.name!r}, "
            f"but this worker only runs {diarise.PIPELINE_ID!r} (no backend-selection flags)"
        )


def _prepare_out_dir(out_dir: Path) -> Path:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(f"cannot create out-dir {out_dir}: {exc}") from exc
    return out_dir


def _verify_audio_hash(request: InferenceRequest) -> None:
    audio_path = Path(request.audio.path)
    try:
        actual = sha256_file(audio_path)
    except OSError as exc:
        raise ContractError(f"cannot read audio artefact {audio_path}: {exc}") from exc
    if actual != request.audio.sha256:
        raise ContractError(
            f"audio hash mismatch for {audio_path}: request declared "
            f"{request.audio.sha256}, actual {actual}"
        )


def _lockfile_path() -> Path:
    # inference-worker/src/inference_worker/__main__.py -> inference-worker/uv.lock.
    # The worker is always run in place via `uv run` from its own project
    # directory, so the lockfile is reliably a sibling of `src/`.
    return Path(__file__).resolve().parents[2] / _LOCKFILE_NAME


if __name__ == "__main__":
    sys.exit(main())
