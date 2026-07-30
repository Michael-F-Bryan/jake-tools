"""CLI entrypoint: ``python -m inference_worker run REQUEST.json --out-dir DIR``.

Exit code 0 whenever a valid ``response.json`` was written, even if
individual stages failed — their status travels in the response. Nonzero
exit is reserved for contract-level failures where no valid response.json
could be produced at all: an unreadable/invalid request, an audio hash
mismatch against the request's declared hash, or an unwritable out-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from inference_worker.models import InferenceRequest
from inference_worker.orchestrator import run_inference
from inference_worker.provenance import sha256_file

_LOCKFILE_NAME = "uv.lock"
_RESPONSE_FILENAME = "response.json"


class ContractError(Exception):
    """A failure that means no valid response.json can be produced."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        request = _load_request(args.request)
        out_dir = _prepare_out_dir(args.out_dir)
        _verify_audio_hash(request)
    except ContractError as exc:
        print(f"inference-worker: {exc}", file=sys.stderr)
        return 1

    response = run_inference(request, out_dir, _lockfile_path())

    try:
        (out_dir / _RESPONSE_FILENAME).write_text(response.model_dump_json(indent=2))
    except OSError as exc:
        print(
            f"inference-worker: cannot write {_RESPONSE_FILENAME}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


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
