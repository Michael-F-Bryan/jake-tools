from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jake_tools.transcripts.bundle.ids import ID_PREFIXES, IdPrefix, mint_id
from jake_tools.transcripts.bundle.records import (
    ArtefactRecord,
    SourceAssociation,
    SourceMembershipRecord,
)


@pytest.mark.parametrize("prefix", ID_PREFIXES)
def test_mint_id_produces_a_prefixed_uuid7_string(prefix: IdPrefix) -> None:
    minted = mint_id(prefix)

    assert re.fullmatch(
        rf"{prefix}_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}",
        minted,
    )


def test_mint_id_calls_are_never_equal() -> None:
    assert mint_id("bundle") != mint_id("bundle")


def test_source_membership_record_rejects_a_bundle_id_where_a_source_id_belongs() -> (
    None
):
    with pytest.raises(ValidationError):
        SourceMembershipRecord(
            source_id=mint_id("bundle"),  # wrong prefix for source_id
            bundle_id=mint_id("bundle"),
            association=SourceAssociation.OPERATOR_ASSERTION,
            evidence="cli",
        )


def test_artefact_record_rejects_a_source_id_where_an_artefact_id_belongs() -> None:
    with pytest.raises(ValidationError, match="artefact_id"):
        ArtefactRecord(
            artefact_id=mint_id("source"),  # wrong prefix for artefact_id
            bundle_id=mint_id("bundle"),
            source_id=mint_id("source"),
            acquisition_locator="/tmp/file.wav",
            sha256="0" * 64,
            blob_ref="blobs/" + "0" * 64,
            kind="audio",
            producer="test",
            created_at=datetime.now(UTC),
        )
