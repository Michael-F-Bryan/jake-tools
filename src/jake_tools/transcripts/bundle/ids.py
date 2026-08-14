"""Typed identity system for the bundle store (M1).

Every store-scoped ID is a string ``<prefix>_<uuid7>``. Each ID kind gets
its own ``Annotated`` pydantic type with a prefix-specific pattern, so
passing e.g. a ``SourceId`` into a field typed ``ArtefactId`` fails
validation instead of silently misassociating two unrelated records — the
prefixes never overlap.

IDs are minted only by :class:`.store.BundleStore`; nothing else in this
package (or its callers) constructs one from scratch. ``mint_id`` is the
single generator every store operation calls; the ``IdPrefix`` literal
gives pyright a closed set to check call sites against, so a typo'd prefix
is a type error, not a runtime surprise.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, get_args

from pydantic import StringConstraints

_UUID7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"


def _pattern(prefix: str) -> str:
    return rf"^{prefix}_{_UUID7}$"


IdPrefix = Literal[
    "bundle",
    "doc",
    "run",
    "source",
    "artefact",
    "rev",
    "component",
    "turn",
    "seg",
    "cluster",
    "participant",
    "review",
    "product_review",
    "render",
    "apply",
    "attempt",
    "continuation",
    "editorial",
    "overlap",
]

ID_PREFIXES: tuple[IdPrefix, ...] = get_args(IdPrefix)

BundleId = Annotated[str, StringConstraints(pattern=_pattern("bundle"))]
DocumentId = Annotated[str, StringConstraints(pattern=_pattern("doc"))]
RunId = Annotated[str, StringConstraints(pattern=_pattern("run"))]
SourceId = Annotated[str, StringConstraints(pattern=_pattern("source"))]
ArtefactId = Annotated[str, StringConstraints(pattern=_pattern("artefact"))]
RevisionId = Annotated[str, StringConstraints(pattern=_pattern("rev"))]
ComponentId = Annotated[str, StringConstraints(pattern=_pattern("component"))]
TurnId = Annotated[str, StringConstraints(pattern=_pattern("turn"))]
SegmentId = Annotated[str, StringConstraints(pattern=_pattern("seg"))]
ClusterId = Annotated[str, StringConstraints(pattern=_pattern("cluster"))]
ParticipantId = Annotated[str, StringConstraints(pattern=_pattern("participant"))]
ReviewId = Annotated[str, StringConstraints(pattern=_pattern("review"))]
ProductReviewId = Annotated[str, StringConstraints(pattern=_pattern("product_review"))]
RenderId = Annotated[str, StringConstraints(pattern=_pattern("render"))]
ApplyId = Annotated[str, StringConstraints(pattern=_pattern("apply"))]
AttemptId = Annotated[str, StringConstraints(pattern=_pattern("attempt"))]
ContinuationId = Annotated[str, StringConstraints(pattern=_pattern("continuation"))]
EditorialNodeId = Annotated[str, StringConstraints(pattern=_pattern("editorial"))]
OverlapGroupId = Annotated[str, StringConstraints(pattern=_pattern("overlap"))]


#: M6: the named coordinate domains a timed component may live in.
#: ``source:<artefact_id>`` is one media/caption artefact's own domain
#: (origin = media start = 0); ``combined:<component_id>`` is an assembled
#: multi-recording timeline's domain. ``wallclock`` and ``external:`` are
#: deliberately absent -- M6 allows both, but no capability may depend on
#: an ``external:`` mapping and nothing in v1 records wallclock as truth,
#: so admitting either into this pattern would only create a way to store
#: a domain no validator can honour.
CoordinateDomain = Annotated[
    str,
    StringConstraints(pattern=rf"^(source:artefact|combined:component)_{_UUID7}$"),
]


def source_domain(artefact_id: ArtefactId) -> str:
    """M6: name the ``source:<artefact_id>`` domain of one media artefact."""
    return f"source:{artefact_id}"


def combined_domain(component_id: ComponentId) -> str:
    """M6: name the ``combined:<component_id>`` domain of one assembled
    timeline component."""
    return f"combined:{component_id}"


def mint_id(prefix: IdPrefix) -> str:
    """Mint a fresh ``<prefix>_<uuid7>`` ID.

    Store-internal use only (M1: "IDs are minted by the store at record
    creation, nowhere else"). ``prefix`` is a closed ``Literal``, so a
    typo'd prefix is caught by pyright at the call site rather than
    producing a silently-wrong ID at runtime.
    """
    return f"{prefix}_{uuid.uuid7()}"
