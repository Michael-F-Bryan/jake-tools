"""Meeting-transcription pipeline: typed intermediates and the run cache.

Stage implementations and CLI wiring land in later plans; this package starts
with the shared contracts (:mod:`.models`) and the on-disk run cache
(:mod:`.cache`) that every stage depends on.
"""

from __future__ import annotations
