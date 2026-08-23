"""Meeting-transcription pipeline: typed intermediates and the run cache.

The shared contracts (:mod:`.models`) and the on-disk run cache
(:mod:`.cache`) that every stage depends on live here; the stage
implementations themselves (audio merge, ASR, adapt, speaker resolution,
chapterisation, polish, minutes, integrate) and their composition into
`jake-tools transcribe` live in this package's other modules and
``cli/transcript.py``/``cli/transcribe.py``.
"""

from __future__ import annotations
