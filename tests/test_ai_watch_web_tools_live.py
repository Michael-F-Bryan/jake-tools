from __future__ import annotations

import pytest

from jake_tools.ai_watch.web_tools import HermesWebTools


@pytest.mark.live
def test_live_web_search() -> None:
    tools = HermesWebTools()
    results = tools.search("site:anthropic.com/engineering harness", limit=1)
    assert results
