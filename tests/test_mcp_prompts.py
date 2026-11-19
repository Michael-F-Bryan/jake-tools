"""Prompt discovery/rendering through real stdio, not agent-quality claims."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


@pytest.mark.parametrize(
    "issue_url",
    [
        "https://github.com/example/project/issues/42?context={literal}",
        "https://example.atlassian.net/browse/PROJECT-42",
    ],
)
async def test_issue_prompts(
    mcp_server: ServerFactory, mcp_env: dict[str, str], issue_url: str
):
    boundaries = {
        "issue_discovery": "This is discovery, not implementation.",
        "issue_planning": "This is planning, not permission to implement.",
        "issue_maintainer_review": "Review only. Do not implement changes",
    }
    async with mcp_server(mcp_env) as session:
        catalogue = {p.name: p for p in (await session.list_prompts()).prompts}
        for name, boundary in boundaries.items():
            prompt = catalogue[name]
            assert prompt.description
            assert prompt.arguments is not None
            assert [(a.name, a.required) for a in prompt.arguments] == [
                ("issue_url", True)
            ]
            result = await session.get_prompt(name, {"issue_url": issue_url})
            assert len(result.messages) == 1
            message = result.messages[0]
            assert message.role == "user"
            assert isinstance(message.content, TextContent)
            text = message.content.text
            assert issue_url in text
            assert boundary in text
            assert "Post " in text
            assert "comment" in text
            assert "Read" in text and "back" in text

        # The transport must reject a missing required argument, not serve an
        # unusable prompt with an unresolved placeholder.
        with pytest.raises(MCPError):
            await session.get_prompt("issue_discovery", {})


async def test_prompts_keep_ownership_and_conditional_wording_contract(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
):
    """Pin where the guidance sits, not its prose; judgement is evidenced elsewhere.

    The review must lead with whether the change is worth owning, before it
    reviews mechanics, and planning must cover expansions and conditions. Whether
    a model follows that is judged by real-model runs, not by this test.
    """
    async with mcp_server(mcp_env) as session:
        review = (
            (await session.get_prompt("issue_maintainer_review", {"issue_url": "u"}))
            .messages[0]
            .content
        )
        planning = (
            (await session.get_prompt("issue_planning", {"issue_url": "u"}))
            .messages[0]
            .content
        )
    assert isinstance(review, TextContent) and isinstance(planning, TextContent)

    opening = review.text.split("\n\n", 1)[0]
    assert "worth owning" in opening
    assert "smallest credible intervention" in opening
    assert review.text.index("worth owning") < review.text.index(
        "Judge the approach before its mechanics"
    )
    assert "materially expand ownership" in review.text
    assert "where necessary" in review.text
    assert "scope reduction or expansion" in planning.text
    assert "where necessary" in planning.text
