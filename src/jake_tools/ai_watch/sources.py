from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceQuery:
    source_id: str
    query: str
    limit: int = 5


DEFAULT_SOURCE_QUERIES: tuple[SourceQuery, ...] = (
    SourceQuery(
        "anthropic_engineering", "site:anthropic.com/engineering agent harness", 5
    ),
    SourceQuery("anthropic_engineering", "site:anthropic.com/engineering MCP", 5),
    SourceQuery(
        "anthropic_engineering", "site:anthropic.com/engineering Claude SDK", 5
    ),
    SourceQuery("anthropic_news", "site:anthropic.com/news Claude SDK agents", 5),
    SourceQuery("anthropic_news", "site:anthropic.com/news Claude Code engineering", 5),
    SourceQuery(
        "anthropic_cookbook",
        "site:github.com/anthropics/anthropic-cookbook Claude SDK agents",
        5,
    ),
    SourceQuery(
        "anthropic_sdk_python",
        "site:github.com/anthropics/anthropic-sdk-python tools agents",
        5,
    ),
    SourceQuery(
        "anthropic_sdk_typescript",
        "site:github.com/anthropics/anthropic-sdk-typescript tools agents",
        5,
    ),
    SourceQuery("claude_code_sdk", '"Claude Code SDK" agents examples', 5),
    SourceQuery("simon_willison", "site:simonwillison.net MCP agent", 5),
    SourceQuery("latent_space", "site:latent.space agent engineering", 5),
    SourceQuery("hn_agents", 'site:news.ycombinator.com "Claude Code"', 5),
    SourceQuery("hn_mcp", "site:news.ycombinator.com MCP agent", 5),
    SourceQuery(
        "github_claude_code",
        "site:github.com/anthropics/claude-code releases",
        5,
    ),
    SourceQuery(
        "github_mcp_servers",
        "site:github.com/modelcontextprotocol/servers releases",
        5,
    ),
)
