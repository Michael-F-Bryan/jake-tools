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
