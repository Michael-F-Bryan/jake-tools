"""User-selected issue preparation prompts; retrieval has no side effects."""

from mcp.server.mcpserver import MCPServer


def register_prompts(app: MCPServer) -> None:
    app.prompt()(issue_discovery)
    app.prompt()(issue_planning)
    app.prompt()(issue_maintainer_review)


def issue_discovery(issue_url: str) -> str:
    """Ground an issue in the project and post the findings as an issue comment."""
    return f"""Investigate {issue_url} so we have a shared understanding of the problem and our starting point before planning implementation.

Read the issue and discussion, relevant project decisions, and the current code and tests. Establish the intended user outcome, scope boundaries, what already works, and the actual gaps. Follow the evidence where it matters; don't repeat discovery already adequately documented.

Distinguish verified facts from assumptions and unresolved questions. Identify the uncertainties that could materially change the implementation, resolving what you can without changing application behaviour or live data.

Post a concise comment on the issue explaining your understanding, the relevant existing code and contracts, and what still needs deciding or verifying. Link the important evidence and identify the inspected revision. Read the comment back to verify it was published correctly, then return its URL.

This is discovery, not implementation. Don't change code or turn tentative findings into agreed requirements."""


def issue_planning(issue_url: str) -> str:
    """Propose an implementation architecture and post it as an issue comment."""
    return f"""Plan the implementation of {issue_url}, using the discovery comment, issue discussion and current code as your starting point.

Recommend the smallest coherent design that delivers the intended outcome. Explain the important responsibilities and boundaries, what existing code or libraries we should reuse, and which domain guarantees must survive the change. Give abstractions a concrete reason to exist; don't build a framework for hypothetical future requirements.

Describe how we'll establish that the implementation works: the user-visible claims, useful test boundaries, and any real-system acceptance needed. Prefer tests that catch plausible regressions and survive sensible refactoring over mocks of our own implementation.

Post a concise architecture proposal on the issue, including the consequential trade-offs, a practical implementation sequence, and any decisions that remain open. Make a recommendation rather than presenting an unranked menu. Read it back and return the comment URL.

This is a proposal for discussion, not permission to implement. Don't change code or silently expand the ticket."""


def issue_maintainer_review(issue_url: str) -> str:
    """Challenge an issue's architecture for maintainability and post the review."""
    return f"""Review the proposed architecture for {issue_url} from the perspective of the primary maintainer who will have to understand, debug and extend this code.

Read the issue, discovery and planning comments, then inspect enough of the actual code to challenge their assumptions. Treat the plan as a proposal, not authority.

Look for misplaced responsibilities, unnecessary abstractions, duplicated mechanisms, hidden state, misleading failure semantics and tests that freeze implementation details rather than protect behaviour. Challenge both overengineering and shortcuts that merely move complexity onto callers or future maintainers.

For each material concern, explain the concrete consequence and recommend a simpler or more maintainable alternative. Tie criticism to evidence and likely changes, not personal taste. Don't manufacture objections: say which parts of the plan should stand.

Post a concise review on the issue with your judgement—ready to implement, needs revision, or blocked on evidence—and the changes needed to reach an implementable plan. Distinguish necessary corrections from optional improvements. Read it back and return the comment URL.

Review only. Don't implement changes, rewrite the issue's scope, or claim to speak on behalf of the human maintainer."""
