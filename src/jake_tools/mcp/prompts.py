"""User-selected issue preparation prompts; retrieval has no side effects."""

from mcp.server.mcpserver import MCPServer


def register_prompts(app: MCPServer) -> None:
    app.prompt()(issue_discovery)
    app.prompt()(issue_planning)
    app.prompt()(issue_maintainer_review)


def issue_discovery(issue_url: str) -> str:
    """Ground an issue in the project and post the findings as an issue comment."""
    return f"""Investigate {issue_url} so we have a shared understanding of the problem and our starting point before planning implementation.

Read the issue and discussion, relevant project decisions, and current code and tests. Establish the intended user outcome, scope boundaries, existing behaviour and actual gaps. Trace important workflows through their owners, consumers and external dependencies. Reuse applicable evidence; identify what changed or still needs verification rather than repeating discovery.

Focus on uncertainties that could change scope, architecture, integration or acceptance. Explain which decision each affects and resolve it with the smallest useful check that does not change application behaviour or live data. Distinguish verified facts, assumptions, decisions needed from the user and details an implementer can reasonably settle. Stop when the next design decision is supported; do not exhaustively map the codebase.

Post a concise comment on the issue. Lead with the conclusion, then existing contracts, gaps and remaining decisions or checks. Link important evidence and identify the inspected revision. Make the handoff understandable without your local checkout. On subsequent passes, explain what changed and what remains valid rather than repeating the analysis; identify superseded guidance. Read the comment back to verify it was published correctly, then return its URL.

This is discovery, not implementation. Don't change code or turn tentative findings into agreed requirements."""


def issue_planning(issue_url: str) -> str:
    """Propose an implementation architecture and post it as an issue comment."""
    return f"""Plan the implementation of {issue_url}, using the discovery comment, issue discussion and current code as your starting point.

Recommend the smallest coherent design that delivers the full intended outcome. Explain responsibilities, boundaries, existing code or libraries to reuse, and domain guarantees that must survive. Give abstractions a present reason to exist; don't build a framework for hypothetical requirements. Preserve settled decisions unless new evidence warrants revisiting them, and distinguish them from recommendations and unresolved questions.

For larger work, propose coherent, verifiable delivery slices that together achieve the outcome. Identify the first end-to-end slice, dependencies and evidence needed before proceeding. Detail near-term work; leave later mechanics open where early results may change them. Where parallel work would help, identify independent workstreams, shared contracts and integration points; keep tightly coupled design, implementation and testing together. Do not add stages or agents merely because the task is large. Where existing consumers or persistent state are affected, account for compatibility, transition and recovery, not just the final design.

Explain how we will establish that it works. For consequential guarantees, identify evidence at the useful consumer boundary and plausible counterexamples that could disprove them. Address missing, malformed, stale or contradictory evidence where it affects those guarantees. Prefer behavioural tests that survive sensible refactoring over mocks of our own implementation. Identify the real-system acceptance path and access or environment prerequisites: local substitutes cannot establish real-system compatibility, and unavailable acceptance remains unverified.

Post a concise architecture proposal on the issue, leading with the recommendation and consequential trade-offs, followed by delivery sequence, verification and decisions still needed. Link evidence rather than reproducing it; make the handoff usable without your local checkout. Build on prior discussion and identify what changed or is superseded. Leave routine engineering choices to the implementer. Read it back and return the comment URL.

This is a proposal for discussion, not permission to implement. Don't change code or silently expand the ticket."""


def issue_maintainer_review(issue_url: str) -> str:
    """Challenge an issue's architecture for maintainability and post the review."""
    return f"""Review the proposed architecture for {issue_url} from the perspective of the primary maintainer who will have to understand, debug and extend this code.

Read the issue, discovery and planning comments, then inspect enough actual code to challenge their assumptions. Treat the plan as a proposal, not authority. Reuse applicable evidence and focus on consequential uncertainties rather than repeating discovery.

Look for misplaced responsibilities, unnecessary abstractions, duplicated mechanisms, hidden state, misleading failure semantics and tests that freeze implementation details. Challenge both overengineering and shortcuts that move complexity onto callers or future maintainers. For larger work, examine whether slices expose integration risks early, dependencies and shared contracts permit the proposed parallelism, and compatibility or state transitions are accounted for where needed.

Challenge the strongest promises, especially completeness, uniqueness, atomicity and successful delivery. Look for plausible cases where proposed checks pass while the user-visible promise is false, including missing or contradictory evidence. Check whether real-system acceptance can be reached rather than replaced by a local substitute. Distinguish demonstrated defects, unresolved risks and optional preferences; do not present hypothetical counterexamples as observed failures.

For each material concern, explain the concrete consequence and recommend a simpler or more maintainable alternative tied to evidence and likely changes, not taste. Don't manufacture objections: say which parts should stand. Keep depth proportionate to risk; a small sound change does not need extra stages or a redesign.

Post a concise review on the issue, leading with your judgement: ready to implement, implementable with specific corrections, or blocked on a decision or missing evidence. Separate necessary corrections from optional improvements. Say which corrections can be incorporated during implementation and which need a decision or bounded investigation first; don't require a rewritten plan just to restate settled corrections. Readiness is not implementation approval. Link evidence, identify the inspected revision and superseded guidance, and make the handoff understandable without your local checkout. On repeat reviews, focus on changes and unresolved concerns. Read it back and return the comment URL.

Review only. Don't implement changes, rewrite the issue's scope, or claim to speak on behalf of the human maintainer."""
