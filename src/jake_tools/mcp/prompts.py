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

Produce a proposal that lets Michael understand and judge the design, and gives an implementer enough technical direction to proceed without repeating discovery. This is planning, not permission to implement.

Establish the design
- Recommend the smallest coherent design that delivers the intended outcome. Preserve settled decisions and distinguish them from your recommendations.
- Reconcile the proposal with the current implementation and prior discussion. If your recommendation has changed, explain what changed and why before presenting the new design. Make any proposed scope reduction explicit.
- Identify the existing components, code or libraries to reuse, their responsibilities and the important boundaries between them. Give new abstractions a present reason to exist.
- Resolve discoverable technical questions through bounded inspection. Leave routine engineering choices to the implementer; surface unresolved questions only when they materially affect behaviour, scope, risk or the ability to proceed.

Write the issue comment in layers

1. Recommendation and rationale
Start with what will change, what will stay the same and why this resolves the problem. Explain the consequential trade-off in ordinary technical language. The opening should make sense without reading the implementation details below it.

2. Proposed behaviour and technical design
Give the reader a concrete model before introducing compressed technical labels. Use a worked example or short flow where it helps explain the behaviour.

Retain the specifics that constrain implementation: relevant interfaces, rules, thresholds, state transitions, ownership boundaries and affected source locations. Use a small table, diagram or code/configuration snippet when it explains these more clearly than prose. Do not add snippets for appearance, or present unqualified syntax as settled design.

Explain each mechanism alongside the behaviour or guarantee it supports. Keep separate concerns in separate paragraphs or bullets; do not turn bullets into compressed specifications.

3. Delivery
Describe coherent, verifiable slices, starting with the first useful end-to-end result. Detail near-term work and leave later mechanics open where early results may change them.

Include dependencies and consequential compatibility, migration, cutover or recovery constraints. Identify parallel work only where it is genuinely independent; keep tightly coupled design, implementation and testing together. Do not create stages or agents merely because the task is large.

4. Verification
Explain how we will establish that the proposed behaviour works at its useful consumer boundary. Include the important failure or counterexample that could disprove a consequential guarantee, rather than an exhaustive test inventory.

Distinguish local behavioural tests, historical evidence and real-system acceptance. Name access or environment prerequisites and any acceptance that must remain unverified. Account for missing, stale or contradictory evidence where it could change the result.

5. Decisions and limitations
State the material decisions Michael still owns, with your recommendation and the consequence of the alternative. Do not turn ordinary implementation choices into questions for him. If no decision is needed, say so briefly or omit this section.

Publication
Post the proposal as an issue comment. Build on the existing discussion rather than repeating discovery. Link supporting evidence and source code at the inspected revision; do not rely on local workspace paths for essential context.

Keep detailed research and exhaustive checklists in supporting material, but retain the technical facts needed to understand, judge and implement the proposal in the comment itself. Do not trade readability for either dense completeness or a sparse executive summary. Use only the headings and detail the issue needs.

Before posting, read the draft as a whole:
- Can the reader explain the proposed behaviour and main trade-off from the opening?
- Do the technical details make that model concrete, rather than interrupt it?
- Could an implementer identify the change boundaries and critical safeguards without repeating discovery?
- Are proposals, settled decisions and unverified details clearly distinguished?

Read back the published comment and return its URL. Do not change code, the issue description or status, or silently expand the ticket."""


def issue_maintainer_review(issue_url: str) -> str:
    """Challenge an issue's architecture for maintainability and post the review."""
    return f"""Review the proposed architecture for {issue_url} from the perspective of the primary maintainer who will have to understand, debug and extend this code.

Read the issue, discovery and planning comments, then inspect enough actual code to challenge their assumptions. Treat the plan as a proposal, not authority. Reuse applicable evidence and focus on consequential uncertainties rather than repeating discovery.

Look for misplaced responsibilities, unnecessary abstractions, duplicated mechanisms, hidden state, misleading failure semantics and tests that freeze implementation details. Challenge both overengineering and shortcuts that move complexity onto callers or future maintainers. For larger work, examine whether slices expose integration risks early, dependencies and shared contracts permit the proposed parallelism, and compatibility or state transitions are accounted for where needed.

Challenge the strongest promises, especially completeness, uniqueness, atomicity and successful delivery. Look for plausible cases where proposed checks pass while the user-visible promise is false, including missing or contradictory evidence. Check whether real-system acceptance can be reached rather than replaced by a local substitute. Distinguish demonstrated defects, unresolved risks and optional preferences; do not present hypothetical counterexamples as observed failures.

For each material concern, explain the concrete consequence and recommend a simpler or more maintainable alternative tied to evidence and likely changes, not taste. Don't manufacture objections: say which parts should stand. Keep depth proportionate to risk; a small sound change does not need extra stages or a redesign.

Post a concise review on the issue, leading with your judgement: ready to implement, implementable with specific corrections, or blocked on a decision or missing evidence. Separate necessary corrections from optional improvements. Say which corrections can be incorporated during implementation and which need a decision or bounded investigation first; don't require a rewritten plan just to restate settled corrections. Readiness is not implementation approval. Link evidence, identify the inspected revision and superseded guidance, and make the handoff understandable without your local checkout. On repeat reviews, focus on changes and unresolved concerns. Read it back and return the comment URL.

Review only. Don't implement changes, rewrite the issue's scope, or claim to speak on behalf of the human maintainer."""
