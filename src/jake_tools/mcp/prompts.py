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
- Reconcile the proposal with the current implementation and prior discussion. If your recommendation has changed, explain what changed and why before presenting the new design. Make any proposed scope reduction or expansion explicit, and keep conditional wording in the request, such as "where necessary", as a condition to establish rather than a requirement.
- Identify the existing components, code or libraries to reuse, their responsibilities and the important boundaries between them. Give new abstractions a present reason to exist.
- Resolve discoverable technical questions through bounded inspection. Leave routine engineering choices to the implementer; surface unresolved questions only when they materially affect behaviour, scope, risk or the ability to proceed.

Write the issue comment in layers

1. Recommendation and rationale
Start with what will change, what will stay the same and why this resolves the problem. Explain the consequential trade-off in ordinary technical language. The opening should make sense without reading the implementation details below it.

2. Proposed behaviour and technical design
Give the reader a concrete model before introducing compressed technical labels. Use a worked example or short flow where it helps explain the behaviour.

Retain the specifics that constrain implementation: relevant interfaces, rules, thresholds, state transitions, ownership boundaries and affected source locations. Use a small table, diagram or code/configuration snippet when it explains these more clearly than prose. Do not add snippets for appearance, or present unqualified syntax as settled design.

Where the proposal changes a public API or communication between components, sketch the affected contract and interaction: who initiates the operation, what crosses the boundary, how success or failure is reported, and who owns the resulting state and lifecycle. Show the relevant request/response, event or message shape, or a short interaction sequence where useful. Distinguish existing contracts from proposed changes, and cover compatibility with current consumers. Specify enough to expose consequential design choices without prescribing every field or internal helper. If no contract change is needed, say which existing interaction is being reused.

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
    return f"""Review {issue_url} as the maintainer who will have to own, understand, debug and extend the result. Your primary job is to challenge whether the proposed change is worth owning, not to make the proposed design complete. Before reviewing its mechanics, compare it with the smallest credible intervention that delivers the required outcome. This is not a mandate to object: accept a small sound change briefly, and do not treat fewer lines as a better design.

Read the issue, discovery and planning comments, then inspect enough actual code and consumers to challenge their assumptions. Treat the plan as a proposal, not authority. Reuse applicable evidence and investigate only gaps that could materially change the judgement. Preserve settled decisions; if new evidence warrants reopening one, make that explicit.

Check the problem framing
Establish the underlying user or operational need, the evidence that something is wrong, and why it warrants intervention. Separate observed symptoms, inferred causes, requested outcomes and proposed mechanisms. Check whether the plan has inherited an unsupported diagnosis or treated a suggested solution as a requirement.

Work that intervention out before judging the proposal; it may be a narrower fix, an operational or configuration change, or no change. Consider whether correcting one part, such as the signal or upstream cause, removes the need for another mechanism. Explain the capability the smaller option sacrifices; doing less is not automatically better, and required behaviour must not be dropped silently. Existing code, tests and planning detail are not evidence that a responsibility is necessary.

Preserve explicit decisions and requirements, including conditional wording such as "where necessary", which asks you to establish the need. If the evidence suggests one should change, present that as a proposed revision for Michael to decide—not permission to redefine the task. If the framing is sound, proceed without manufacturing a challenge.

Distinguish corrections within the selected approach from proposals that materially expand ownership or operating responsibility beyond what was asked, for example substantial new persistent state, a migration or an operated CI job. Only a material expansion needs Michael's decision, not membership of a category: most helpers, fields, dependencies and tests are ordinary implementation choices for the implementer. Repetition across agent comments does not settle an expansion.

Judge the approach before its mechanics
Ask whether the proposed responsibilities are necessary, whether an existing component should own them, and whether any mechanism can disappear while preserving the required behaviour.

When you find a defect, reconsider the mechanism that creates it before it becomes another requirement: remove or relocate that mechanism, or check that the defect survives the smaller intervention, before adding a guard, state field or recovery path. Judge the combined burden of understanding, debugging, operating and extending the whole change, including maintained scripts, documents, dependencies and verification machinery, not whether each addition has a local justification. Compare ownership, coordination and debugging costs, not just code size. Challenge both overengineering and shortcuts that push complexity onto callers. Do not simplify by silently dropping requirements, and do not renovate unrelated code for consistency.

Examine responsibilities and interfaces
Look for misplaced responsibilities, duplicated mechanisms, hidden state, implicit ordering and changes that require coordinated edits across several components. Judge abstractions by the coherent complexity they encapsulate and what callers no longer need to know. One real consumer can justify a useful boundary; hypothetical consumers do not justify a framework.

Where public APIs or component interactions change, trace a representative operation across the proposed boundaries and existing consumers. Check:
- Who initiates the operation and owns authoritative state and lifecycle.
- What crosses each boundary and what a response or event actually guarantees.
- How completion, failure and recovery become visible to the caller.
- Compatibility with actual consumers and consequential retry, cancellation or ordering behaviour.

Use representative caller code or an interaction sketch where it exposes the design more clearly than signatures alone. What must the caller know, sequence, configure and recover from? Prefer interfaces that make ordinary correct use straightforward and localise likely changes, without hiding capabilities a real consumer needs.

If the plan lacks enough detail to assess a consequential boundary, identify the specific missing contract or interaction—not a general request for more design. Leave routine implementation choices to the implementer.

Challenge guarantees and evidence
Test the strongest promises, especially completeness, uniqueness, atomicity and successful delivery. Look for concrete scenarios where the proposed checks pass while the user-visible promise is false. Consider missing, malformed, stale or contradictory evidence where it could change the result.

Distinguish demonstrated defects, unresolved design risks and optional preferences. A plausible counterexample is not an observed failure; explain the conditions under which it matters.

Check that tests exercise meaningful behaviour rather than freezing internal structure or reproducing mock choreography. Establish whether real-system acceptance is reachable and what evidence it requires. Local substitutes cannot establish provider or protocol compatibility.

Do not recommend production hooks, orchestration or compatibility machinery mainly to make an unavailable acceptance test locally demonstrable. Prefer an explicit verification gap and the smallest credible evidence path over a different, more complicated production design.

Check delivery where it affects the design
For larger work, examine whether the slices expose integration risks early and produce useful end-to-end results. Check that actual dependencies and shared contracts support any proposed parallelism. Account for compatibility, persistent-state transitions, cutover and recovery where needed.

Keep depth proportionate to risk. Do not demand extra stages, a redesign or a complete specification for a small sound change.

Write the review
Post an issue comment leading with your judgement:
- Ready to implement.
- Implementable with specific corrections.
- Blocked on a decision or missing evidence.

Make a material problem-framing or ownership concern prominent in that judgement rather than burying it beneath design details.

Follow with the concerns that determine the judgement, ordered by consequence. For each material concern:
- Identify the problem assumption, proposed behaviour or boundary.
- Show the concrete scenario or source evidence that exposes the problem.
- Explain the consequence for users, callers or maintainers.
- Recommend the smallest coherent correction, including removal or reuse where appropriate.
- State whether it can be incorporated during implementation or needs a decision or bounded investigation first.

Include a short interaction sequence, contract excerpt or code reference when it makes a concern clearer. Explain the concrete case before introducing compressed technical labels. Keep separate concerns separate; do not compress several findings into a paragraph-sized bullet.

Separate necessary corrections from optional improvements and proposed requirement changes. Tie maintainability concerns to present friction or credible changes, not taste. Do not manufacture objections or fill the comment with generic endorsements. Explain why an existing part should stand only when that helps resolve a real design question. If no material concern remains, give a brief judgement.

Do not reproduce the plan, dump a review checklist or require a rewritten plan merely to restate settled corrections. Readiness is not implementation approval.

Link supporting evidence and source code at the inspected revision. Identify superseded guidance where relevant, and make the review understandable without access to your local checkout. On repeat reviews, focus on changes and unresolved concerns rather than reopening settled findings.

Read back the published comment and return its URL.

Review only. Do not implement changes, rewrite the issue's scope, or claim to speak on behalf of the human maintainer."""
