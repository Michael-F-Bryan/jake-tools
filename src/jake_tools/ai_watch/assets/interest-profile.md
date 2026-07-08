# Interest Profile

## Working hypothesis

Michael is interested in AI developments that change how real work is done with agents — especially where agents become part of the toolchain rather than a chatbot bolted onto an app.

The core theme:

> AI is becoming a programmable collaborator layer between humans, tools, and interfaces. The exciting developments are the ones that make that layer more capable, inspectable, steerable, and useful for real work.

This profile should bias towards practical AI engineering, interface design, and automation patterns that Michael could adapt into Hermes, jake-tools, Obsidian, Sunfish, or his daily software-development workflow.

Recency matters. Prefer newly released AI techniques, tools, and product surfaces. Old evergreen agent advice, familiar harness patterns, and articles Michael has effectively been applying in his day-to-day workflows for months should be rejected unless they introduce a genuinely new capability or unusually concrete evidence.

Michael uses Cursor for development and dislikes Xcode. Xcode-specific AI tooling is low value unless the article contains a clearly transferable non-Xcode pattern.

## Calibration examples

Source notes Michael provided from the Obsidian vault:

- `/Users/work/Documents/Vault/3 Resources/AI/Using Claude Code - The unreasonable effectiveness of HTML.md`
- `/Users/work/Documents/Vault/3 Resources/AI/The Weird Future Of User Interfaces.md`
- `/Users/work/Documents/Vault/3 Resources/AI/Harness design for long-running application development.md`

## Strong positive signals

### Agent-native workflows

Examples:

- Claude Code, Codex, Hermes, MCPs, CLIs, and tools that let agents operate across files, git history, browsers, SaaS APIs, and local context.
- Products exposing headless or agent-operable surfaces.
- Agent workflows that replace brittle manual UI work with typed APIs, CLI commands, or repeatable local automation.
- Articles that show real traces, artefacts, diffs, generated apps, or failure modes rather than polished demos only.
- New or recently changed tools, APIs, agent harnesses, workflows, or techniques with clear release timing.

Why it matters: Michael wants agents integrated into the toolchain, not chatbots pasted beside existing software.

### Long-running agent reliability

Examples:

- Planner/generator/evaluator architectures.
- Context handoffs and resets.
- Agent run manifests and resumable artefacts.
- QA agents, reviewer agents, and verification loops.
- Criteria tuning and model capability boundary testing.
- Techniques for simplifying scaffolding as models improve.

Why it matters: Michael treats harness design as engineering, especially for tasks that are too long or stateful for one prompt.

### Workflow-level automation

Examples:

- Automation that turns messy, multi-step agent work into deterministic commands.
- Tools with state, schemas, manifests, verification, and auditability.
- CLI/MCP designs where humans, cron, and agents can all invoke the same workflow.
- Repeatable pipelines for ingesting media, transcripts, docs, notes, repo context, or upstream changes.

Why it matters: this maps directly onto `jake-tools`, where Python glue is preferred for repeatable filesystem/API work and skills are better for judgement/process guidance.

### Agent-operable software and MCP-style surfaces

Examples:

- MCP servers and clients.
- SaaS products exposing CLIs or headless agent APIs.
- Typed local stdio tools for agents.
- Explicit tool boundaries, permission models, structured outputs, and compact result schemas.
- Products moving from screen-first to API/tool-first operation.

Why it matters: Michael is interested in software shaped for agents to operate safely and clearly.

### Interfaces after chat

Examples:

- Generative UI.
- Adaptive UI.
- Task-specific HTML artefacts.
- Throwaway editors and review surfaces.
- Human-agent loops that mix direct manipulation with delegation.
- Interfaces that export structured diffs, JSON, prompts, or code back into the workflow.

Why it matters: Michael sees AI as a medium for bespoke tools — design explorers, ticket triage boards, prompt editors, simulations, explainers, reports, and review surfaces.

### Rich workflow artefacts

Examples:

- HTML reports, explainers, design prototypes, code review artefacts, annotated diffs, visual plans, interactive dashboards, and generated editors.
- Artefacts that make agent work more legible and reviewable.
- Formats that improve human attention and reduce unreadable Markdown walls.

Why it matters: richer artefacts help humans stay in the loop when agents do more complex work.

### Verification, audit, and evidence trails

Examples:

- Append-only logs of agent decisions.
- Source-grounded summaries.
- Deterministic checks around LLM output.
- Evidence packs, manifests, run artefacts, and replayable pipelines.
- Systems that make false positives/false negatives visible.

Why it matters: Michael values reproducibility, visible evidence, and systems that can be tuned from real outcomes rather than vibes.

### Practical developer workflow improvements

Examples:

- PR review agents with annotated diffs.
- Repository brief generation.
- Codebase inspection and impact analysis.
- CI triage and targeted fix workflows.
- Agents that use tests, browser checks, and actual runtime verification.
- Tools that help with typed systems, compilers, Rust/Go/TypeScript/Python, and durable software maintenance.

Why it matters: Michael is a senior software engineer; useful AI developments should affect building, reviewing, debugging, or operating software.

Avoid surfacing IDE-specific news for tools Michael does not use, especially Xcode-specific workflows. Prefer Cursor, CLI, MCP, web, API, or editor-agnostic developments.

### Obsidian and knowledge-work automation

Examples:
