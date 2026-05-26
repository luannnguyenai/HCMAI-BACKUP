---
id: ADR-0009
title: Spec-Driven Development is the team workflow
status: Accepted
decided_on: 2026-05-26
deciders:
  - team lead
---

# ADR-0009 — Spec-Driven Development is the team workflow

## Status

Accepted.

## Context

The repository already contains a substantial corpus of strategy, research, and architecture-level proposals (proposals 01–09 plus the recently-added 08 original-contributions and 09 LLM-path bakeoff). What it lacks is a workflow that ensures implementation traces back to the proposals, that irreversible decisions are recorded, and that no team member — human or AI — can silently make a behaviour-changing change without a written specification.

The team has 17 weeks to finals and 5 people. Without a workflow, the cost of relitigating decisions at PR review and re-implementing the same module twice will exceed the cost of the workflow itself by week 4.

## Decision

The team adopts **Spec-Driven Development (SDD)** as the binding workflow, documented in [`CONTRIBUTING.md`](../../CONTRIBUTING.md) at the repo root with an agent-specific supplement in [`AGENTS.md`](../../AGENTS.md). Key rules:

1. **No code without a spec.** Each component-level deliverable has a SPEC-NNNN under `docs/specs/`, written before implementation. Specs follow [`docs/specs/template.md`](../specs/template.md) and are tracked in [`docs/specs/INDEX.md`](../specs/INDEX.md).
2. **No irreversible decision without an ADR.** Architecture decisions go under `docs/adr/`, follow [`docs/adr/template.md`](template.md), and are append-only: changing a decision means writing a new ADR that supersedes the old one.
3. **PRs reference the spec ID** via [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md), branch names, commit subjects, and source-file headers.
4. **Tests are named for acceptance criteria** so a green test maps unambiguously to a satisfied criterion.
5. **Eval evidence is required** for any PR that affects competition score: `bin/eval` output attached to the PR, no regression on the relevant slice.
6. **Specs and ADRs have explicit lifecycles** (Draft ? Review ? Approved ? Implementing ? Implemented ? Deprecated for specs; Proposed ? Accepted ? Superseded for ADRs).

Reserved spec IDs (`SPEC-0001` through `SPEC-0019`) and accepted ADR IDs (`ADR-0001` through this one, ADR-0009) are pre-populated in the respective INDEX files so the team can start authoring immediately.

## Consequences

### Positive
- Every behaviour-changing change is traceable from intent ? architecture ? spec ? code ? test.
- Onboarding a new team member becomes "read the INDEX files in order"; no tribal knowledge required.
- AI assistants (Cursor, Claude Code, Codex) have a single instruction file (`AGENTS.md`) that aligns their behaviour with the team's process.
- Decisions are not relitigated at PR review; if you disagree with an ADR, you must propose a successor.
- The post-competition paper / press kit assembles itself from the spec + ADR + eval corpus.

### Negative
- ~30 minutes of spec-authoring overhead per new module. Mitigated by template + AI assistance.
- Initial setup cost (this ADR + supporting files) of ~1 day.
- Discipline must be enforced — once a team member skips a spec "just this once," the practice erodes.

### Neutral / observable
- `docs/specs/` and `docs/adr/` become first-class artifacts of the same standing as `docs/proposals/`.
- PR review focuses on "does this match the spec" rather than "is this design good" — moving the design conversation to spec review, where it belongs.
- The bakeoff in proposal 09 will produce a SPEC-0002 implementation; the result will be referenced by future ADRs (e.g. an "ADR-0010: chosen planner path is X").

## Alternatives considered

- **No formal workflow** — fastest in the first week — rejected because the cost surfaces in week 4+ as rework and decision relitigation.
- **GitHub Issues as the single source of truth** — lower overhead — rejected because Issues are not version-controlled, not greppable from the codebase, and don't survive a repo migration; specs and ADRs in `docs/` do.
- **Adopt GitHub's spec-kit toolchain** — turnkey scaffolding — kept as a future option but not adopted now because the team's needs are well-served by a small set of markdown templates and the toolchain adds dependencies we don't need yet.
- **Lightweight RFC process (one file per decision)** — common in OSS — partially adopted: ADRs play the role of RFCs but are tighter and append-only.

## References

- Michael Nygard's original ADR pattern — <https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions>
- adr-tools — <https://github.com/npryce/adr-tools>
- GitHub spec-kit — <https://github.com/github/spec-kit>
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — the workflow itself
- [`AGENTS.md`](../../AGENTS.md) — agent supplement
