# Contributing — Spec-Driven Development (SDD)

> This repository follows **Spec-Driven Development**. No code lands without a spec. Read this once before your first contribution; reference it every time you open a PR.

## The shape of the repo

```
docs/
  strategy/          intent          - WHY we're doing this  (slow-changing)
  research-notes/    background      - what we learned from prior art
  proposals/         architecture    - WHAT we'll build at the architecture level
  specs/             component spec  - EXACTLY how each module behaves + is tested
  adr/               decisions       - IMMUTABLE record of irreversible choices
  papers/            references      - downloaded source material
  illustrations/     visuals         - diagrams for team discussion

src/                 the code        - HOW (implements specs)
experiments/         experiments     - ablations, bakeoffs, throwaway notebooks
eval-results/        eval outputs    - numbers, never edited by hand
```

The flow is **intent ? architecture ? spec ? ADR (when irreversible) ? code ? test ? eval**. Code that doesn't trace back to a spec, and decisions that don't trace back to an ADR, do not exist.

## The two artifact types you'll author

### Specs (`docs/specs/SPEC-NNNN-name.md`)
Component-level "exactly what". Each spec defines one buildable unit with:
- a clear API contract
- acceptance criteria mapped to tests
- non-functional requirements (latency, memory, accuracy)
- explicit dependencies on other specs

Template: [`docs/specs/template.md`](docs/specs/template.md). Registry: [`docs/specs/INDEX.md`](docs/specs/INDEX.md).

### ADRs — Architecture Decision Records (`docs/adr/ADR-NNNN-name.md`)
Immutable record of an irreversible decision. ADRs capture **why** at the moment of the decision, so future contributors don't re-litigate it.

Template: [`docs/adr/template.md`](docs/adr/template.md). Registry: [`docs/adr/INDEX.md`](docs/adr/INDEX.md).

A decision warrants an ADR if at least one is true:
- it's expensive to undo (e.g. data schema, evaluation server choice, primary embedding model)
- multiple alternatives were seriously considered
- the team will be asked "why did you do it that way?"

## The workflow

### 1. Before writing code
1. Find or write the relevant spec in `docs/specs/`. If it doesn't exist, write it (?30 minutes; use the template).
2. If the spec depends on a non-obvious decision, write the ADR first.
3. Set the spec status to **Approved** when the owner (you or a delegate named in the spec frontmatter) is satisfied. PR review for spec is allowed but not required for solo work.

### 2. While writing code
1. Every source file that implements a spec must reference it at the top: `# Implements SPEC-0007 SS 4`.
2. Every test must reference the acceptance-criterion ID: `def test_planner_emits_valid_json_AC2():`.
3. PR title must include the spec ID: `[SPEC-0007] vLLM-served planner skeleton`.

### 3. PR template
PRs use [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md). The template prompts for spec ID, ADR references, acceptance-criteria checklist, eval-harness results. Don't bypass it.

### 4. Eval evidence
For PRs that touch retrieval, ranking, fine-tuning, or any path that affects competition score, include `bin/eval` output (proposal 05 / SPEC for the harness) showing the relevant metrics did not regress. Numbers, not vibes.

## Spec lifecycle

```
Draft  ???  Review  ???  Approved  ???  Implementing  ???  Implemented  ???  Deprecated
  ?                                          ?
  ????????? push back if blocked ?????????????
```

| Status | Meaning |
|---|---|
| **Draft** | Author still iterating; not yet ready for review. |
| **Review** | Open for comments; should not be implemented yet. |
| **Approved** | Locked enough to start coding. Acceptance criteria are frozen. |
| **Implementing** | At least one PR open against this spec. |
| **Implemented** | All acceptance criteria pass on `main`. |
| **Deprecated** | Superseded by a newer spec; mark with the SPEC-XXXX that replaces it. |

Status is stored in the spec frontmatter `status:` field. The INDEX is regenerated from the frontmatter.

## ADR lifecycle

```
Proposed  ???  Accepted  ???  Superseded (by ADR-NNNN)
```

ADRs are append-only. Never edit the **Decision** or **Context** sections of an Accepted ADR. If the decision changes, write a new ADR that supersedes the old one and update the old one's status to "Superseded by ADR-NNNN".

## Numbering

- Specs: `SPEC-0001`, `SPEC-0002`, ... (4 digits, never reuse)
- ADRs:  `ADR-0001`, `ADR-0002`, ... (4 digits, never reuse)

When you reserve a number, append a row to the relevant INDEX immediately, even before the doc is filled in, so two people don't pick the same number.

## What is NOT spec-driven

Acceptable to skip the spec for:
- One-line fixes that don't change behaviour (typo, lint, formatting).
- Refactors with zero observable change (no API, no perf, no correctness shift).
- Throwaway exploration code in `experiments/` *that is not imported by anything in `src/`*.

If in doubt, write the spec. It's faster than re-litigating a design at PR review.

## Spec-driven for AI agents

When using AI assistants (Cursor, Claude Code, Codex, etc.) to write code in this repo, the agent must follow the same rules. The agent's runtime instructions are in [`AGENTS.md`](AGENTS.md).

## Quick-reference checklist for a new feature

- [ ] Spec drafted in `docs/specs/SPEC-NNNN-name.md` (frontmatter status `Draft`)
- [ ] ADR(s) drafted in `docs/adr/ADR-NNNN-name.md` for any irreversible decisions
- [ ] Owner approves spec ? status `Approved`
- [ ] Branch named `spec/NNNN-short-name`
- [ ] Code references the spec via header comment
- [ ] Tests named with acceptance-criterion ID
- [ ] PR opened with template filled in
- [ ] `bin/eval` output attached if competition-score-relevant
- [ ] Spec status set to `Implementing` on PR open, `Implemented` on merge

That's it. The discipline pays for itself within two weeks. Skip it and the 17-week timeline collapses into rework.
