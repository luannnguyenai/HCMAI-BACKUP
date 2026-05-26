# AGENTS.md — Instructions for AI Coding Assistants

> Read this before performing any task in this repository. This file is the contract between human contributors and AI assistants (Cursor, Claude Code, Codex, GitHub Copilot, etc.). It supplements — not replaces — [`CONTRIBUTING.md`](CONTRIBUTING.md).

## The one rule

**No code without a spec. No decision without a record.**

If a task asks you to write code that affects behaviour, find the spec under `docs/specs/`. If there is none, your first deliverable is the spec, not the code. If a task asks you to make an irreversible architectural choice, your first deliverable is an ADR under `docs/adr/`.

## Repo orientation, in dependency order

1. **`docs/strategy/00-master-strategy.md`** — why this competition, what wins. Read first.
2. **`docs/proposals/`** — architecture-level decisions. Always relevant.
   - `01-interactive-system-architecture.md` is the spine.
   - `02-automatic-track-agent.md` is the LangGraph agent.
   - `08-original-contributions.md` defines the original-work (C1, C2, C4) — our differentiation.
   - `09-llm-path-bakeoff.md` defines the bakeoff methodology (in-flight).
3. **`docs/specs/`** — component-level "what behaviour exactly". Implements pieces of proposals.
4. **`docs/adr/`** — append-only record of irreversible decisions. Read before suggesting alternatives.
5. **`docs/research-notes/`** — background grounding. Read when working on Vietnamese stack, LSC/VBS prior art, or foundation-model choices.
6. **`docs/papers/`** — 37 reference PDFs. Cite when relevant.

## What to do before writing or editing code

1. **Find the spec.** Look in `docs/specs/INDEX.md` for the matching spec ID. If found, read it end-to-end including frontmatter status.
2. **If no spec exists**, propose one first. Use [`docs/specs/template.md`](docs/specs/template.md). Hand the spec back to the human for status `Approved` before writing code.
3. **Find the ADRs** referenced by the spec. Read them. Do not override their decisions; if you think they're wrong, propose a *new* ADR that supersedes the old one.
4. **Find the existing tests** that exercise the spec's acceptance criteria. New tests must follow the same naming convention.

## What to include in every code change

- Source files begin with a comment: `# Implements SPEC-NNNN SS X.Y` (or `// Implements SPEC-NNNN SS X.Y` for TS).
- Tests are named after acceptance criteria: `def test_planner_emits_valid_json_AC2()`.
- Branch names: `spec/NNNN-short-kebab-name`.
- Commit subject: `[SPEC-NNNN] short description`.
- PR titles same as commit subjects when single-commit; otherwise `[SPEC-NNNN] umbrella description`.

## How to write a spec (the fast version)

Use [`docs/specs/template.md`](docs/specs/template.md). Aim for:
- **One purpose per spec**, not a feature roadmap.
- **Acceptance criteria are testable assertions**, not goals.
- **Non-functional requirements include numbers**, not adjectives ("p95 < 800 ms", not "fast").
- **Dependencies on other specs are explicit**, by ID.
- **No code in the spec**, except type signatures (Python `def` headers, TypeScript interfaces, JSON Schema, Pydantic models).

Length target: 100–250 lines. Specs longer than 400 lines should be split.

## How to write an ADR (the fast version)

Use [`docs/adr/template.md`](docs/adr/template.md). An ADR is ~30–80 lines. It has four sections:
- **Context**: what's the situation at the moment of the decision?
- **Decision**: what we're doing, in one paragraph.
- **Consequences**: positive and negative, including what we're closing off.
- **Alternatives considered**: at least two, with the one-sentence reason each was not chosen.

Do not edit the Decision or Context of an Accepted ADR. To change the decision, write a new ADR that supersedes the old one and update the old one's status.

## Things you must not do without explicit permission

1. **Push to `main` directly.** Always open a PR.
2. **Force-push** to shared branches.
3. **Edit accepted ADRs.** Write a superseding ADR instead.
4. **Edit frozen criteria** (e.g. [`docs/proposals/09-llm-path-bakeoff.md`](docs/proposals/09-llm-path-bakeoff.md) §3 success criterion is frozen at its commit SHA).
5. **Commit secrets** (.env, credentials, API keys, model weights). Check `.gitignore` covers your case.
6. **Add new dependencies** without noting them in the spec under "Dependencies" and explaining the choice.
7. **Delete or rewrite tests** to make them pass. Tests track acceptance criteria; if a test is wrong, the spec is wrong, fix the spec.

## When a task is ambiguous

In order:
1. Re-read the relevant spec and ADRs.
2. Re-read the proposal that the spec sits under.
3. Ask the user a single, targeted clarifying question. Do not start writing speculative code.
4. If the user is unavailable and the work is unblockable, write a Draft spec capturing your interpretation, mark open questions explicitly, and stop there until the user confirms.

## Quality bar for code in this repo

- **Python**: 3.11+, type-hinted, `pyproject.toml`-managed (uv preferred). Format with `ruff format`, lint with `ruff check`. Tests with `pytest`.
- **TypeScript**: Strict mode, `vite` build, no `any` without a `// reason:` comment.
- **No print debugging in committed code.** Use `logging` with named loggers.
- **No hardcoded paths.** Use `pathlib.Path` + config.
- **No magic numbers in algorithms.** Surface them as constants with comments citing the spec section.

## Quality bar for documentation in this repo

- **Plain ASCII / common Unicode only** for the body of markdown files. Vietnamese is fine. ASCII box-drawing characters are fine. Curly quotes, em-dashes from rich text editors are *not* (they sometimes break under Windows PowerShell + CP1252; we learned the hard way).
- **No emojis in committed docs.** They're noise.
- **Cite sources by URL and section**, not by hearsay.
- **Numbers, not adjectives.** "WER 8.14 on CMV-Vi" beats "good Vietnamese ASR".

## Speed expectations

- Spec authoring: **30–60 minutes** for a typical module spec.
- ADR authoring: **15–30 minutes** for a typical decision.
- PR turnaround for a single-spec implementation: **same day or next day** including review and eval evidence.

If you are an AI assistant working on a long task, batch tool calls aggressively and keep the human informed with a TODO list. Do not silently disappear into a 20-minute search; surface progress.

## When in doubt

Prefer **leaving a comment in the spec** over making a unilateral choice. The cost of an extra clarifying question is seconds; the cost of building the wrong thing is days.
