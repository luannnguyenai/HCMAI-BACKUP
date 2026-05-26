---
id: ADR-0010
title: Borrow from the 2025 baseline repo under explicit attribution
status: Accepted
decided_on: 2026-05-26
deciders:
  - team lead
related_adrs:
  - ADR-0009
---

# ADR-0010 — Borrow from the 2025 baseline repo under explicit attribution

## Status

Accepted.

## Context

A May 26 review (see [`docs/research-notes/05-baseline-2025-analysis.md`](../research-notes/05-baseline-2025-analysis.md)) of the 2025 baseline at <https://github.com/ThanhToan2111/AIC_2026.git> found roughly a week's worth of de-risked, reusable plumbing: a complete DRES login + submit client, a well-thought-out set of Pydantic request schemas covering KIS / temporal / intelligent search, a TransNetV2 PyTorch wrapper with weights, a BLIP-2 reranker glue layer, and a primitive TRAKE algorithm.

Re-implementing these from scratch costs time we do not have. Copying them silently violates [ADR-0009](ADR-0009-sdd-workflow.md) (no untraceable code) and creates licensing risk. We need a clear policy.

## Decision

We may borrow source code, schemas, weights, and design patterns from the 2025 baseline repo (<https://github.com/ThanhToan2111/AIC_2026.git>) under the following policy:

1. **Per-file attribution.** Every borrowed file (or substantially-borrowed function/class) begins with a header comment naming the source file, the source commit SHA, and the SPEC under which we are using it:
   ```
   # Borrowed from ThanhToan2111/AIC_2026:streamlit_api.py:122-200
   # at commit c3c3545. Adapted under SPEC-0018 (DRES integration).
   ```

2. **A central `THIRD_PARTY.md` at the repo root** lists every borrowed item with: source repo + SHA + path + range, target location in our repo, the SPEC/ADR authorising the borrow, and the licence status.

3. **Licence verification before merge.** For each borrow, one of:
   - The source has an explicit OSI-approved licence (MIT/Apache/BSD) that allows reuse with attribution.
   - The source is implicitly inherited under a parent licence we can verify (e.g. a model architecture wrapper that is itself derivative).
   - We obtain explicit written permission from the original author (`ThanhToan2111`) — and the permission is filed under `docs/permissions/`.
   - Failing all of the above, we **rewrite** the module clean-room.

   The 2025 baseline repo currently has **no LICENCE file**. Default-by-omission for GitHub repos is "all rights reserved" — we therefore need explicit permission OR a clean-room rewrite for any non-trivial borrow.

4. **Borrowed code is still spec-driven.** It must be referenced by a SPEC-NNNN, follow our naming/testing/linting conventions, and have acceptance criteria. Borrowing is *implementation-shortcut*, not *spec-shortcut*.

5. **Borrowed code is reviewed normally.** A PR that borrows code goes through the same PR template ([`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md)) as original work. The PR description names the borrow explicitly.

6. **Bug-fix-forward, not back-port.** If we find a bug in borrowed code, we fix it in our copy. We do not file issues against the 2025 baseline repo unless the original author has explicitly asked for upstream contributions.

## Consequences

### Positive
- Saves ~1 week of engineer-time on DRES integration and request-schema design.
- Preserves attribution and licensing hygiene from day one (vs the painful retrofit some teams have to do before publishing).
- Makes audit trivial: any reviewer can run `grep -r "Borrowed from"` to find every external dependency on prior work.
- Sets a precedent for handling any *future* prior-art borrowings (e.g. LSC reference implementations).

### Negative
- ~10 minutes of overhead per borrow for the header + `THIRD_PARTY.md` entry. Acceptable.
- We are blocked on getting permission from `ThanhToan2111` (or doing clean-room rewrites) before any substantive borrow lands on `main`. The DRES URL itself (a literal string) is not copyrightable and can be used immediately; the integration code is.

### Neutral / observable
- A new top-level `THIRD_PARTY.md` file enters the repo on the first borrow.
- Reviewers must check the borrowing header in addition to the normal SDD checks.

## Alternatives considered

- **Rewrite everything from scratch** — purest licensing posture — rejected because we lose ~1 engineer-week and the 2025 schemas are genuinely good designs.
- **Borrow silently without attribution** — fastest — rejected because it violates [ADR-0009](ADR-0009-sdd-workflow.md) ("no code without a trace") and creates legal risk if we publish a SoICT paper.
- **Fork the 2025 repo and submodule it in** — least friction — rejected because their repo has no licence file, no tests, no CI, and a single squashed commit; we cannot tolerate that as an upstream.

## References

- 2025 baseline repo: <https://github.com/ThanhToan2111/AIC_2026.git> at commit `c3c3545`
- Analysis: [`docs/research-notes/05-baseline-2025-analysis.md`](../research-notes/05-baseline-2025-analysis.md)
- First borrow: [`docs/specs/SPEC-0018-dres-integration.md`](../specs/SPEC-0018-dres-integration.md)
- Workflow context: [`docs/adr/ADR-0009-sdd-workflow.md`](ADR-0009-sdd-workflow.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
