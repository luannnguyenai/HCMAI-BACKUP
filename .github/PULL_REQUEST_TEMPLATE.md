<!--
PR title format:  [SPEC-NNNN] short description
Branch  format:   spec/NNNN-short-kebab-name
-->

## Spec(s) this PR implements

<!-- One or more SPEC IDs. If none, this is a docs/refactor/typo PR; delete this section
     but keep the "Out of SDD scope" checkbox in the checklist below. -->

- SPEC-NNNN ([`docs/specs/SPEC-NNNN-name.md`](../docs/specs/SPEC-NNNN-name.md))

## ADR(s) referenced

<!-- Decisions this change relies on. Read them; do not silently violate them. -->

- ADR-NNNN ([`docs/adr/ADR-NNNN-name.md`](../docs/adr/ADR-NNNN-name.md))

## Summary

<!-- 2-4 sentences. What changed. Why. -->

## Acceptance criteria

<!-- Mirror the spec's acceptance criteria. Tick each as you verify. -->

- [ ] AC1: <criterion text>
- [ ] AC2: <criterion text>
- [ ] AC3: <criterion text>

## Test plan

<!-- How you verified. Paste output of `bin/eval` or `pytest -k SPEC-NNNN` here. -->

```
$ bin/eval --tasks tests/<set>.jsonl --system <branch>
...
```

## Risk

<!-- What could this break? What did you do to prevent it? -->

- Regression risk: <low / medium / high>
- Rollback plan: <describe>

## Checklist

- [ ] Spec status is `Approved` or `Implementing` (not `Draft`)
- [ ] Code references the spec ID in file header comments
- [ ] Tests are named for the acceptance criteria they prove
- [ ] No new dependencies without noting them in the spec
- [ ] No secrets / credentials / model weights committed
- [ ] If competition-score relevant: `bin/eval` output included and shows no regression
- [ ] If this PR has no spec: it's a typo/lint/no-behaviour-change refactor (check one): out of SDD scope
