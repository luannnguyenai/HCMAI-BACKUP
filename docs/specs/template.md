---
id: SPEC-NNNN
title: <Short Title>
status: Draft
owner: <name or role>
created: YYYY-MM-DD
updated: YYYY-MM-DD
implements_proposal: <e.g. docs/proposals/01-interactive-system-architecture.md SS 5.X>
related_adrs:
  - ADR-NNNN
depends_on:
  - SPEC-NNNN
---

# SPEC-NNNN — <Title>

> One-paragraph summary. What does this component do, and what is it part of?

## 1. Context

Why does this spec exist now? What proposal does it implement? What is the user-visible or competition-visible behaviour that this enables?

## 2. Scope

### 2.1 In scope
- <bullet>
- <bullet>

### 2.2 Out of scope
- <bullet — be explicit about what this spec is NOT responsible for>

## 3. API contract / interface

Function signatures, data shapes, message schemas. Use Python `def` headers, TypeScript `interface`, or JSON Schema. Do not include implementation code.

```python
def example_function(
    input: ExampleInput,
    *,
    config: Config | None = None,
) -> ExampleOutput:
    """Brief description; behaviour defined in section 4."""
    ...
```

Data classes / Pydantic models / TypeScript interfaces go here too.

## 4. Behaviour

Specify behaviour by case. Each case is testable.

- **Normal case**: <what happens>
- **Empty input**: <what happens>
- **Error case A**: <what happens>
- **Error case B**: <what happens>

If the spec is for an algorithm, describe it stepwise. Reference papers by URL when borrowing a method.

## 5. Acceptance criteria

Each criterion is a testable assertion. Tests must be named for the criterion ID.

- **AC1**: <testable assertion, e.g. "When given a valid Vietnamese query, the function returns a JSON object conforming to schema X.">
- **AC2**: <...>
- **AC3**: <...>

## 6. Non-functional requirements

- **Latency**: p50 < ___ ms, p95 < ___ ms on <hardware spec>
- **Memory**: peak VRAM < ___ GB
- **Throughput**: ? ___ queries/sec
- **Accuracy / quality**: <metric> ? ___ on <eval set>
- **Compatibility**: <runtime versions, OS, GPU>
- **Cost**: ? $___ per <unit>

## 7. Dependencies

- **Internal**: SPEC-NNNN, SPEC-NNNN (link to specs this depends on)
- **External**: <library>=<version>, <model checkpoint>, <service>
- **Data**: <which datasets, which indexes>

## 8. Test plan

- **Unit tests** (`tests/unit/test_<name>.py`):
  - `test_<name>_AC1`
  - `test_<name>_AC2`
- **Integration tests** (`tests/integration/test_<name>.py`):
  - <describe scenarios>
- **Eval-harness tasks** (`bin/eval --tasks <set>`):
  - <which mock-task slice, which metrics>

## 9. Open questions

- <bullet — flag anything the implementer needs to confirm with the spec owner>

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| YYYY-MM-DD | <name> | Created (Draft) |
| YYYY-MM-DD | <name> | Moved to Review |
| YYYY-MM-DD | <name> | Moved to Approved |
