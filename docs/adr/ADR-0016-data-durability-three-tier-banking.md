---
id: ADR-0016
title: Data durability policy - three-tier banking and bank-before-consume
status: Accepted
decided_on: 2026-06-08
deciders:
  - team lead
related_adrs:
  - ADR-0011
supersedes:
superseded_by:
---

# ADR-0016 - Data durability policy: three-tier banking and bank-before-consume

## Status

Accepted. Extends [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md); does not supersede it.

## Context

[ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) decided *where* artifacts live (Cloudflare R2 as the durable store, four storage tiers across lease rollovers). It did not decide *which* artifacts must be banked, *when*, or *what may consume them*. That gap caused a data-loss incident.

On 2026-06-08 an ephemeral GPU lease box rebooted and wiped `/tmp`. That destroyed the only copy of ~121,457 AIC2025 proxy keyframe JPGs (~19 GB), because the keyframes had been staged into `/tmp` and never banked to R2 - only the embeddings and manifests *derived* from them had been banked. The expensive derived artifacts that were banked incrementally as each shard/stage finished (the SPEC-0006 index lanes, the SPEC-0014 C1 heads, the SPEC-0005 OCR output) all survived the reboot and were recovered from R2. The loss was confined to one class: a fragile external raw input that had been treated as if it were cheap scratch.

Two latent assumptions made the loss possible: (1) `/tmp` was implicitly trusted to persist across a job's lifetime; (2) downstream stages depended on box-local files rather than on R2 objects, so "the keyframes are right there in `/tmp`" felt safe. Both are false on a lease box that can be reclaimed or rebooted with no notice. ADR-0011 already embodies the right instinct for derived artifacts (incremental banking via the `*_bank_watcher.sh` scripts); [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md) already embodies bank-before-consume for serving (it ingests indexes and hydrates image tiers *from R2*, not from a box). The incident shows the discipline must be a written, enforced policy, not a per-script habit.

## Decision

We adopt a **three-tier durability classification** for every artifact a lease job touches, and **three operational rules** that enforce it. This is orthogonal to ADR-0011's storage-location tiers: it governs *what to bank and when*, not *where things live*.

| Durability tier | Examples | Banking rule |
|---|---|---|
| **Tier 1 - external / fragile raw input** | Drive keyframe JPGs, organiser corpora, anything we cannot trivially re-fetch | Bank to R2 **immediately, before any processing, exactly once** |
| **Tier 2 - expensive derived artifact** | embeddings + index lanes, trained C1/C2/C4 heads, OCR / ASR output | Bank **incrementally** as each shard / stage completes |
| **Tier 3 - cheap / recomputable scratch** | `uv` venv, HF cache, intermediate tensors, sentinels | **Do not bank** (recompute on demand) |

Three operational rules bind the tiers:

- **(a) `/tmp` is volatile, always.** It is never the sole copy of a Tier 1 or Tier 2 artifact. A copy must exist in R2 before the job that produced or staged it can be considered done.
- **(b) Bank-before-consume.** A downstream stage may depend only on inputs that exist in R2, never on box-local files left by a previous stage. Stages hydrate their inputs from R2 (as SPEC-0026 already does for serving).
- **(c) Inventory + precondition check.** We maintain an R2 inventory of the prefixes a job requires, and a lease job runs a precondition check that verifies those prefixes exist and are non-empty before it starts, failing fast with a clear message otherwise. The check is specified and implemented in [SPEC-0028](../specs/SPEC-0028-r2-preflight-check.md) and wired as an opt-in guard in the `bin/remote` runner (ADR-0011 / SPEC-0022).

## Consequences

### Positive

- A whole class of "the only copy was on the box" losses is closed: Tier 1 inputs are durable before any compute touches them, and Tier 2 outputs survive a mid-run reboot.
- Bank-before-consume makes pipelines restartable on a fresh lease: any stage can be re-run from R2 inputs without re-staging from the original external source.
- The precondition check turns a silent "input missing -> job produces garbage or crashes late" into an immediate, legible failure before GPU time is spent.

### Negative

- Tier 1 banking adds an up-front upload cost (e.g. the ~19 GB keyframe set) and the R2 storage line item before the first useful compute runs. We accept this; it is small against the cost of re-deriving lost work under a 1-2 week lease.
- The policy adds a step (bank raw inputs first) that is easy to skip under time pressure; the precondition check is the backstop, but it only guards jobs that opt in and declare their prefixes.

### Neutral / observable

- The classification must be applied per job: a new runner author must decide each artifact's tier. The default for anything fragile and external is Tier 1.
- This does not change ADR-0011's storage model or the ledger schema; it adds a banking-discipline layer on top of it. The `*_bank_watcher.sh` scripts already implement Tier 2 incremental banking and need no change.

## Alternatives considered

- **Leave it as a per-script convention (status quo)** - zero new artifacts to maintain - rejected because the incident proves the convention is not reliably applied, especially to Tier 1 inputs that "feel" present in `/tmp`.
- **Snapshot all of `/tmp` to R2 on a timer** - catches everything without classification - rejected because it banks Tier 3 scratch (venv, HF cache, intermediate tensors) wastefully and still races a sudden reboot; classification + bank-before-consume is cheaper and deterministic.
- **Provision persistent cluster storage instead of relying on R2** - removes the volatility at the source - rejected because the lease provides no persistent storage (SPEC-0024 context, user-confirmed); R2 is the only durable layer we control.

## References

- [`ADR-0011`](ADR-0011-r2-artifact-store-and-lease-rollover.md) - the R2 store + lease-rollover storage model this extends.
- [`SPEC-0028`](../specs/SPEC-0028-r2-preflight-check.md) - the precondition-check behaviour and implementation.
- [`SPEC-0022`](../specs/SPEC-0022-remote-gpu-runner.md) - the `bin/remote` runner the guard is wired into.
- [`SPEC-0026`](../specs/SPEC-0026-mvp-serving-api.md) - prior art for bank-before-consume (serving ingests from R2).
- `infra/remote/index_bank_watcher.sh`, `infra/remote/ocr_bank_watcher.sh`, `infra/remote/c1_bank.sh` - the Tier 2 incremental-banking scripts.
