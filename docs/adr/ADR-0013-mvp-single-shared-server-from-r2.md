---
id: ADR-0013
title: MVP deployment topology is one shared server fed from R2
status: Proposed
decided_on: 2026-06-05
deciders:
  - team lead
related_adrs:
  - ADR-0003
  - ADR-0004
  - ADR-0011
---

# ADR-0013 - MVP deployment topology is one shared server fed from R2

## Status

Proposed.

## Context

SPEC-0006 is merged: the Milvus multi-vector keyframe store, `bin/index`, and a
single-lane `MilvusBackend` exist and have been validated on the lease box
(121,457 keyframes / 546 videos of the AIC2025 proxy corpus, ingest 64 s,
query p95 16-51 ms, recall@200 0.968-0.985). The artifacts that back it -
the per-encoder `.npy` indexes and the keyframe JPGs - live in Cloudflare R2
(bucket `aic2026-artifacts`, index prefix `index/aic2025-proxy-3enc-20260604/`)
per [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md).

We now need the team to qualitatively try Vietnamese KIS retrieval end to end
(query -> ranked keyframes) before the June 25 corpus and before any ground
truth exists. The deployment question is: where does that testable system run?
Three shapes were on the table: (1) each tester runs the whole stack locally;
(2) one shared server the team opens by URL; (3) a managed cloud service.
The dominant cost is hosting ~19 GB of AIC2025 keyframe JPGs plus the Milvus
indexes; replicating that per tester is wasteful, and the corpus has no
ground-truth answer key, so testing is human judgement over the same shared
data, not an automated score.

## Decision

The MVP is **one shared server**. It runs Milvus standalone (Docker), the
SPEC-0026 serving API, the static keyframe image tier, and the SPEC-0027 UI;
testers open a single URL and need nothing installed locally. The server is
**fed from R2 at provision time**: the SPEC-0006 indexes are pulled from
`index/aic2025-proxy-3enc-20260604/` and ingested into Milvus standalone via
`bin/index ingest-all`, and the keyframe images are hydrated from R2 to local
disk (image provenance and thumbnailing are [ADR-0015](ADR-0015-keyframe-image-hosting-thumbnails.md)).
R2 stays the source of truth (ADR-0011); the shared server is a rebuildable
cache of it, so a wiped box is restored by re-running provisioning, not by
hand-copying state.

## Consequences

### Positive

- One copy of the ~19 GB image corpus and the indexes, not one per tester;
  testers need only a browser.
- The shared box exercises the real Milvus-standalone path (ADR-0014), so what
  the team tries is the finals-aligned code path, not a local approximation.
- Rebuildable from R2 (ADR-0011): a lost or rolled-over box is restored by
  re-provisioning, with no unique state on the server.

### Negative

- Single point of failure for the test window: if the box is down, all testers
  are blocked. Acceptable for an internal MVP; not a finals constraint.
- The server holds the corpus and a public-ish URL; access control is a new
  concern (see open item in SPEC-0026).

### Neutral / observable

- This is a test/staging topology, not the finals topology. Finals run on two
  offline RTX 5070 laptops (ADR-0003); nothing here commits the finals path.
- Provisioning gains a new responsibility: ingest indexes + hydrate images from
  R2. SPEC-0026 owns that startup contract.

## Alternatives considered

- **Per-tester local stack** - no shared infra, each tester fully isolated -
  rejected because every tester would download and host ~19 GB of JPGs plus the
  indexes and run their own Milvus, which is slow to onboard and wastes storage.
- **Managed cloud retrieval service** (hosted vector DB + object store + CDN) -
  least ops on our side - rejected because it diverges from the finals
  Milvus-standalone path (ADR-0014), adds spend and an external account, and is
  overkill for an internal qualitative test.
- **Static pre-rendered result dump** (precompute top-k for a fixed query list)
  - trivial to host - rejected because it kills the interactive free-text KIS
  loop, which is the entire point of the test.

## References

- [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md) - the serving API + R2 startup contract.
- [SPEC-0027](../specs/SPEC-0027-mvp-operator-ui.md) - the KIS console served by this box.
- [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) - R2 as the source of truth.
- [SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md) - the merged Milvus store this hosts.
