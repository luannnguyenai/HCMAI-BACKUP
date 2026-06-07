---
id: ADR-0014
title: The MVP retrieval path reuses the SPEC-0006 MilvusBackend on Milvus standalone
status: Accepted
decided_on: 2026-06-07
deciders:
  - team lead
related_adrs:
  - ADR-0004
  - ADR-0006
  - ADR-0008
related_specs:
  - SPEC-0006
  - SPEC-0004
---

# ADR-0014 - The MVP retrieval path reuses the SPEC-0006 MilvusBackend on Milvus standalone

## Status

Accepted (2026-06-07, PR #24).

## Context

The MVP serving API (SPEC-0026) must turn a Vietnamese KIS text query into
ranked keyframes. Two retrieval paths could back it:

1. The **merged SPEC-0006 path**: the `MilvusBackend` single-lane
   `harness.Backend`, the `MilvusKeyframeStore` ANN query, and the `bin/index`
   ingest CLI, all already validated on Milvus standalone (Docker) against the
   full 121,457-frame AIC2025 proxy corpus (SPEC-0006 SS 11.8).
2. A **throwaway numpy path**: load the per-encoder `.npy` matrices into RAM and
   brute-force inner product per query (the SPEC-0025 `DenseRetriever`
   precedent), no Milvus, no Docker.

The numpy path is faster to stand up for one box, but it is a second retrieval
implementation that the finals path does not use, so any quirk found while
testing it (scoring, tie-handling, filter semantics) would not transfer. The
team has an explicit "one code path, finals-aligned" goal for the MVP, and
SPEC-0006 already paid the cost of making Milvus standalone work (two start-up
deviations documented, recall NFR settled on stable 2.5.x).

## Decision

The MVP retrieval path **reuses the merged SPEC-0006 `MilvusBackend` and
`MilvusKeyframeStore` against a Milvus standalone (Docker) instance** on the
shared server (ADR-0013); it does **not** introduce a numpy or Milvus-Lite
retrieval path for serving. The server runs the same `bin/index ingest-all`
ingest from the R2 indexes and the same online per-field ANN query the lease-box
eval used. SPEC-0026 wraps these existing classes in an HTTP/WebSocket service
(the web framework is FastAPI per [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md);
this ADR does not re-decide the framework). Multi-lane combination in the MVP is
**single-lane or RRF** (the runtime fallback per [ADR-0008](ADR-0008-rrf-as-runtime-fallback.md));
the C2 learned fusion (SPEC-0015) is ground-truth-blocked and deferred.

## Consequences

### Positive

- One retrieval implementation from MVP test to finals; bugs and behaviour found
  while testing transfer directly to the competition path.
- Reuses validated deployment evidence (ingest 64 s, query p95 16-51 ms,
  recall@200 >= 0.95 at efSearch=1024) rather than re-proving a second path.
- The MVP exercises real ANN + scalar filters + the `Submission` adapter, so the
  serving contract is grounded in the actual store, not a stand-in.

### Negative

- The shared server must run Docker + Milvus standalone, which is heavier than an
  in-process numpy array and adds an operational dependency to the box.
- Couples the MVP availability to the Milvus container health (mitigated by
  ADR-0013 rebuild-from-R2).

### Neutral / observable

- Milvus Lite stays the dev/CI mode (SPEC-0006 SS 7); standalone is the served
  mode. The store already selects FLAT vs HNSW from the URI, so no serving code
  branches on this.
- The MVP ships exactly the three floor lanes (siglip2, metaclip2, qwen3vl);
  qwen3vl is queryable here (SPEC-0006 Q-d) but the online query encoder is the
  siglip2/metaclip2 text tower (ADR-0003), so the default served lanes are
  siglip2 and metaclip2.

## Alternatives considered

- **Throwaway numpy brute-force retriever** - simplest to host on one box, no
  Docker - rejected because it is a second retrieval path the finals system does
  not use, so test findings would not transfer and recall/scoring quirks would
  be re-discovered later.
- **Milvus Lite (embedded) for serving** - no Docker, same client API - rejected
  because Lite is FLAT/CPU and does not build the HNSW index the corpus needs;
  it is the dev/CI mode, not a serving mode (SPEC-0006 SS 7).
- **A managed/hosted vector DB** - least ops - rejected for the same divergence
  and cost reasons as in ADR-0013.

## References

- [SPEC-0006](../specs/SPEC-0006-milvus-schema-and-queries.md) - the merged store + `MilvusBackend` reused here.
- [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md) - the service that wraps it.
- [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md) - the FastAPI + React + WebSocket stack this aligns with.
- [ADR-0008](ADR-0008-rrf-as-runtime-fallback.md) - RRF as the runtime fusion fallback used by the MVP.
