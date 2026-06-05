---
id: ADR-0015
title: Keyframe images are served as a pre-generated thumbnail tier banked to R2
status: Proposed
decided_on: 2026-06-05
deciders:
  - team lead
related_adrs:
  - ADR-0004
  - ADR-0011
  - ADR-0013
---

# ADR-0015 - Keyframe images are served as a pre-generated thumbnail tier banked to R2

## Status

Proposed.

## Context

The MVP UI (SPEC-0027) shows a grid of ranked keyframes; the operator scans
images, not text. The AIC2025 proxy corpus is 121,457 keyframe JPGs totalling
~19 GB at full resolution. The grid renders thumbnails at ~240x135 px
(proposal 06 SS 2), so shipping full-resolution JPGs to the browser for a
48-thumbnail page would move tens of MB per query - the exact image-bytes cost
[ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md) already ruled must not
travel through Python and must come off a static server.

ADR-0004 fixes the serving mechanism ("nginx serving static JPEG thumbnails").
It does not say where the thumbnails come from, nor how the ~19 GB is hydrated
onto a rebuildable-from-R2 shared server (ADR-0013). Three sourcing strategies
exist: (1) generate thumbnails on the fly per request; (2) generate them once at
provision time on the server; (3) generate them offline on the lease box and
bank them to R2 as a first-class tier next to the indexes. The corpus is fixed
(it changes only when a new index build lands), and ADR-0011 already makes R2 the
source of truth for derived artifacts.

## Decision

Thumbnails are a **pre-generated tier banked to R2** alongside the indexes:
an offline pass produces a downsized JPEG per keyframe (long edge ~320 px,
quality ~75) and uploads it under an R2 thumbnail prefix
(`thumbs/aic2025-proxy-3enc-20260604/<video_id>/<frame_id>.jpg`), keyed by the
same `video_id` / `frame_id` the SPEC-0006 store uses. At provision the shared
server hydrates the thumbnail tier (and, separately, the full-resolution JPGs)
from R2 to local disk; both are served as **static files** (nginx in production,
a dev static route otherwise) so image bytes never pass through the FastAPI
process ([ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md)). The grid uses
thumbnails; the frame-detail view uses the full-resolution JPG. R2 stays the
source of truth (ADR-0011); the local image trees are a rebuildable cache.

## Consequences

### Positive

- Grid payload drops from full-res (~150 KB+ each) to thumbnails (~10-20 KB
  each), keeping a 48-thumbnail page small and the grid render within the
  proposal 06 SS 9 budget.
- Thumbnails are built once offline (on the GH200/lease box per ADR-0003), not
  recomputed per request or per provision; provisioning is a download, not a CPU
  job on the shared box.
- Same R2-as-truth discipline as the indexes (ADR-0011): a rebuilt server pulls
  the thumbnail tier the same way it pulls the indexes.

### Negative

- Adds a derived tier to maintain: a new index build (new keyframes) requires a
  matching thumbnail pass before serving, or the grid shows gaps.
- Stores the corpus twice in R2 (full JPGs + thumbnails); at ~19 GB full and an
  estimated ~1-3 GB of thumbnails this is a modest storage increase at R2 rates
  (ADR-0011).

### Neutral / observable

- The thumbnail key scheme is bound to the SPEC-0006 `video_id`/`frame_id`
  identity, so the serving layer maps a ranked `pk` to an image URL with no
  lookup table.
- The offline thumbnail pass is a new provisioning/runner step (an
  `infra/remote/` job, following the existing pattern); SPEC-0026 owns the
  provision-time hydration contract.

## Alternatives considered

- **On-the-fly thumbnailing** (resize per request with Pillow) - no extra
  storage, no offline pass - rejected because it puts image bytes and CPU resize
  cost on the request path, contradicting ADR-0004, and repeats work for the
  fixed corpus on every view.
- **Generate thumbnails at provision time on the shared server** - no second R2
  tier - rejected because it makes every provision a CPU-heavy 121k-image resize
  job on the serving box instead of a download, slowing rebuilds (ADR-0013).
- **Serve full-resolution JPGs to the grid and let the browser downscale** -
  simplest, no thumbnail tier - rejected because it moves ~19 GB-scale bytes for
  large result sets and blows the grid render budget.

## References

- [ADR-0004](ADR-0004-no-streamlit-react-websocket-ui.md) - nginx static image serving; image bytes never through Python.
- [ADR-0011](ADR-0011-r2-artifact-store-and-lease-rollover.md) - R2 as the artifact source of truth.
- [ADR-0013](ADR-0013-mvp-single-shared-server-from-r2.md) - the shared server hydrated from R2.
- [SPEC-0026](../specs/SPEC-0026-mvp-serving-api.md) - the image-serving endpoints and provision contract.
- proposal 06 SS 2, SS 9 - thumbnail sizing and the grid performance budget.
