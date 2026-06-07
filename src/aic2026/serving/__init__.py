# Implements SPEC-0026 (MVP serving API).
"""The MVP serving API package.

A thin FastAPI + WebSocket service (ADR-0004) that wraps the merged SPEC-0006
`MilvusKeyframeStore` / `MilvusBackend` to turn a Vietnamese KIS text query into
a ranked list of keyframes, serves the keyframe thumbnail + full images as
static files (ADR-0015), captures in-UI issue reports to GitHub with a local
fallback, and reports health / readiness. KIS only; QA / TRAKE / Ad-hoc / the
automatic agent track / C2 learned fusion are out of scope.

`pymilvus` and `fastapi` are imported lazily inside the modules that need them
so this package imports cleanly on a box without the `index` / `serving`
extras (mirrors `aic2026.index`).
"""
