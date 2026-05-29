---
id: SPEC-0004
title: Image-embedding service (SigLIP-2 + Meta CLIP 2 + InternVideo2-1B)
status: Implementing
owner: unassigned
created: 2026-05-29
updated: 2026-05-29
implements_proposal: docs/proposals/01-interactive-system-architecture.md SS 5.3
related_adrs:
  - ADR-0003
  - ADR-0006
  - ADR-0007
depends_on:
  - SPEC-0001
---

# SPEC-0004 - Image-embedding service (SigLIP-2 + Meta CLIP 2 + InternVideo2-1B)

> The component that turns keyframes (offline) and Vietnamese queries (online) into L2-normalised dense vectors for the three image-text encoders that form the floor of our retrieval ensemble. Produces vectors only; storage / ANN / fusion lives in SPEC-0006 and SPEC-0015.

## 1. Context

[`docs/proposals/01-interactive-system-architecture.md`](../proposals/01-interactive-system-architecture.md) SS 5.3 names the three encoders that form the floor of the retrieval ensemble: **SigLIP-2 So400m/16@384** (primary, 1024-d), **Meta CLIP 2 ViT-H/14** (Vietnamese-capable, 1024-d), and **InternVideo2-Stage2_1B-224p-f4** (video, 768-d). [ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md) splits the workload: **image towers run offline on the GH200** and produce pre-indexed vectors; **only text towers run online** on the 12 GB RTX 5070 (query encoding). [ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md) makes the floor a precondition for C1/C2/C4 - none of the original-contribution ablations can run against a real backend until embeddings exist.

Today the harness exercises a `StubBackend` only (SPEC-0001 Tier 1). This spec adds the embedding layer; storage and query (Milvus) live in SPEC-0006; ranker composition into a `Backend.search` impl is SPEC-0015.

The full dataset releases on June 25, 2026. Per [`docs/research-notes/06-aic2026-dataset-shape.md`](../research-notes/06-aic2026-dataset-shape.md) SS 2.3 organisers ship a pre-extracted-but-weak CLIP that we plan to keep as a fourth lane in fusion - **our own three encoders are what gives us the Edge**. The slice in this spec runs on synthetic / sample images so it can be developed before June 25.

## 2. Scope

### 2.1 In scope
- A single `Embedder` Protocol with `encode_text(list[str]) -> ndarray` and `encode_image(list[Path]) -> ndarray`. Vectors are `float32`, `(n, dim)`, L2-normalised.
- Concrete implementations:
  - **`DummyEmbedder`** - deterministic, numpy-only, CPU-only. The default for CI and for tests across the rest of the package.
  - **`SigLip2Embedder`** - real SigLIP-2 So400m/16@384, 1024-d. Lazy-imports `torch` + `open_clip` inside methods; gated behind a `[project.optional-dependencies] embedding` extra.
- An offline extraction module `aic2026.embedding.extract` that batches a directory of images through an `Embedder` and writes:
  - `<out>.npy` - a `float32` matrix of shape `(n, dim)`
  - `<out>.manifest.jsonl` - one row per vector with `{row, frame_id, path}`, byte-for-byte aligned with the matrix.
- `bin/embed` CLI (Typer; mirrors `bin/eval`) with two subcommands: `images` (offline extraction) and `text` (one-shot debug encode).
- Quality gate: per-row L2 norm within `1.0 +- 1e-3`.

### 2.2 Out of scope
- **Meta CLIP 2 ViT-H/14** and **InternVideo2-Stage2_1B-224p-f4** concrete encoders. The Protocol is sized for all three (`dim` is per-encoder); follow-up PRs slot them in.
- **Milvus schema and ingestion** (SPEC-0006). The extraction CLI writes `.npy + .jsonl`; loading those into Milvus is SPEC-0006's job.
- The organisers' pre-computed CLIP as a fourth lane (SPEC-0006).
- `EmbeddingBackend.search` composition into a `harness.Backend` (SPEC-0015 - it consumes embeddings + Milvus + fusion).
- INT4 / FP4 **quantisation** of the text tower for the RTX 5070 finals deployment (ADR-0006; the slice runs FP16 on GH200 / CPU only).
- Real-dataset wiring (the AIC2026 corpus lands June 25; the slice runs on synthetic / sample files only).
- Production batching, sharding, retry, error-budget logic. The extractor is a single-process sequential walker.

## 3. API contract / interface

```python
# aic2026/embedding/base.py

from pathlib import Path
from typing import Protocol
import numpy as np

class Embedder(Protocol):
    """A dense bi-encoder for image-text retrieval.

    Implementations may run on GPU; they must not require network access at
    call time. The two `encode_*` methods are independent: image-tower work
    is offline (GH200), text-tower work is online (RTX 5070) - see ADR-0003.
    """

    model_id: str            # stable identifier; e.g. "siglip2-so400m-p16-384"
    dim: int                 # output dimensionality; e.g. 1024 for SigLIP-2

    def encode_text(self, texts: list[str]) -> np.ndarray: ...
    def encode_image(self, paths: list[Path]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation. Returns float32, shape unchanged."""
```

```python
# aic2026/embedding/extract.py

from dataclasses import dataclass
from pathlib import Path
from aic2026.embedding.base import Embedder

@dataclass
class ExtractionResult:
    n: int
    dim: int
    vectors_path: Path     # *.npy
    manifest_path: Path    # *.manifest.jsonl

def extract_image_embeddings(
    paths: list[Path],
    embedder: Embedder,
    *,
    out: Path,
    batch_size: int = 32,
) -> ExtractionResult: ...
```

```
bin/embed images --input DIR --output PATH [--encoder dummy|siglip2] [--dim 1024] [--batch-size 32]
bin/embed text   --text "..."             [--encoder dummy|siglip2] [--dim 1024]
```

## 4. Behaviour

- **Normal extraction**: walk `--input` for `*.jpg|*.jpeg|*.png|*.webp` sorted by path (deterministic), batch through `Embedder.encode_image`, accumulate to a `float32 (n, dim)` matrix, write `.npy`; write `.manifest.jsonl` whose row `i` describes the `i`-th vector with `frame_id = path.stem` and `path = str(path)`.
- **Empty input**: exit 0; write a zero-row `.npy` and an empty manifest.
- **Encoder unknown**: exit non-zero with a clear error listing the registry's known encoders.
- **Encoder needs deps but extras aren't installed**: import error surfaces *with* a hint to `uv sync --extra embedding`.
- **Quality**: each row's L2 norm is within `1.0 +- 1e-3` (asserted by `extract_image_embeddings`).
- **Determinism**: with `DummyEmbedder`, two runs over the same input directory produce byte-identical `.npy` and `.manifest.jsonl`.
- **Offline / online split**: `encode_image` is documented offline-only (ADR-0003 SS Decision); the online code path on RTX 5070 only calls `encode_text`. The slice ships both methods on every encoder so a single class is callable in either mode; the split is enforced by deployment, not by a runtime guard.

## 5. Acceptance criteria

- **AC1**: For any `Embedder`, `encode_text(list-of-length-n)` and `encode_image(list-of-length-n)` return a `float32` `(n, dim)` array whose rows have L2 norm `1.0 +- 1e-3` and whose `dim` matches the encoder's `dim`. Verified in `tests/unit/test_embedding_dummy_AC1.py`.
- **AC2**: `DummyEmbedder` is deterministic per `(model_id, dim, input)` and distinct inputs yield distinct vectors (no collisions on the unit test inputs). Verified in `tests/unit/test_embedding_dummy_AC2.py`.
- **AC3**: `extract_image_embeddings` over a directory of `n` fake files yields `.npy` with shape `(n, dim)` and `.manifest.jsonl` whose row count and `frame_id`s match, in deterministic sort order. Re-running on the same input produces byte-identical outputs. Verified in `tests/unit/test_embedding_extract_AC3.py`.
- **AC4**: `SigLip2Embedder` lazy-loads its heavy deps (no `import torch` at module import time) and, when those deps are present, returns `1024`-d L2-normalised vectors for both `encode_text` and `encode_image`. The test uses `pytest.importorskip("torch")` and skips in CI (the `embedding` extra is not installed in CI by design). Verified in `tests/unit/test_embedding_siglip2_AC4.py`.
- **AC5**: The spec and module docstrings document the offline-only nature of `encode_image` (ADR-0003) and the optional-extra packaging of the heavy backbones (ADR-0006). Verified by inspection (no test).

## 6. Non-functional requirements

- **Latency** (slice): `DummyEmbedder.encode_text([q])` <= 5 ms on CPU; extraction over 100 fake files <= 1 s on CPU. SigLIP-2 latency targets land with the GH200 / 5070 deploy spec, not here.
- **Memory**: `DummyEmbedder` allocates `n * dim * 4` bytes; the extractor streams batches and does not pin the full corpus in memory before write-out (we still build the matrix in-process for the slice; a streamed-write variant lands when `n` exceeds RAM, deferred).
- **Determinism**: SHA-256-seeded numpy RNG inside `DummyEmbedder`. No global RNG state.
- **Compatibility**: Python 3.11+, NumPy >= 1.26. SigLIP-2 backbone: `torch >= 2.4`, `open-clip-torch` or `transformers`, `pillow`. CI installs neither - they are in the `embedding` extra.

## 7. Dependencies

- **Internal**: SPEC-0001 (`Submission`/`MockTask` are not consumed here; the dependency is the harness scaffolding + the CI gate from SPEC-0021).
- **External**:
  - Core (always installed): `numpy >= 1.26`.
  - Optional `embedding` extra: `torch >= 2.4`, `open-clip-torch >= 2.32` (or `transformers >= 4.45`), `pillow >= 10`.
- **Data**: synthetic / fake files in `tmp_path` for the slice tests. Real corpus integration is post-June-25 and tracked by a follow-up under SPEC-0006.

## 8. Test plan

- **Unit tests** (`tests/unit/`):
  - `test_embedding_dummy_AC1.py` - shape, dtype, dim, L2 norm.
  - `test_embedding_dummy_AC2.py` - deterministic + input-distinct.
  - `test_embedding_extract_AC3.py` - extract over `tmp_path` fake files, manifest alignment, run-to-run byte-equality.
  - `test_embedding_siglip2_AC4.py` - `pytest.importorskip("torch")`; asserts 1024-d normalised on a synthetic image when deps are present (skipped in CI).
- **CLI smoke** (local mirror of the CI smoke):
  - `./bin/embed images --input <tmpdir> --output <tmp>/v --encoder dummy --dim 64` produces `<tmp>/v.npy` and `<tmp>/v.manifest.jsonl`; exit 0.
  - `./bin/embed text --text "hello" --encoder dummy --dim 64` prints a 64-d normalised vector; exit 0.

## 9. Open questions

- **Q1**: Which library backs `SigLip2Embedder` - `open-clip-torch` or HF `transformers`? Either works; `open-clip-torch` has the canonical SigLIP-2 weights. Decision deferred to the follow-up PR that actually ships SigLIP-2 inference; the slice loads-and-runs against whichever the implementer wires.
- **Q2**: Streamed-write for extractions larger than RAM. Out of scope for the slice; gated by the real-dataset run-size on June 25.
- **Q3**: How to wire the organisers' weak CLIP as a 4th encoder lane - happens in SPEC-0006 (Milvus schema) and SPEC-0015 (fusion), not here. Flagged so it isn't accidentally pulled into SPEC-0004.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-29 | implementer (user-directed) | Created; Draft -> Approved -> Implementing in one pass for solo work per CONTRIBUTING. Slice (DummyEmbedder + SigLip2Embedder + extraction CLI) ships in branch `spec/0004-image-embedding-service`; Meta CLIP 2 + InternVideo2 land in follow-up PRs against this spec. |
