---
id: SPEC-0014
title: C1 - DiacriticBERT diacritic-robust retrieval head (noise, corpus, training, eval)
status: Draft
owner: unassigned
created: 2026-05-30
updated: 2026-05-30
implements_proposal: docs/proposals/08-original-contributions.md (3)
related_adrs:
  - ADR-0001
  - ADR-0007
  - ADR-0011
depends_on:
  - SPEC-0001
  - SPEC-0022
---

# SPEC-0014 - C1 DiacriticBERT: a diacritic-robust Vietnamese retrieval head

> Trains a small late-interaction projection head over a **frozen BGE-M3** so retrieval survives the systematic diacritic corruption that Vietnamese ASR (PhoWhisper) and OCR (PaddleOCR) introduce. This is primary Edge contribution **C1** ([ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md); proposal 08 part 3). The reason this spec can ship before June 25: the head's training data is **generated** by applying a controlled diacritic-noise schedule to *any* clean Vietnamese text - it does **not** require the competition corpus. So the full pipeline (noise -> corpus -> train -> eval) is buildable now and a real baseline can be trained this lease, on **public Vietnamese caption + ASR text**, on the H200.

## 1. Context

[`docs/proposals/08-original-contributions.md`](../proposals/08-original-contributions.md) part 3 specifies C1: Vietnamese ASR/OCR make systematic diacritic errors (`"cho cho"` for `"con cho"`, dropped tones, tone swaps), and off-the-shelf BGE-M3 - trained on clean Vietnamese - has no model of that noise, so its similarity collapses. This is the failure mode called out as item 3 of [`docs/strategy/00-master-strategy.md`](../strategy/00-master-strategy.md) section 7 and catalogued in [`docs/research-notes/05-baseline-2025-analysis.md`](../research-notes/05-baseline-2025-analysis.md) section 4.1 (the 2025 Elasticsearch `asciifolding` analyser strips diacritics outright). [ADR-0007](../adr/ADR-0007-original-contributions-c1-c2-c4.md) makes C1 a **primary** contribution; expected lift **+2-5% R@1** on OCR/ASR-bridged queries.

**Why this is trainable now (and C2/C4 are not).** C2 learned fusion (SPEC-0015) needs ranked lists from a live backend over the dev set; C4 self-distillation (SPEC-0016) needs operator traces. Both are blocked on the June-25 data and a working system. C1 is different: its only data requirement is *clean Vietnamese strings to corrupt*. Pre-June-25 we draw those from **public** Vietnamese datasets; post-June-25 we re-run the **identical** pipeline over our own index text (Qwen2.5-VL captions + cleaned PhoWhisper output, as proposal 08 part 3.2 step 2 describes). This is the same "synthetic-now / real-corpus-later" split that [SPEC-0004](SPEC-0004-image-embedding-service.md) uses for embeddings.

**Boundaries with neighbouring specs.** The trained head emits a ColBERT-style MaxSim score that becomes one ranker lane. Learning the *weight* of that lane inside the OCR/ASR/caption fusion is **C2 (SPEC-0015)**, not this spec. Serving the head online (query-time, on the RTX 5070) is a future online-encoder spec. SPEC-0014 produces an **offline** artefact: a trained checkpoint, the contrastive corpus it was trained on, and an offline robustness eval.

## 2. Scope

### 2.1 In scope

- **`aic2026.train.diacritic_noise`** - the controlled noise function: 4 modes (`drop_all`, `random_drop`, `tone_swap`, `mixed`). Pure, deterministic under a seed, fuzz-tested, never raises on arbitrary unicode.
- **`aic2026.train.diacritic_corpus`** - a builder that harvests clean Vietnamese strings from a configurable set of public HF datasets (default: KTVIC + UIT-OpenViIC captions, VIVOS + Bud500 ASR transcripts), normalises + dedups them, generates K=4 noisy variants each, mines hard negatives, and writes `data/diacritic_pairs.parquet`. **Fault-tolerant per source** (skip-on-failure, exactly like the `cache-weights` job).
- **`aic2026.train.diacritic_bert`** - frozen BGE-M3 + a 2-layer projection head per side + ColBERT-style MaxSim scoring; **InfoNCE** (temperature 0.05) over in-batch + mined hard negatives. **Trains the head only** (backbone frozen). Lazy-imports torch. Writes a checkpoint + `train_meta.json` (provenance: backbone id, input dim read from the model, step count, final loss, the corpus's source manifest).
- A **`train-c1` remote job** (registered for `bin/remote run`) that wraps corpus-build -> train -> eval and banks the checkpoint + parquet + metrics to R2 ([SPEC-0022](SPEC-0022-remote-gpu-runner.md) / [ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md) pattern).
- **`aic2026.eval.diacritic_robustness`** - the synthetic noise-sweep eval (proposal 05 part 13.1): `degradation@10 = R@10(noisy) / R@10(clean)` over a held-out clean->noisy set, against a **pluggable** encoder (`DummyEmbedder` for CI; the trained head on the box).
- **`bin/train`** CLI (Typer; mirrors `bin/embed`) with `c1-corpus`, `c1-fit`, `c1-eval` subcommands.
- A `[project.optional-dependencies] train` extra (`torch`, `transformers`, `sentencepiece`, `datasets`, `pyarrow`, BGE-M3 loader). **CI installs none of it** - the CPU path uses `DummyEmbedder` + a fixture corpus.

### 2.2 Out of scope

- **Optional LoRA r=8 on BGE-M3's last 4 layers** (proposal 08 part 3.2 step 3: "optional and ablated"). Head-only first; LoRA is a follow-up ablation under this same spec.
- **The real-task slice eval** (proposal 05 part 13.2: R@1/R@5/NDCG@10 on the 300-task dev set, tagged for proper-nouns / scene-text). Needs the dev set + a real retrieval backend -> post-June-25, gated by SPEC-0015 + SPEC-0006/0007.
- **Wiring the MaxSim lane weight into runtime fusion** - that is C2 (SPEC-0015).
- **Online (query-time) serving** of the head on the RTX 5070, and INT4/FP4 quantisation (ADR-0006). This spec is offline train + offline eval only.
- **The SeaLLMs-v3 query-rewrite fallback** (proposal 08 part 3.6) - a separate cheap path shipped only if C1 fails its gate; not built here.

## 3. API contract / interface

```python
# aic2026/train/diacritic_noise.py
import random
from enum import Enum

class NoiseMode(str, Enum):
    DROP_ALL = "drop_all"        # "tre em" <- strip every diacritic + tone mark
    RANDOM_DROP = "random_drop"  # drop each mark w.p. p ~ Beta(2, 5) sampled per string
    TONE_SWAP = "tone_swap"      # swap one tone mark for another valid Vietnamese tone
    MIXED = "mixed"              # random_drop composed with tone_swap

def noise(text: str, mode: NoiseMode, *, rng: random.Random) -> str:
    """Apply one diacritic-noise mode to a clean Vietnamese string.

    Pure and deterministic given (text, mode, rng state). Never raises on
    arbitrary unicode; non-Vietnamese characters pass through unchanged.
    """

def variants(text: str, *, k: int = 4, seed: int) -> list[tuple[str, NoiseMode]]:
    """K noisy variants of `text` (cycles the modes). Deterministic per (text, seed)."""
```

```python
# aic2026/train/diacritic_corpus.py
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

@dataclass(frozen=True)
class SourceSpec:
    hf_id: str                      # e.g. "ai-enthusiasm-community/KTVIC" (verify on box)
    text_fields: tuple[str, ...]    # columns to harvest clean strings from
    split: str = "train"

DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    # captions (life-domain) + ASR transcripts; the captions_asr starting mix
    SourceSpec("ai-enthusiasm-community/KTVIC", ("caption", "segment_caption")),
    SourceSpec("<uit-openviic-hf-id>",          ("caption",)),          # Q3: confirm id
    SourceSpec("vivos",                          ("sentence",)),
    SourceSpec("linhtran92/viet_bud500",         ("transcription",)),
)

@dataclass
class CorpusResult:
    n_clean: int
    n_pairs: int
    out: Path                       # data/diacritic_pairs.parquet
    sources_used: list[str]
    sources_skipped: list[str]      # sources that errored on load (non-fatal)

def build_corpus(
    sources: Sequence[SourceSpec] = DEFAULT_SOURCES,
    *,
    out: Path,
    k: int = 4,
    max_per_source: int | None = None,
    hard_negatives: int = 7,
    seed: int = 0,
    clean_strings: Sequence[str] | None = None,   # bypass HF (tests + index-text path)
) -> CorpusResult:
    """Harvest clean Vietnamese strings -> normalise + dedup -> K noisy variants ->
    mine hard negatives -> Parquet with columns
    {anchor_clean, positive_noisy, mode, hard_negs}.

    Any source that raises on load is recorded in `sources_skipped` and the build
    continues. `clean_strings` overrides HF loading entirely - used by CI fixtures
    and (post-June-25) by the path that feeds our own index text in.
    """
```

```python
# aic2026/train/diacritic_bert.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    backbone: str = "BAAI/bge-m3"
    proj_dims: tuple[int, int] = (384, 384)   # head: in_dim (read from model) -> 384 -> 384
    temperature: float = 0.05
    batch_size: int = 256
    max_steps: int = 250_000
    lr: float = 2e-4
    hard_negs: int = 7

@dataclass
class TrainResult:
    checkpoint: Path           # head weights + serialised TrainConfig
    meta: Path                 # train_meta.json: backbone, in_dim, steps, final_loss, corpus manifest
    final_loss: float

def train_diacritic_head(
    pairs: Path, cfg: TrainConfig, *, out_dir: Path, device: str | None = None
) -> TrainResult:
    """Frozen-backbone InfoNCE training of the projection head.

    Backbone params have requires_grad=False (asserted); only the per-side MLP
    trains. `in_dim` is READ FROM the loaded backbone's hidden size, never
    hardcoded - see Open Question Q1 and the SPEC-0004 1024->1152 lesson.
    """
```

```python
# aic2026/eval/diacritic_robustness.py
from collections.abc import Sequence
from aic2026.embedding.base import Embedder
from aic2026.train.diacritic_noise import NoiseMode

def degradation_at_k(
    clean_queries: Sequence[str],
    encoder: Embedder,
    *,
    k: int = 10,
    modes: Sequence[NoiseMode] = (NoiseMode.DROP_ALL, NoiseMode.RANDOM_DROP,
                                  NoiseMode.TONE_SWAP, NoiseMode.MIXED),
    seed: int = 0,
) -> dict[str, float]:
    """For each clean query, build noisy variants, retrieve top-k against the index
    of clean targets, and return degradation@k = mean R@k(noisy) / R@k(clean),
    overall and per mode. Pure and deterministic given (queries, encoder, seed)."""
```

```
bin/train c1-corpus --out data/diacritic_pairs.parquet [--source ID:field ...] [--max-per-source N] [--seed 0]
bin/train c1-fit    --pairs data/diacritic_pairs.parquet --out-dir runs/c1 [--backbone BAAI/bge-m3] [--max-steps N]
bin/train c1-eval   --checkpoint runs/c1 --queries eval/diacritic_dev.txt [--k 10]
```

## 4. Behaviour

- **Noise modes.** `DROP_ALL`: NFD-decompose, drop combining diacritic + tone marks, recompose -> idempotent. `RANDOM_DROP`: sample `p ~ Beta(2, 5)` per string, drop each mark with probability `p`. `TONE_SWAP`: replace one Vietnamese tone diacritic with another valid tone. `MIXED`: `RANDOM_DROP` then `TONE_SWAP`. ASCII / non-Vietnamese characters pass through unchanged. Same `(text, mode, rng)` -> same output; arbitrary unicode never raises.
- **Corpus.** Clean strings are deduped (exact + case/space-normalised). Each clean string yields `k` variants. Hard negatives = top-N BGE-M3 neighbours of the clean anchor minus the positive set **when a real encoder is available**; in CI / fixture mode (`clean_strings` set, no torch) hard-neg mining falls back to random in-corpus strings. A source that fails to load is skipped, not fatal. The Parquet schema is stable and documented in the module docstring.
- **Training.** Backbone is frozen (no backbone parameter has `requires_grad=True` - asserted at startup). The head trains under InfoNCE (temp 0.05) over in-batch + mined hard negatives. A checkpoint and `train_meta.json` are written; `in_dim` in the meta is read from the model config. Runs a few steps on CPU over a fixture (the AC3 test) and the full corpus on the H200.
- **Eval.** `degradation@10` is in `[0, 1]`. proposal 05 part 13.1 targets: **with C1 on, `>= 0.85`**; BGE-M3-only baseline expected `~0.65-0.75`. Pure / deterministic.
- **`train-c1` remote job.** `--dry-run` prints the plan (corpus -> fit -> eval + the artefact upload) without executing. A real run writes a `RunContext`, appends a manifest entry, and uploads the checkpoint + parquet + metrics to R2.

## 5. Acceptance criteria

- **AC1**: each `NoiseMode` returns valid output; `DROP_ALL` leaves **no** combining marks and is idempotent; `noise()` is deterministic per `(text, mode, seed)` and never raises across a fuzz corpus of arbitrary unicode (including emoji, CJK, control chars). Verified in `tests/unit/test_diacritic_noise_AC1.py`.
- **AC2**: `build_corpus(clean_strings=<fixture>, k=4)` writes a Parquet with the documented columns, exactly 4 positives per clean anchor, deduped, and populates `sources_used` / `sources_skipped`; a source whose loader raises is skipped while the result is still written. CPU-only, **no network** (uses the `clean_strings` override). Verified in `tests/unit/test_diacritic_corpus_AC2.py`.
- **AC3**: `train_diacritic_head` on a tiny fixture Parquet for a few steps (a) keeps the backbone frozen (no backbone grad), (b) reduces the loss (`final_loss < initial_loss`), and (c) writes a checkpoint + `train_meta.json` whose `in_dim` was read from the model. Uses `pytest.importorskip("torch")`; **skipped in CI**. Verified in `tests/unit/test_diacritic_bert_AC3.py`.
- **AC4**: `degradation_at_k` with `DummyEmbedder` returns overall + per-mode values in `[0, 1]`, deterministic per seed, and returns `1.0` when noise is a no-op (identical text, large `k`). Verified in `tests/unit/test_diacritic_robustness_AC4.py`.
- **AC5**: the `train-c1` job is registered and `bin/remote run train-c1 --dry-run` plans corpus -> fit -> eval and the artefact upload without executing; a real run against mocked R2 appends a manifest entry. Verified in `tests/unit/test_remote_train_c1_AC5.py` (moto).
- **AC6**: this spec and the module docstrings document (a) the public-corpus-now vs index-text-later split, (b) the deferred LoRA + real-task-slice + fusion-weight wiring, and (c) the read-`in_dim`-from-model rule. Verified by inspection (no test).

## 6. Non-functional requirements

- **Determinism**: every stochastic step is seeded (noise RNG, variant cycling, hard-neg sampling, training data order). `noise()` is `O(len)`.
- **CI**: CPU-only, no network, no torch. The corpus builder runs via the `clean_strings` override; the eval runs against `DummyEmbedder`; training is `importorskip`-gated.
- **H200 full-run budget**: proposal 08 part 3.2 step 5 estimates ~3 GPU-hours for 250k steps at batch 256 on a single A6000; on one H200 it should be well under that, and the corpus build (load + noise a few hundred thousand strings + one BGE-M3 embed pass for hard negs) is minutes. Comfortably inside the 2-day lease.
- **Compatibility**: Python 3.11+, `torch >= 2.4`, `transformers >= 4.45`, `sentencepiece`, `datasets >= 2.19`, `pyarrow`. BGE-M3 = `BAAI/bge-m3`; add it to the `cache-weights` `DEFAULT_FLOOR_REPOS` so it is warm on the box (one-line follow-up to spec/0023 - BGE-M3 is the only floor *text* encoder, currently uncached).

## 7. Dependencies

- **Internal**: SPEC-0001 (provides the `Embedder` pattern reused by the eval and the metric conventions), SPEC-0022 (remote runner + R2 for the `train-c1` job), SPEC-0004 (the `Embedder` protocol + optional-extra packaging precedent + the read-dim-from-model lesson).
- **External**: HF `datasets` (public sources), `transformers` / BGE-M3 loader, `torch`, `pyarrow`.
- **Data**: public HF datasets for the real run (KTVIC, UIT-OpenViIC, VIVOS, Bud500 - ids verified on the box, builder is fault-tolerant); a tiny in-repo clean-string fixture for CI.

## 8. Test plan

- **Unit tests** (`tests/unit/`, all CPU-only / offline):
  - `test_diacritic_noise_AC1.py` - per-mode output, `DROP_ALL` completeness + idempotence, determinism, unicode fuzz.
  - `test_diacritic_corpus_AC2.py` - fixture `clean_strings`, column schema, 4-per-anchor, dedup, skip-on-source-failure.
  - `test_diacritic_bert_AC3.py` - `importorskip("torch")`; frozen backbone, loss decreases, meta written (skipped in CI).
  - `test_diacritic_robustness_AC4.py` - `DummyEmbedder`, value ranges, determinism, no-op == 1.0.
  - `test_remote_train_c1_AC5.py` - job registered; dry-run plan; mocked-R2 manifest append (moto).
- **GPU smoke on the box** (records evidence for AC3/AC4 on real BGE-M3):
  - `./bin/train c1-corpus --max-per-source 2000 --out /tmp/pairs.parquet`
  - `./bin/train c1-fit --pairs /tmp/pairs.parquet --out-dir /tmp/c1 --max-steps 2000`
  - `./bin/train c1-eval --checkpoint /tmp/c1 --queries eval/diacritic_dev.txt` -> degradation@10 with the head vs a BGE-M3-only baseline.

## 9. Open questions

- **Q1 (blocks implementation)**: BGE-M3 head input width. proposal 08 part 3.2 step 3 writes the head as `768 -> 384 -> 384`, but BGE-M3 (XLM-RoBERTa-large) has hidden size **1024** and also ships a *native ColBERT multi-vector head*. Resolve at implementation by **reading the hidden size off the loaded model config** (never hardcode - the SPEC-0004 `1024 -> 1152` bug is the cautionary precedent), and decide whether to project BGE-M3's native ColBERT token vectors or its last-hidden-state token embeddings. Recommendation: project last-hidden-state token embeddings; benchmark against the native ColBERT vectors as an ablation.
- **Q2**: noise-schedule realism. The `Beta(2, 5)` drop rate is an assumption. proposal 04 part 6 (risk row) says validate it against a ~200-sample slice of real PhoWhisper / PaddleOCR output and retune. That real slice does not exist until we have ASR/OCR over the corpus (post-June-25). Until then, sanity-check the noise visually against a handful of known PhoWhisper error examples and record the assumption in `train_meta.json`.
- **Q3**: dataset-id verification. The KTVIC / UIT-OpenViIC / VIVOS / Bud500 HF ids must be confirmed on the box (the `DEFAULT_SOURCES` list holds best-known ids; `<uit-openviic-hf-id>` is an explicit placeholder). The builder is fault-tolerant, so an unavailable source degrades gracefully. The exact datasets that land in corpus v1 are recorded in `train_meta.json` for provenance.
- **Q4**: hard-negative mining requires a BGE-M3 forward pass over the clean side (a one-time embed, minutes on the H200 for a few hundred thousand strings). CI / fixture mode falls back to random in-corpus negatives. Confirm in an ablation that mined negatives beat random by enough to justify the extra pass.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-30 | implementer (user-directed) | Created at status **Draft** for review (user chose spec-first). Scopes C1 as a now-trainable contribution on **public** Vietnamese caption + ASR text (KTVIC + UIT-OpenViIC + VIVOS + Bud500), with an identical post-June-25 re-run over our own index text. Implementation is gated on user approval of this spec. |
