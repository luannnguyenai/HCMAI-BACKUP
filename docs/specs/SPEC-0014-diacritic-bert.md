---
id: SPEC-0014
title: C1 - DiacriticBERT diacritic-robust retrieval head (noise, corpus, training, eval)
status: Implementing
owner: unassigned
created: 2026-05-30
updated: 2026-06-01
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

- **`aic2026.train.diacritic_noise`** - the controlled noise function. **v1 (diacritic-focused)**: `drop_all`, `random_drop`, `tone_swap`, `mixed`. **v2 (OCR-character)**: `space_split` (over-segmentation: `quả táo` -> `q u ả t á o`), `char_confuse` (visual misreads: `5<->S`, `1<->l/I`, `0<->O`, `6<->G/b`, `8<->B`, `2<->Z/z`, `9<->g/q`, `7<->T`, `4<->A/h`, plus the two-char pairs `rn<->m` and `cl<->d`), and `mixed_ocr` (composes random_drop + char_confuse + space_split for the realistic-worst-case OCR output). **v3 Tier A (Hoang & Aw 2012 Class 2/3)**: `word_merge` (Class 3 under-segmentation, symmetric inverse of `space_split`: `quả táo` -> `quảtáo`), `homophone_swap` (Class 2 ASR real-syllable substitution; vowel-anchored tone re-assignment that can ADD a tone to a level syllable -- the dominant PhoWhisper failure mode that `tone_swap` cannot reach because it is mark-anchored), and `case_noise` (Class 3 case shift; per-word random lower/upper/title). Pure, deterministic under a seed, fuzz-tested, never raises on arbitrary unicode. `variants(text, k=None)` defaults to one variant per `NoiseMode` so every category is represented per training anchor.
- **`aic2026.train.diacritic_corpus`** - a builder that harvests clean Vietnamese strings from a configurable set of public HF datasets (default: KTVIC + UIT-OpenViIC captions, VIVOS + Bud500 ASR transcripts), normalises + dedups them, generates K=4 noisy variants each, mines hard negatives, and writes `data/diacritic_pairs.parquet`. **Fault-tolerant per source** (skip-on-failure, exactly like the `cache-weights` job).
- **`aic2026.train.diacritic_bert`** - frozen BGE-M3 + a 2-layer projection head per side + ColBERT-style MaxSim scoring; **InfoNCE** (temperature 0.05) over in-batch + mined hard negatives. **Trains the head only** (backbone frozen). Lazy-imports torch. Writes a checkpoint + `train_meta.json` (provenance: backbone id, input dim read from the model, step count, final loss, the corpus's source manifest).
- A **`train-c1` remote job** (registered for `bin/remote run`) that wraps corpus-build -> train -> eval and banks the checkpoint + parquet + metrics to R2 ([SPEC-0022](SPEC-0022-remote-gpu-runner.md) / [ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md) pattern).
- **`aic2026.eval.diacritic_robustness`** - the synthetic noise-sweep eval (proposal 05 part 13.1): `degradation@10 = R@10(noisy) / R@10(clean)` over a held-out clean->noisy set. Pluggable retriever (cosine or MaxSim). Includes `build_heldout_queries` (disjoint from the training corpus) and `compare_c1_vs_baselines` (the three-way ship-gate).
- **`aic2026.eval.retrievers`** - the head-as-encoder retrieval surface: `Retriever` Protocol, `DenseRetriever` (cosine over any `Embedder`, including `DummyEmbedder`), `MaxSimRetriever` (BGE-M3 + optional `DiacriticHead`, chunked over queries), `BgeM3DenseEmbedder` (mean-pool baseline), and `load_head` (rebuild a head from `head.pt`).
- A **`eval-c1` remote job** + `infra/remote/c1_eval.sh` runner: restores `head.pt` + `pairs.parquet` from R2 (`c1-baseline/<sha>/`), harvests held-out queries, runs the three-way comparison, and writes `c1_eval.json` for the runner to upload to R2.
- **`bin/train`** CLI (Typer; mirrors `bin/embed`) with `c1-corpus`, `c1-fit`, a `c1-eval` that runs either the `DummyEmbedder` smoke (no checkpoint) or the three-way ship-gate (`--checkpoint`), and a `c1-demo` (SS 6) for live side-by-side comparisons.
- **`aic2026.eval.demo`** (SS 6) - the live demo surface: a curated `CANNED_EXAMPLES` set (12 examples) keyed by `NoiseMode`, weighted toward `mixed_ocr` (5 of 12 - the realistic-worst-case mode C1 wins on) and covering every failure family C1 was built to attack: 2x `drop_all` (short placename + long sentence with a foreign token), 2x `char_confuse` (address + sports score), `word_merge`, `homophone_swap` (the `Mỹ Tâm` -> `Mỹ Tấm` ASR case), and a clean-sanity control kept last. Each non-control example pins a `noise_seed` chosen to produce dramatic-but-readable corruption (so the demo isn't seed-roulette). A side-by-side block formatter marks the gold target with `[TRÚNG]`, a `run_canned` batched runner emits a win/tie/loss tally vs the two baselines, and a `run_interactive` REPL handles audience queries. CPU-testable via injected `FakeRetriever`; GPU-runs via the same three retrievers the ship-gate uses, so what the demo shows and what the eval measures cannot diverge.
- A `[project.optional-dependencies] train` extra (`torch`, `transformers`, `sentencepiece`, `datasets`, `pyarrow`, BGE-M3 loader). **CI installs none of it** - the CPU path uses `DummyEmbedder` + a fixture corpus.

### 2.2 Out of scope

- **Optional LoRA r=8 on BGE-M3's last 4 layers** (proposal 08 part 3.2 step 3: "optional and ablated"). Head-only first; LoRA is a follow-up ablation under this same spec.
- **The real-task slice eval** (proposal 05 part 13.2: R@1/R@5/NDCG@10 on the 300-task dev set, tagged for proper-nouns / scene-text). Needs the dev set + a real retrieval backend -> post-June-25, gated by SPEC-0015 + SPEC-0006/0007.
- **Wiring the MaxSim lane weight into runtime fusion** - that is C2 (SPEC-0015).
- **Online (query-time) serving** of the head on the RTX 5070, and INT4/FP4 quantisation (ADR-0006). This spec is offline train + offline eval only.
- **The SeaLLMs-v3 query-rewrite fallback** (proposal 08 part 3.6) - a separate cheap path shipped only if C1 fails its gate; not built here.
- **Mined-hard-negative training** (Q4): the corpus carries the `hard_negs` column but training v1 uses only in-batch negatives. Adding a head-loss term over the mined negatives is a follow-up under this same spec.
- **Bootstrap CI / significance testing** on the degradation@10 numbers (proposal 05 SS 14.1). The ship-gate as defined is point-estimate; CI bands are a separate, small-scope follow-up.

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
- **Eval.** `degradation@10` is in `[0, 1]`. proposal 05 part 13.1 targets: **with C1 on, `>= 0.85`**; BGE-M3-only baseline expected `~0.65-0.75`. Pure / deterministic. Three retrievers are reported in the same run: **c1_on** = `MaxSimRetriever(BGE-M3, head=trained)`, **baseline_maxsim** = `MaxSimRetriever(BGE-M3, head=None)` (raw last-hidden tokens, isolates the head's lift), **baseline_dense** = `DenseRetriever(BgeM3DenseEmbedder)` (mean-pool single vector, the live-system style). The **ship-gate verdict** = `c1.overall >= 0.85` **AND** `c1.overall > maxsim.overall` **AND** `c1.overall > dense.overall`.
- **Held-out queries.** `build_heldout_queries(n, exclude_corpus=pairs.parquet)` re-harvests the public sources and removes any string whose case/space-normalised form appears in the training Parquet's `anchor_clean` column. Held-out / training disjointness is enforced by construction.
- **`train-c1` remote job.** `--dry-run` prints the plan (corpus -> fit -> eval + the artefact upload) without executing. A real run writes a `RunContext`, appends a manifest entry, and uploads the checkpoint + parquet + metrics to R2.
- **`eval-c1` remote job.** Restores `head.pt` + `pairs.parquet` from R2 `c1-baseline/<sha>/` (or accepts explicit `--config checkpoint=...,pairs_path=...`), builds the held-out query set, runs the three-way comparison, and writes `c1_eval.json` under `ctx.local_run_dir` (uploaded to R2 by the generic runner). Heavy imports are deferred so the registry stays light.

## 5. Acceptance criteria

- **AC1**: each `NoiseMode` returns valid output; `DROP_ALL` leaves **no** combining marks and is idempotent; `noise()` is deterministic per `(text, mode, seed)` and never raises across a fuzz corpus of arbitrary unicode (including emoji, CJK, control chars). Verified in `tests/unit/test_diacritic_noise_AC1.py`.
- **AC2**: `build_corpus(clean_strings=<fixture>, k=4)` writes a Parquet with the documented columns, exactly 4 positives per clean anchor, deduped, and populates `sources_used` / `sources_skipped`; a source whose loader raises is skipped while the result is still written. CPU-only, **no network** (uses the `clean_strings` override). Verified in `tests/unit/test_diacritic_corpus_AC2.py`.
- **AC3**: `train_diacritic_head` on a tiny fixture Parquet for a few steps (a) keeps the backbone frozen (no backbone grad), (b) reduces the loss (`final_loss < initial_loss`), and (c) writes a checkpoint + `train_meta.json` whose `in_dim` was read from the model. Uses `pytest.importorskip("torch")`; **skipped in CI**. Verified in `tests/unit/test_diacritic_bert_AC3.py`.
- **AC4**: `degradation_at_k` with `DummyEmbedder` returns overall + per-mode values in `[0, 1]`, deterministic per seed, and returns `1.0` when noise is a no-op (identical text, large `k`). Verified in `tests/unit/test_diacritic_robustness_AC4.py`.
- **AC5**: the `train-c1` job is registered/resolvable and the registry import path stays free of heavy deps (torch/transformers/datasets imported *inside* the job, so CI can import the jobs package without the `train` extra). Dry-run planning + the R2 artefact upload of `ctx.local_run_dir` are exercised by the generic runner tests (SPEC-0022/0024). Verified in `tests/unit/test_remote_train_c1_AC5.py`.
- **AC6**: this spec and the module docstrings document (a) the public-corpus-now vs index-text-later split, (b) the deferred LoRA + real-task-slice + fusion-weight wiring, and (c) the read-`in_dim`-from-model rule. Verified by inspection (no test).
- **AC7**: `DenseRetriever(Embedder)` score matrix equals `q @ d.T`; `degradation_at_k` accepts a `Retriever` *or* an `Embedder` (back-compat: the AC4 path through `DummyEmbedder` is exactly equivalent to `DenseRetriever(DummyEmbedder)`); `MaxSimRetriever(stub_backbone, head=...)` returns a `(nq, nd)` matrix in `[-1, 1]` and tolerates empty inputs; `load_head(head.pt)` rebuilds an eval-mode, frozen `DiacriticHead` whose forward output is L2-normalised per token. Verified in `tests/unit/test_retrievers_AC7.py` (CPU; torch parts `importorskip`).
- **AC8**: `compare_c1_vs_baselines(...)` returns the three per-retriever blocks (`c1_on` / `baseline_maxsim` / `baseline_dense`) with per-mode + `overall` values in `[0, 1]`, plus a `ship_gate` verdict that is the conjunction of `c1_overall >= target`, `> baseline_maxsim_overall`, and `> baseline_dense_overall`. Deterministic per seed and supports a `target=` override. Verified in `tests/unit/test_compare_c1_AC8.py` (torch + stub backbone; head is randomly initialised - the contract is correctness, not the real numbers).
- **AC9**: the `eval-c1` job is registered/resolvable and the registry import path is free of heavy deps (torch / transformers / `aic2026.eval.diacritic_robustness` / `aic2026.eval.retrievers` / `aic2026.train.diacritic_bert` all imported *inside* the job). Verified in `tests/unit/test_remote_eval_c1_AC9.py`.
- **AC10** (SS 6, live demo): `CannedExample.make_noised` is deterministic and matches the training-time `noise()` distribution; the canned set is >= 10 examples (currently 12), covers `DROP_ALL`, `CHAR_CONFUSE`, `MIXED_OCR`, `WORD_MERGE`, `HOMOPHONE_SWAP`, and exactly one clean-sanity example kept last, with `mixed_ocr` the most-represented mode (>= 3), unique ids, and every non-control example producing visible noise (its noised form differs from the clean target). `format_example_block` marks the gold target with `[TRÚNG]` at the correct rank, reports gold position when it falls outside top-k, and renders an "(không có — đối chứng)" line for clean-sanity examples. `run_canned` returns the correct `{wins, ties, losses}` tally for hand-built `FakeRetriever` score matrices and prints a Vietnamese summary; the doc index combines `doc_pool` with canned targets (deduped, insertion-order preserved) so the gold answer is always indexable. `run_interactive` is EOF-safe (immediate exit on empty query; clean exit on EOF mid-session) and counts queries executed. Verified in `tests/unit/test_eval_demo_AC1.py` (17 tests; CPU-only via `FakeRetriever`, no torch import).

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
  - `test_remote_train_c1_AC5.py` - job registered/resolvable; registry import path free of heavy deps.
  - `test_retrievers_AC7.py` - `DenseRetriever` cosine equality, `degradation_at_k` back-compat with `Embedder`, `MaxSimRetriever` shape + empty inputs (torch-gated), `load_head` round-trip (torch-gated).
  - `test_compare_c1_AC8.py` - three-way blocks present, verdict conjunction, target override, determinism (torch-gated; stub backbone).
  - `test_remote_eval_c1_AC9.py` - `eval-c1` registered; module imports without heavy deps.
- **GPU smoke on the box** (records evidence for AC3 + the ship-gate on real BGE-M3):
  - `./bin/train c1-corpus --max-per-source 2000 --out /tmp/pairs.parquet`
  - `./bin/train c1-fit --pairs /tmp/pairs.parquet --out-dir /tmp/c1 --max-steps 2000`
  - `bash infra/remote/c1_eval.sh` (or `bin/remote run eval-c1`) -> three-way `degradation@10` + ship-gate verdict, written to R2 under `runs/<run_id>/c1_eval.json`.
- **Live demo on the box** (SS 6, AC10; for team/external presentations while the lease is up):
  - `ssh aic2026-gpu 'bash c1_demo.sh canned'` - canned showcase only, ~20s, prints 12 side-by-side examples (5 of them `mixed_ocr`) + tally. Default `n_docs=2000` (de-saturates the index so the `mixed_ocr` wins are visible - at the demo-default 300 most modes tie).
  - `ssh -t aic2026-gpu 'bash c1_demo.sh both'` - canned followed by an interactive REPL (audience types Vietnamese queries; needs `-t` for the TTY).

## 9. Open questions

- **Q1 (blocks implementation)**: BGE-M3 head input width. proposal 08 part 3.2 step 3 writes the head as `768 -> 384 -> 384`, but BGE-M3 (XLM-RoBERTa-large) has hidden size **1024** and also ships a *native ColBERT multi-vector head*. Resolve at implementation by **reading the hidden size off the loaded model config** (never hardcode - the SPEC-0004 `1024 -> 1152` bug is the cautionary precedent), and decide whether to project BGE-M3's native ColBERT token vectors or its last-hidden-state token embeddings. Recommendation: project last-hidden-state token embeddings; benchmark against the native ColBERT vectors as an ablation. **Resolved (implementation):** `BgeM3Backbone` reads `model.config.hidden_size` (no hardcoded dim) and exposes last-hidden-state token embeddings; the head projects those, MaxSim is **mean-pooled over query tokens** (score in [-1, 1] so temperature 0.05 behaves like standard InfoNCE). Native-ColBERT-vector projection remains a future ablation.
- **Q2**: noise-schedule realism. The `Beta(2, 5)` drop rate is an assumption. proposal 04 part 6 (risk row) says validate it against a ~200-sample slice of real PhoWhisper / PaddleOCR output and retune. That real slice does not exist until we have ASR/OCR over the corpus (post-June-25). Until then, sanity-check the noise visually against a handful of known PhoWhisper error examples and record the assumption in `train_meta.json`. **v2 update:** OCR-character noise modes (`space_split`, `char_confuse`, `mixed_ocr`) were added in 2026-06-01 to cover failure modes the v1 schedule missed: PaddleOCR over-segmentation (`q u ả t á o`) and visual confusables (`5<->S`, `1<->l`, etc.). **v3 Tier A update:** following the Hoang & Aw (2012, EACL workshop, "Spell Checking in Vietnamese OCR-scanned Texts") 3-class taxonomy survey, three more modes were added on the same day to close the remaining high-impact gaps: `word_merge` (Class 3 under-segmentation, the inverse of `space_split` that v2 missed), `homophone_swap` (Class 2 real-syllable substitution -- the dominant PhoWhisper failure where the same base syllable confuses across its 5-6 tonal homophones; mark-anchored `tone_swap` cannot reach this), and `case_noise` (Class 3 case shift, preserved by spell-check). Same Beta(2,5) per-string severity prior; final calibration against real OCR output still post-June-25.
- **Q3**: dataset-id verification. The KTVIC / UIT-OpenViIC / VIVOS / Bud500 HF ids must be confirmed on the box (the `DEFAULT_SOURCES` list holds best-known ids; `<uit-openviic-hf-id>` is an explicit placeholder). The builder is fault-tolerant, so an unavailable source degrades gracefully. The exact datasets that land in corpus v1 are recorded in `train_meta.json` for provenance.
- **Q4**: hard-negative mining requires a BGE-M3 forward pass over the clean side (a one-time embed, minutes on the H200 for a few hundred thousand strings). CI / fixture mode falls back to random in-corpus negatives. Confirm in an ablation that mined negatives beat random by enough to justify the extra pass.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-30 | implementer (user-directed) | Created at status **Draft** for review (user chose spec-first). Scopes C1 as a now-trainable contribution on **public** Vietnamese caption + ASR text (KTVIC + UIT-OpenViIC + VIVOS + Bud500), with an identical post-June-25 re-run over our own index text. Implementation is gated on user approval of this spec. |
| 2026-05-30 | implementer | **Implemented** (status Draft -> Implementing). Shipped `train/diacritic_noise.py`, `train/diacritic_corpus.py`, `train/diacritic_bert.py`, `eval/diacritic_robustness.py`, the `train-c1` job, and the `bin/train` CLI; `train` extra + `pyarrow` (dev). AC1-AC5 green (AC3 validated with torch on CPU via an injected stub backbone; loss decreases, backbone frozen). Q1 resolved (read dim from model; last-hidden tokens; mean-pooled MaxSim). v1 decisions: training uses **in-batch negatives** (the mined `hard_negs` column is carried in the corpus but not yet consumed - Q4 follow-up); the degradation eval runs against single-vector encoders, **head-as-encoder MaxSim retrieval is a follow-up**. |
| 2026-05-31 | implementer | **Trained on real hardware (8x H200).** First C1 baseline: Vietnamese Wikipedia (`20231101.vi`) + KTVIC = 9,938 clean strings -> 39,752 pairs; **loss 1.530 -> 0.062** over 2000 steps (batch 128, lr 2e-4, wd 0.01), `in_dim=1024` read from BGE-M3 (confirms Q1 on real hardware). Corpus sources hardened after the first on-box smoke surfaced Q3: Wikipedia is the ungated parquet-native anchor + prose-column auto-detect for KTVIC; VIVOS / Bud500 / UIT-OpenViIC dropped from defaults (loader-script / gated / wrong-id under `datasets>=4.x`). Checkpoint + corpus + meta banked to R2 at `c1-baseline/7f18c88/` (durable across leases). Reproducible runners: `infra/remote/c1_smoke.sh` (corpus+fit) + `infra/remote/c1_bank.sh` (R2 upload via R2Client). Next: head-as-encoder `degradation@10` (the ship-gate metric). |
| 2026-05-31 | implementer | **Head-as-encoder ship-gate eval shipped.** Added `aic2026.eval.retrievers` (`Retriever` Protocol, `DenseRetriever`, `MaxSimRetriever`, `BgeM3DenseEmbedder`, `load_head`) and the `compare_c1_vs_baselines` + `build_heldout_queries` surface. `degradation_at_k` now accepts a `Retriever` *or* an `Embedder` (duck-typed; AC4 unchanged). `bin/train c1-eval --checkpoint head.pt` runs the three-way comparison; new `eval-c1` remote job + `infra/remote/c1_eval.sh` runner restore the baseline from R2, run the eval, and bank `c1_eval.json` to R2. New ACs **AC7-AC9** verified on CPU (`pytest` 4 new tests on the CPU path + 5 torch-gated tests green under `uv run --with torch`). Real `degradation@10` numbers + ship-gate verdict produced on a follow-up lease run. |
| 2026-06-01 | implementer | **Ship-gate PASSED on the 8x H200 lease (run `c1-baseline/7f18c88`).** First eval at `n=200` saturated (everything > 0.99); second at `n=5000` de-saturated and produced the real Phase-2 number: **c1_on overall 0.9848 vs baseline_maxsim 0.9741 vs baseline_dense 0.9654** -> passes absolute target (>=0.85) and beats both baselines monotonically. Win is concentrated on the hardest mode `drop_all`: **+4.6 pp over raw BGE-M3 MaxSim and +7.4 pp over dense mean-pool** -- exactly the failure mode the spec was built to attack (PhoWhisper / PaddleOCR diacritic stripping). Easier modes (`random_drop` / `tone_swap` / `mixed`) saturated at the 5000-string Wikipedia index scale because BGE-M3 is itself competent at lightly-noised Vietnamese; the proposal-05 SS 13.1 prediction of baseline 0.65-0.75 likely assumes a production-scale OCR/ASR/caption index (millions of strings). Useful ablation finding: BGE-M3 *token MaxSim alone* (no head) already beats dense mean-pool, so the late-interaction surface is itself contributing. Both `n200` and `n5000` JSONs banked to R2 under `c1-eval/7f18c88/{n200,n5000}/c1_eval.json` via `infra/remote/c1_eval_bank.sh`. **C1 ships as a Phase-2 contribution.** v2 follow-ups under this spec: mined hard-negatives (Q4), a longer training run, optional LoRA on BGE-M3's last layers. |
| 2026-06-01 | implementer | **v2 noise schedule (OCR-character).** User-flagged that the v1 schedule modelled only diacritic noise; real Vietnamese OCR also produces (a) letter-spacing artifacts (`quả táo` -> `q u ả t á o`) and (b) visual-confusable swaps (`5<->S`, `1<->l/I`, `0<->O`, `6<->G/b`, `8<->B`, `2<->Z/z`, `9<->g/q`, `7<->T`, `4<->A/h`, plus two-char `rn<->m` / `cl<->d`). Added `NoiseMode.SPACE_SPLIT`, `NoiseMode.CHAR_CONFUSE`, and `NoiseMode.MIXED_OCR` (the realistic-worst-case composition: `random_drop + char_confuse + space_split`). Bumped `variants()` and `build_corpus()` defaults so every mode is represented per anchor (`k = len(NoiseMode) = 7`); the corpus is now 75% larger per anchor than v1. `eval/diacritic_robustness.DEFAULT_MODES` extended to sweep all 7 modes. AC1 + AC2 tests expanded (5 new tests; 14 total under AC1; 4 total under AC2). |
| 2026-06-01 | implementer | **v2 trained + evaluated on the H200 lease.** Retrained the head on the expanded schedule (`n_pairs = 69,566` = 9,938 × 7 modes; loss 1.620 -> 0.084 over 2000 steps). Eval at `n=5000` produced **VERDICT=PASS** (c1_on overall **0.9510** vs baseline_maxsim 0.9409 vs baseline_dense 0.9305). Banked at `c1-baseline/7f18c88-v2-ocr/` + `c1-eval/7f18c88/n5000-v2-ocr/`. Key per-mode findings on the **new OCR modes** (the eval that didn't exist in v1): **`mixed_ocr`** (realistic combined OCR output) is the only mode that doesn't saturate -- retrieval falls to 0.72-0.79 range and **C1 wins by +4.0 pp over raw MaxSim and +6.3 pp over dense**, the largest absolute lift in the whole eval; **`char_confuse`** shows a small but real +0.3/+1.4 pp lift; **`space_split` alone shows a -0.6 pp regression vs raw MaxSim** -- the head can't add value when every char is already its own token (`q u ả t á o`), and we record this as a known limitation. Diacritic modes are within +/-0.6 pp of v1 (no regression). v2 ships: it preserves v1's diacritic wins and adds a genuine win on the realistic-worst-case OCR mode, with one honest negative-result mode (`space_split` standalone). |
| 2026-06-01 | implementer | **v3 Tier A noise schedule (Hoang & Aw 2012 Class 2/3).** Web-search audit of real Vietnamese OCR error taxonomies (Hoang & Aw 2012 EACL workshop) found three high-impact failure modes v2 still missed: **Class 3 under-segmentation** (we had over-seg but not the inverse), **Class 2 ASR real-syllable substitution** (tonal homophones -- the dominant PhoWhisper failure that survives spell-check because each candidate is a valid Vi word), and **Class 3 case shift** (also survives cleaning). Added `NoiseMode.WORD_MERGE` (symmetric inverse of `space_split`: `quả táo` -> `quảtáo`), `NoiseMode.HOMOPHONE_SWAP` (vowel-anchored tone re-assignment -- can ADD a tone to a level syllable like `ma` -> `má`/`mã`/etc. and strip/swap existing tones, the symmetric case that mark-anchored `tone_swap` cannot reach), and `NoiseMode.CASE_NOISE` (per-word random lower/upper/title). Per-vowel (not per-syllable-nucleus) tone re-assignment is documented as a known simplification -- mechanically what real PhoWhisper produces when it mis-positions tones. Corpus auto-expands from 9,938 x 7 = 69,566 to 9,938 x 10 = ~99,380 pairs (~75% bigger again). `DEFAULT_MODES` auto-extends to sweep all 10 modes. AC1 tests expanded (5 new tests; 19 total under AC1). The v2 head doesn't know the v3 modes -- a v3 lease run is needed to retrain on the expanded schedule and eval honestly. |
| 2026-06-01 | implementer | **v3 trained + evaluated on the H200 lease.** Retrained on the 10-mode schedule (n_pairs ~99,380; same 2000 steps batch 128). Eval at n=5000 produced **VERDICT=PASS** (c1_on overall **0.9768** vs maxsim 0.9581 vs dense 0.9504). Banked at `c1-baseline/7f18c88-v3-tier-a/` + `c1-eval/7f18c88/n5000-v3-tier-a/`. **Headline finding**: v3 training data (more diverse noise) strictly improves the v2 head on every existing mode -- not just adds coverage. Apples-to-apples on the 7 v2 modes: v3 c1 averages **0.968 vs v2 c1 0.951** (+1.7 pp). Per-mode lifts vs v2: drop_all **+2.1 pp** (0.943 -> 0.964), char_confuse **+1.6 pp** (0.966 -> 0.982), **mixed_ocr +6.4 pp** (0.786 -> 0.850, first time a hard combined mode crosses the 0.85 absolute target), space_split **+0.9 pp** (0.975 -> 0.984, **flipping v2's -0.6 pp regression vs MaxSim to a +0.3 pp win**). Diacritic modes (random_drop, tone_swap, mixed) within +0.3 pp -- no regression despite per-mode training budget shrinking from 1/7 to 1/10. **Honest finding I predicted wrong**: the 3 NEW modes (word_merge, homophone_swap, case_noise) all saturated at n=5000 because each Wikipedia held-out sentence has 5-8 syllables and single-syllable noise gets disambiguated by the other syllables. To see homophone_swap actually bite would need short proper-noun queries (the real PhoWhisper failure case, e.g. "Nguyễn Văn Mã") or much higher noise rates -- this is an eval design limitation, not a head problem (the head WAS trained on these modes; the eval can't see the difference at this query length / index scale). v3 ships as the best head so far. |
| 2026-06-01 | implementer | **Live demo surface (SS 6).** Added `aic2026.eval.demo` (`CannedExample`, `CANNED_EXAMPLES`, `format_example_block`, `run_canned`, `run_interactive`), `bin/train c1-demo {canned\|interactive\|both}`, `infra/remote/c1_demo.sh` runner, and AC10 (12 CPU-only tests via `FakeRetriever`). Canned set is 5 examples curated to exhibit one strength per failure family C1 was built to attack: `drop_all` placename, `char_confuse` address, `mixed_ocr` long sentence (the v3 win), `word_merge` short phrase, and a clean-sanity example. The demo wires the *same* three retrievers as the ship-gate (`MaxSimRetriever(head)` vs `MaxSimRetriever(None)` vs `DenseRetriever(BgeM3DenseEmbedder)`), so what the demo shows and what the eval measures cannot diverge. Vietnamese labels throughout (`[TRÚNG]`, `C1 THẮNG/HÒA/THUA`, `Tổng kết`) for external presentation. Runs in ~20s for the canned showcase + as long as needed for the REPL. Restores `head.pt` + `pairs.parquet` from R2 `c1-baseline/<sha>/` exactly like `c1_eval.sh`; tested with the v3 head. |
