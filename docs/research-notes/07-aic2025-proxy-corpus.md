# Research Note 07 — AIC2025 proxy corpus (last-year dataset as a pre-June-25 stand-in)

> A teammate shared the **AIC2025** competition dataset (last year's edition) via Google Drive on 2026-06-02. This is the closest thing we have to the June-25 AIC2026 corpus, and it lets us validate large parts of the pipeline **now** instead of waiting. This note captures the structure from the Drive listing + the strategic value; the empirical numbers are filled in by [`infra/remote/profile_aic2025.py`](../../infra/remote/profile_aic2025.py) once the data is on the box.
>
> **Status: provisional (listing-level).** Ground-truth fields marked _PENDING-PROFILE_ until the profiler runs. Supersedes nothing; complements [research-note 06](06-aic2026-dataset-shape.md) (the AIC2026 dataset-shape intel).

Source: Google Drive folder shared 2026-06-02 — <https://drive.google.com/drive/folders/1eO4XpkeF0gq1J5P5-_N4TUMqQ_c9vn4R>

## 1. What's in the drop (from the Drive listing)

| Item | Type | Size | Reading |
|---|---|---|---|
| `query/` | folder | — | **Official query set** (KIS / Q&A / TRAKE prompts). Highest-value item — the real query format + language. |
| `video_batch_1/`, `video_batch_2/` | folders | — | Raw source videos (originals behind the keyframes). |
| `Keyframes_L25.zip` … `Keyframes_L30.zip` (incl. `L26_c/d/e`) | 8 zips | **≈ 19.3 GB** | Pre-extracted keyframes, one archive per video collection ("L-group"). |

**Collection naming.** AIC groups videos into collections `L01, L02, …`. This folder shows only **L25–L30** (L26 split into `c/d/e`) — almost certainly the **tail subset** of the full corpus, not all of it. The full keyframe set (L01–L30+) is likely larger or in sibling folders. Mixed modified-dates (Oct 2024 / Aug 2025 / May 2026) suggest the folder aggregates assets across editions.

## 2. Scale estimate (listing-level; confirm with the profiler)

19.3 GB of JPEG keyframes at a typical competition resolution (~100–200 KB/frame) ≈ **~100k–200k keyframes for L25–L30 alone**. If the full corpus spans L01–L30 at similar density, that is **order-of-magnitude ~1M keyframes** — consistent with the cold-indexing sizing in [research-note 03 §J](03-foundation-models-2026.md) (~50 GPU-hours for 1M frames on an A6000; far less on the H200). **One H200 lease can index the whole thing.**

> ⚠️ This is an estimate from archive sizes, not a frame count. The profiler reports the real per-collection counts, naming scheme, and resolution histogram.

## 3. Why this matters — it's our pre-June-25 real-data proxy

We have repeatedly been blocked on "can't validate on real data until June 25." This corpus breaks that for four work-items:

1. **C1 noise calibration ([SPEC-0014](../specs/SPEC-0014-diacritic-bert.md) Q2 — open).** The spec says the real PhoWhisper/PaddleOCR error distribution "does not exist until we have ASR/OCR over the corpus." It exists now: the real `query/` text gives the clean-query distribution, and OCR/ASR over these keyframes/videos gives the real noisy-text distribution. The calibration harness ([`src/aic2026/train/calibrate.py`](../../src/aic2026/train/calibrate.py), `bin/train c1-calibrate`) compares both against our synthetic `noise()` output and recommends knob tweaks. **This is the most direct unblock.**
2. **Full offline-pipeline dry-run.** keyframe → SigLIP-2 / Meta CLIP 2 embed → Milvus index → query, end-to-end on real data before the real data exists. De-risks SPEC-0004 / 0006 / 0015 integration.
3. **The Qwen3-VL-Embedding bench.** Run the encoder bake-off (Qwen3-VL-Embedding-2B vs the SigLIP-2 / Meta CLIP 2 / InternVideo2 floor) on **real Vietnamese keyframes + real queries**, turning the model-selection debate (per the 2026-06-02 meeting) into a measurement. See the encoder-selection report / forthcoming research note.
4. **Keyframe-density audit ([note 06](06-aic2026-dataset-shape.md) Q-DS-3).** Are the provided keyframes dense enough, or do we need our own TransNetV2 pass? Answerable now from the per-collection frame counts.

## 4. What's NOT in this folder (vs note 06's expected assets)

[Note 06](06-aic2026-dataset-shape.md) expects organizers to also ship **metadata (YouTube URL, description)**, **object detection**, and the **weak CLIP embeddings**. None are visible in this listing — only keyframes + videos + queries. Either they live in sibling folders, or this 2025 drop omitted them. This matters because the **YouTube-URL → yt-dlp transcript** lane (note 06 §2.1) depends on that metadata existing. Confirm during the profiler run / by browsing the Drive.

## 5. Caveats

- **Listing, not bytes.** §1–§2 are inferred from the Drive listing + AIC domain knowledge. Frame counts, naming scheme, resolution, and query format are _PENDING-PROFILE_.
- **Permission.** [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md) / [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md) cover borrowing the 2025 *baseline code*. Using the 2025 *dataset* for our dev should get an explicit nod from the organizers or the team member who shared it. **Action: confirm before publishing any results derived from it.**

## 6. Ground truth — _PENDING-PROFILE_

Run on the box and paste the JSON back here:

```bash
scp infra/remote/profile_aic2025.{sh,py} aic2026-gpu:.
ssh aic2026-gpu 'bash profile_aic2025.sh "https://drive.google.com/drive/folders/1eO4XpkeF0gq1J5P5-_N4TUMqQ_c9vn4R"'
# or, if already downloaded: ssh aic2026-gpu 'uv run python profile_aic2025.py --root /tmp/aic2025 --out /tmp/aic2025/profile.json'
```

| Field | Source | Value |
|---|---|---|
| Total keyframes (L25–L30) | profiler `keyframes.total_frames` | _PENDING_ |
| Per-collection counts | `keyframes.per_collection` | _PENDING_ |
| Frame naming scheme | `keyframes.sample_paths` | _PENDING_ |
| Resolution distribution | `keyframes.resolution_hist` | _PENDING_ |
| Query count | `queries.n_strings` | _PENDING_ |
| Query format / ext | `queries.ext_hist` | _PENDING_ |
| Query length (chars/words) | `queries.char_len` / `word_len` | _PENDING_ |
| Query diacritic density | `queries.vietnamese_diacritic_ratio_mean` | _PENDING_ |
| Video count / containers | `videos.total_videos` / `ext_hist` | _PENDING_ |

## 7. References

- Drive folder: <https://drive.google.com/drive/folders/1eO4XpkeF0gq1J5P5-_N4TUMqQ_c9vn4R>
- Profiler: [`infra/remote/profile_aic2025.py`](../../infra/remote/profile_aic2025.py) + [`infra/remote/profile_aic2025.sh`](../../infra/remote/profile_aic2025.sh)
- C1 calibration harness: [`src/aic2026/train/calibrate.py`](../../src/aic2026/train/calibrate.py) (`bin/train c1-calibrate`); [SPEC-0014](../specs/SPEC-0014-diacritic-bert.md) Q2.
- AIC2026 dataset-shape intel: [research-note 06](06-aic2026-dataset-shape.md).
- 2025 baseline code analysis: [research-note 05](05-baseline-2025-analysis.md); borrowing policy [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md).
