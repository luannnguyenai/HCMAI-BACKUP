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

## 1.1 Actual structure + fetch status (confirmed on box 2026-06-02)

`gdown --folder` recursed into the subfolders and revealed this is the **full AIC2025 finals dataset**, much richer than the top-level listing. What the recursion exposed (and what landed before a Drive quota stopped the fetch):

| Path | Contents | Fetch status |
|---|---|---|
| `query/` | `DanhSachTruyVanAIC_Chungket.xlsx` ("AIC Finals Query List") + `query-p{1,2,3}-groupA.zip` | ✅ **landed** (~80 KB) |
| `video_batch_1/clip-features-32-aic25-b1.zip` | **organizers' CLIP ViT-B/32 features** (the "weak CLIP" of note 06; resolves **Q-DS-1**) | ✅ landed (168 MB) |
| `video_batch_1/media-info-aic25-b1.zip` | **metadata** (unlocks the yt-dlp transcript lane, note 06 §2.1) | ✅ landed (1.1 MB) |
| `video_batch_1/objects-aic25-b1.zip` | **object detection** (free OD filter lane, note 06 §2.5) | ✅ landed (640 MB) |
| `video_batch_1/Videos_L21–L30_*.zip` | raw videos (L-series) | ⚠️ partial — L21–L24 + part of L25, then quota |
| `video_batch_2/Videos_K01–K20.zip` | raw videos (K-series) | ❌ not reached (quota) |
| `Keyframes_L25–L30.zip` | **pre-extracted keyframes** (what we most need) | ❌ **not reached (quota)** — they are LAST in gdown's order |

**Drive quota.** The fetch died with *"Too many users have viewed or downloaded this file recently"* after ~22 GB — gdown's well-known failure on large/popular shared folders. The keyframes (last in order) never came. See §6 for the robust re-fetch path.

## 2. Scale estimate (listing-level; confirm with the profiler)

19.3 GB of JPEG keyframes at a typical competition resolution (~100–200 KB/frame) ≈ **~100k–200k keyframes for L25–L30 alone**. If the full corpus spans L01–L30 at similar density, that is **order-of-magnitude ~1M keyframes** — consistent with the cold-indexing sizing in [research-note 03 §J](03-foundation-models-2026.md) (~50 GPU-hours for 1M frames on an A6000; far less on the H200). **One H200 lease can index the whole thing.**

> ⚠️ This is an estimate from archive sizes, not a frame count. The profiler reports the real per-collection counts, naming scheme, and resolution histogram.

## 3. Why this matters — it's our pre-June-25 real-data proxy

We have repeatedly been blocked on "can't validate on real data until June 25." This corpus breaks that for four work-items:

1. **C1 noise calibration ([SPEC-0014](../specs/SPEC-0014-diacritic-bert.md) Q2 — open).** The spec says the real PhoWhisper/PaddleOCR error distribution "does not exist until we have ASR/OCR over the corpus." It exists now: the real `query/` text gives the clean-query distribution, and OCR/ASR over these keyframes/videos gives the real noisy-text distribution. The calibration harness ([`src/aic2026/train/calibrate.py`](../../src/aic2026/train/calibrate.py), `bin/train c1-calibrate`) compares both against our synthetic `noise()` output and recommends knob tweaks. **This is the most direct unblock.**
2. **Full offline-pipeline dry-run.** keyframe → SigLIP-2 / Meta CLIP 2 embed → Milvus index → query, end-to-end on real data before the real data exists. De-risks SPEC-0004 / 0006 / 0015 integration.
3. **The Qwen3-VL-Embedding bench.** Run the encoder bake-off (Qwen3-VL-Embedding-2B vs the SigLIP-2 / Meta CLIP 2 / InternVideo2 floor) on **real Vietnamese keyframes + real queries**, turning the model-selection debate (per the 2026-06-02 meeting) into a measurement. See the encoder-selection report / forthcoming research note.
4. **Keyframe-density audit ([note 06](06-aic2026-dataset-shape.md) Q-DS-3).** Are the provided keyframes dense enough, or do we need our own TransNetV2 pass? Answerable now from the per-collection frame counts.

## 4. Provided assets — all present (resolves note 06 expectations)

[Note 06](06-aic2026-dataset-shape.md) expected organizers to also ship metadata, object detection, and the weak CLIP embeddings. **Confirmed present** in `video_batch_1/` (see §1.1): `clip-features-32` (CLIP **ViT-B/32**, resolving **Q-DS-1** — it is the 512-d 2021-era model, consistent with note 05's finding that the 2025 baseline defaulted to ViT-B/32; weak, as expected), `media-info` (metadata → the **yt-dlp transcript lane**, note 06 §2.1, is viable), and `objects` (OD → free filter lane, note 06 §2.5; schema TBD on unzip, **Q-DS-2**). All three landed before the quota cut-off.

## 4.1 Real query shape (profiled on box 2026-06-02) — important for C1 + the demo

The finals `query/DanhSachTruyVanAIC_Chungket.xlsx` has columns **`Query Name | Description (vi) | Trans (en)`** and query IDs like `tkis-query-01` (**t**extual **KIS**). The profiler parsed 81 cells (incl. headers + the English `Trans` column + IDs — the loader is naive; a column-aware extract is a refinement). Headline:

- **Queries are long paragraph-length descriptions, not short phrases**: char-len mean **291**, median **322**, max **899**; word-len mean **56**, median **62**, max **167**.
- Example (one `tkis` query, abridged): *"Trong đoạn video, có một nhóm khoảng 4 người… Đây là video giới thiệu về kỷ lục Guiness về làm sợi bún khoai tây dài nhất thế giới"* — multi-sentence, detail-stacked.

**Implications:**
1. **The query is clean human Vietnamese; the noise is on the *index* side** (OCR text from keyframes, ASR transcripts). So C1's real job is *clean-query → noisy-indexed-text* matching. The Q2 calibration that matters most is **synthetic noise vs real OCR/ASR output** (still blocked on keyframes), more than query-distribution matching.
2. **Our C1 anchors (Wikipedia sentences) and demo examples (short phrases) under-represent length.** Real targets are paragraph-scale. This corroborates the v3 eval finding that single-syllable noise modes saturate — at paragraph length they matter even less. Consider longer, multi-sentence anchors for a more representative corpus + demo.

## 5. Caveats

- **Partially profiled.** §1.1, §4, §4.1, §6 are confirmed on the box (2026-06-02): query set + provided assets are real. **Keyframe** counts / naming / resolution remain _PENDING_ — the Drive quota blocked the keyframe zips (see §6 for the re-fetch path). §2's frame-count is still an estimate.
- **Permission.** [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md) / [`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md) cover borrowing the 2025 *baseline code*. Using the 2025 *dataset* for our dev should get an explicit nod from the organizers or the team member who shared it. **Action: confirm before publishing any results derived from it.**

## 6. Ground truth (profiled 2026-06-02; keyframes still pending the re-fetch)

| Field | Value |
|---|---|
| Query set format | `.xlsx`, cols `Query Name \| Description (vi) \| Trans (en)`, IDs `tkis-query-NN` |
| Query rows parsed | 81 cells (incl. headers + English `Trans` + IDs — loader is column-naive) |
| Query length (chars) | mean **291**, median **322**, max **899** |
| Query length (words) | mean **56**, median **62**, max **167** |
| Provided CLIP | **ViT-B/32** (`clip-features-32`), 168 MB — resolves Q-DS-1 |
| Metadata / OD | `media-info` (1.1 MB) + `objects` (640 MB) present |
| Total keyframes | **PENDING (Drive quota blocked the keyframe zips)** |
| Per-collection counts / naming / resolution | PENDING (need keyframes) |
| Video count / containers | PENDING (only partial L21–L25 landed; not unzipped) |

### Re-fetching the keyframes (gdown quota-failed — use a robust path)

`gdown --folder` on the whole folder dies with *"Too many users have downloaded recently"* on the large zips. Don't just re-run it. Options, best first:

1. **rclone with the team Google account** (most reliable for quota'd folders): configure a Drive remote once, then `rclone copy "remote:<folder>" /tmp/aic2025 --drive-shared-with-me`. Authenticated transfers don't hit the anonymous-download quota the same way.
2. **Manual browser download** of just the 8 `Keyframes_L*.zip` (the only thing we still need) → scp to the box. Tedious but unblocked.
3. **Per-file `gdown` by ID after the quota resets** (hours), fetching only keyframes (IDs captured from the fetch log), e.g. `uv run --with gdown gdown <id> -O /tmp/aic2025/kf/Keyframes_L25.zip`. Still risks re-hitting quota on the big zips.

Once keyframes land: `ssh aic2026-gpu 'bash profile_aic2025.sh "" /tmp/aic2025'` re-profiles in place (skips download, unzips `Keyframes_*`, fills the PENDING rows).

## 7. References

- Drive folder: <https://drive.google.com/drive/folders/1eO4XpkeF0gq1J5P5-_N4TUMqQ_c9vn4R>
- Profiler: [`infra/remote/profile_aic2025.py`](../../infra/remote/profile_aic2025.py) + [`infra/remote/profile_aic2025.sh`](../../infra/remote/profile_aic2025.sh)
- C1 calibration harness: [`src/aic2026/train/calibrate.py`](../../src/aic2026/train/calibrate.py) (`bin/train c1-calibrate`); [SPEC-0014](../specs/SPEC-0014-diacritic-bert.md) Q2.
- AIC2026 dataset-shape intel: [research-note 06](06-aic2026-dataset-shape.md).
- 2025 baseline code analysis: [research-note 05](05-baseline-2025-analysis.md); borrowing policy [ADR-0010](../adr/ADR-0010-borrow-from-2025-baseline.md).
