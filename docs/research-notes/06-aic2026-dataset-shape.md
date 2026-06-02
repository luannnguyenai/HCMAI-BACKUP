# Research Note 06 ù AIC2026 dataset shape (partial team-channel intel)

> Captures what we know about the **official AIC2026 dataset** format ahead of the June 25, 2026 release. Source: team-channel discussion on 2026-05-28 with a teammate ("Hoùng") who has prior-edition context. This is **provisional**; supersede with organiser-issued documentation when it lands.

## 1. Confirmed dataset contents

Per the team-channel discussion, organisers will provide:

| Asset | Provided as | Notes |
|---|---|---|
| **Video files** | yes | one per task corpus item |
| **Metadata** | yes | includes `url` (YouTube), `description`, and presumably title / publish-date / channel |
| **Object detection (OD)** | yes, pre-computed | model unspecified; treat as advisory until verified |
| **Keyframes** | yes, pre-extracted | extraction strategy unspecified; likely uniform sample or a low-quality shot detector |
| **CLIP embeddings** | yes, pre-computed | **organisers' embedding model is reportedly weak** (teammate: "hùng c?a BTC xùi model cùi nùn khùng trùng ch? gù") ù treat as baseline lane only |

What is **not** provided in this account (and that we still own):

- Vietnamese ASR transcripts (we ingest)
- OCR text (we ingest)
- Image embeddings from our chosen 3-encoder ensemble (we compute)
- Captions in Vietnamese (we generate with Qwen2.5-VL)
- ADL / place / time tags beyond what's in `metadata`

## 2. Implications for the proposals and reserved specs

### 2.1 yt-dlp transcript ingestion is a new first-class lane

Because the metadata carries the **YouTube URL**, the cheapest and highest-quality source of Vietnamese transcript text is **YouTube's own auto-captions**, pulled via `yt-dlp`. The teammate recommends:

> "Metadata s? cù url youtube -> yt-dlp trùch xu?t transcript (vi) khùng cù thù fallback dùng model Whisper. (Cùn mùy x?n process cùi vùo thù thùi kh?i ch?n :))"

Interpreted as a pipeline:

```
metadata.url --+--> yt-dlp --auto-subs--> transcript_vi (preferred when present)
               |
               +--> ffmpeg --extract audio--> PhoWhisper-large (fallback when no captions)
```

This change has **two downstream consequences**:

1. **PhoWhisper licence risk drops** (master strategy ù10 item 2). If yt-dlp covers most of the corpus, our hard dependency on a CC-BY-NC-SA model shrinks to "fallback when captions absent". In the limit, we may not need PhoWhisper at all - depends on the empirical fraction.
2. **ASR ingestion structure changes** (affects SPEC-0005). It becomes a two-source merge with provenance tracking (`source: "yt-dlp" | "phowhisper"`), not a single Whisper pass.

### 2.2 Shot detection is no longer Phase 1 critical

Organisers ship pre-extracted keyframes. SPEC-0003 (data ingestion) no longer needs **TransNetV2 / AutoShot / TransVLM** as a Phase 1 dependency. Our pipeline can start from the provided keyframes.

The teammate suggested two upgrade paths for shot detection if we ever need our own:

| Option | When to consider | Risk |
|---|---|---|
| **TransNetV2** | If we need our own keyframes for any reason | low; battle-tested at LSC/VBS |
| **AutoShot** (CVPR 2023) | If the organiser keyframes are visibly under-sampled and we have GPU budget | medium; less prior art at LSC/VBS |
| **TransVLM** | If we want SOTA + have engineering budget | high; reproducing from a paper |

**Decision for now**: defer shot detection entirely to Phase 2 as a contingency. If during Phase 1 we observe organiser keyframes are too sparse for KIS or Ad-hoc tasks, escalate to AutoShot; do not preempt by re-running TransNetV2.

### 2.3 We still compute our own image embeddings (SPEC-0004 unchanged)

The teammate confirms what we already assumed: the provided CLIP is "model cùi" ù weak. Our [ADR-0007 Edge contribution C2](../adr/ADR-0007-original-contributions-c1-c2-c4.md) and the three-encoder ensemble (SigLIP-2 + Meta CLIP 2 + InternVideo2-1B) are precisely what beats it.

**However**, we should still **ingest the provided CLIP embedding as a fourth lane** in Milvus, for two reasons:

1. It's a fair, reproducible baseline our edge has to beat in ablations.
2. If our compute pipeline fails for any video, the provided embedding is a graceful fallback.

This adds one ranked list to the per-task-type learned fusion in C2 ù measurable cost ~hours of indexing storage; measurable gain a small lift on the slice of queries where the organisers' model happens to find a near-match.

### 2.4 Milvus + Elasticsearch schema gain organiser-metadata fields

- **Milvus** (SPEC-0006): structured columns for `youtube_url`, optionally a fourth dense vector field `clip_organiser` for the provided embedding.
- **Elasticsearch** (SPEC-0007): a `description` text field, Vietnamese-aware analyser (NOT asciifolding ù same trap as [research-note 05 ù4.1](05-baseline-2025-analysis.md)).

### 2.5 Pre-computed OD is a free filter lane

We can use the organisers' object-detection labels as structured filter input (place, person count, vehicle, etc.) **for free**, alongside our planned YOLOv8 / Places365 / LSC-ADL labels. Mismatch risk: their classes likely follow COCO or similar; document the schema once it lands and decide whether to keep, replace, or merge with our own labels.

## 3. Updates to open questions

| Master strategy ù10 | Before | After |
|---|---|---|
| Q2 (CC-BY-NC licence verification) | "must email organisers" | "still verify, but risk reduced - yt-dlp may cover most of the corpus and demote PhoWhisper to optional fallback" |
| Q7 (dataset preview) | "open" | "partial team-channel intel captured in research-note 06; **last-year AIC2025 corpus now in hand as a proxy ([research-note 07](07-aic2025-proxy-corpus.md))**; full AIC2026 schema awaits June 25 release" |

## 4. Open questions surfaced by this intel

These add to (not replace) the existing open questions:

- **Q-DS-1**: What model did the organisers use for the pre-computed CLIP embeddings? (need to know the dimensionality, the metric, and any normalisation conventions) **(2026-06-02: RESOLVED for the AIC2025 proxy ó the provided file is `clip-features-32` = CLIP ViT-B/32, 512-d, 2021-era; weak as expected. See [research-note 07](07-aic2025-proxy-corpus.md) ß4. AIC2026's may differ; re-confirm on the June-25 drop.)**
- **Q-DS-2**: What object-detection schema is provided? COCO 80 classes? Open-vocabulary? Custom?
- **Q-DS-3**: How dense is the keyframe sampling - one per second? One per shot? Uniform N per video? **(2026-06-02: now answerable on last-year data ù the AIC2025 proxy corpus ([research-note 07](07-aic2025-proxy-corpus.md)) ships pre-extracted keyframes; run [`infra/remote/profile_aic2025.py`](../../infra/remote/profile_aic2025.py) for per-collection counts + naming scheme.)**
- **Q-DS-4**: Are the YouTube videos all still available in 2026, or have any been taken down? Audit early.
- **Q-DS-5**: For yt-dlp transcript extraction: what fraction of videos have Vietnamese auto-captions? Need an empirical pass once we have the URLs.
- **Q-DS-6**: Is the `description` field reliably populated or sparse?

All Q-DS items go on the agenda for the post-2025-baseline-author interview ([`docs/permissions/2025-baseline-reuse.md`](../permissions/2025-baseline-reuse.md) ù4) **and** on the post-June-25 dataset-shake-down checklist.

## 5. References

- Source: team-channel discussion 2026-05-28; teammate "Hoùng" surfacing recommendations and dataset shape.
- TransNetV2: <https://github.com/soCzech/TransNetV2>
- AutoShot (CVPR 2023): "AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection", <https://arxiv.org/abs/2304.06116>
- TransVLM: search-pending; reference in teammate's message implies a vision-language-model-based shot detector ù needs verification.
- yt-dlp: <https://github.com/yt-dlp/yt-dlp>
- PhoWhisper licence (CC-BY-NC-SA-4.0): <https://huggingface.co/vinai/PhoWhisper-large> - see also [strategy ù10 item 2](../strategy/00-master-strategy.md).
- Earlier analysis of the 2025 baseline: [`docs/research-notes/05-baseline-2025-analysis.md`](05-baseline-2025-analysis.md).
