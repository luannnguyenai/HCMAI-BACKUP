# Proposal 03 - Data Pipeline

> How we ingest the AIC2026 dataset, extract features, and build searchable indexes. This is the offline backbone everything else depends on.

## 1. Inputs (expected)

Based on prior AIC HCMC editions (2023-2025), the AIC2026 dataset will include:
- Source videos (Vietnamese broadcast / news / ambient) - estimated 500-2000 hours
- Pre-extracted keyframes (provided by organisers OR extracted by us via TransNetV2)
- Optional starter kit: object labels, ASR transcripts, OCR text
- Sample task set + scoring rubric

Until the dataset releases (June 25), we develop against:
- **LSC'24 lifelog dataset** (725K Narrative Clip images + metadata) - publicly available
- **VBS V3C1+V3C2** (28K videos, ~2400 hours) - publicly available
- A synthetic Vietnamese news corpus we generate by downloading **VTV / HTV / Tuoi Tre / Thanh Nien** YouTube clips with `yt-dlp` (legal grey area; for internal practice only)

## 2. Storage layout

```
data/
  raw/
    videos/<video_id>.mp4
    audio/<video_id>.wav   (extracted with ffmpeg, 16kHz mono)
  intermediate/
    shots/<video_id>.json       (TransNetV2 boundaries)
    keyframes/<video_id>/<idx>.jpg
    asr/<video_id>.json         (Whisper segments + ts)
    ocr/<video_id>.json         (PaddleOCR boxes)
    captions/<video_id>.json    (Qwen2.5-VL Vietnamese)
  embeddings/
    siglip2.parquet             (frame_id, vector_1024)
    metaclip2.parquet
    internvideo2.parquet
    clap.parquet
  indexes/
    milvus/                     (data files)
    elasticsearch/              (data files)
  metadata/
    frames.parquet              (frame_id, video_id, ts, place, adl, objects)
```

Conventions:
- `video_id` = stable 8-char hash of original filename (sha256 truncated).
- `frame_id` = `<video_id>_<frame_idx>` (zero-padded 6 digits).
- All Parquet files use Snappy compression + Arrow schema.

## 3. Stage-by-stage spec

### 3.1 Video unpacking
- `ffmpeg -i input.mp4 -ar 16000 -ac 1 audio.wav` to extract audio.
- Original video stays as-is on disk; we *do not* re-encode.

### 3.2 Shot detection
- **TransNetV2** off-the-shelf model.
- Output: list of `(start_frame, end_frame, confidence)` tuples per video.
- For videos already supplied as folders of images (LSC-style), use content-based segmentation (HSV histogram drift + CLIP feature deltas) ala SnapSeek 3.0.

### 3.3 Keyframe extraction
- For each shot: extract the **centre I-frame** of the shot using ffmpeg seek.
- Plus: uniform 1 fps padding for shots longer than 5s.
- For TRAKE/4-scene support: also extract one 3x3 collage per shot for VLM rerank input.

### 3.4 Image embedding (3 models, parallel)

```
for batch in DataLoader(keyframes, batch_size=256):
    siglip2_vec = siglip2(batch).cpu()
    metaclip2_vec = metaclip2(batch).cpu()
    # internvideo2 runs on clips not frames; see 3.5
```

- Use `bfloat16` inference; cast to `float16` on disk.
- Write to Parquet in shards of ~50K rows each.

### 3.5 Video embedding (InternVideo2)
- For each shot: sample 4 frames evenly.
- Encode with InternVideo2-1B at 224x224.
- Store one 768-d vector per shot (not per frame).
- During retrieval, expand to all frames in the shot for grid display.

### 3.6 OCR
- **PaddleOCR PP-OCRv5** detection -> bboxes.
- For each box, recognize with `latin` (Vietnamese diacritic-aware) recognizer.
- If confidence < 0.6, retry with **VietOCR** Seq2Seq.
- Output: list of `{bbox, text, conf}` per frame.

### 3.7 ASR
- **PhoWhisper-large** via Faster-Whisper (int8 CTranslate2).
- VAD with Pyannote before Whisper to skip silence.
- Force-align with WhisperX for word-level timestamps.
- **Post-process**: Gemini 2.5 Flash diacritic-correction pass (optional but high-value).

### 3.8 Captioning (selective, 1/8 frames)
- Prompt: "Mo^ ta? chi tie^?t bu+?c a?nh na`y ba`?ng tie^?ng Vie^.t. Bao go^`m: dia? die^?m, nha^n va^.t, ha`nh do^.ng, ddo^` va^.t, ma`u sa?c chu? da.o."
- Model: **Qwen2.5-VL-7B BF16** local.
- Run on 1/8 sampled keyframes to keep cost manageable; expand to all if cost permits.

### 3.9 Object/scene/ADL labels
- **YOLOv8x** for COCO objects (80 classes).
- **Places365** ResNet-50 for scene labels (one top-1 + top-5).
- **LSC-ADL 35-class clustering** for ADL tags (run Qwen-VL-7B on a representative subset to bootstrap labels, then propagate via Places365 + CLIP similarity).

### 3.10 Metadata table
- One Parquet with columns: `frame_id, video_id, ts_ms, shot_id, place_label, place_top5, adl_label, objects[], duration_ms, has_ocr, has_asr, has_caption`.
- This is the structured-filter table joined into Milvus.

## 4. Indexing into Milvus

```python
from pymilvus import MilvusClient, FieldSchema, DataType

client = MilvusClient("milvus.db")

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("frame_id", DataType.VARCHAR, max_length=64, is_primary=True)
schema.add_field("video_id", DataType.VARCHAR, max_length=64)
schema.add_field("ts_ms", DataType.INT64)
schema.add_field("siglip2", DataType.FLOAT_VECTOR, dim=1024)
schema.add_field("place_label", DataType.VARCHAR, max_length=64)
schema.add_field("adl_label", DataType.VARCHAR, max_length=64)

# HNSW index
index_params = client.prepare_index_params()
index_params.add_index("siglip2", index_type="HNSW", metric_type="IP",
                       params={"M": 32, "efConstruction": 200})
```

One collection per embedding model. Structured fields (`place_label`, `adl_label`, etc.) are duplicated in each collection so we can filter without joins.

## 5. Indexing into Elasticsearch

Indexes:
- `idx_ocr` - text + bbox + ts
- `idx_asr` - text + word-level ts
- `idx_caption` - long Vietnamese captions

Analyser config:
```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "vi_search": {
          "tokenizer": "icu_tokenizer",
          "filter": ["lowercase", "icu_normalizer", "asciifolding"]
        }
      }
    }
  }
}
```

For BGE-M3 dense vectors, store as `dense_vector` field with `index: true, similarity: cosine`. Then we can do hybrid keyword + dense search in a single Elasticsearch query.

## 6. End-to-end pipeline orchestration

We will use **Prefect 2.x** (open-source) for orchestration. Why: lightweight, Python-native, good for both batch and streaming. Alternative: Airflow (too heavy), Dagster (too opinionated for our timeline).

Pipeline definition `flows/index_pipeline.py`:
```python
@flow
def index_video(video_path: Path):
    audio = extract_audio(video_path)
    shots = detect_shots(video_path)
    keyframes = extract_keyframes(video_path, shots)

    asr_task = run_asr.submit(audio)
    ocr_task = run_ocr.submit(keyframes)
    embed_siglip = run_siglip2.submit(keyframes)
    embed_metaclip = run_metaclip2.submit(keyframes)
    embed_video = run_internvideo2.submit(video_path, shots)
    captions = run_captioning.submit(keyframes, sample_ratio=0.125)
    objects = run_yolo.submit(keyframes)
    places = run_places365.submit(keyframes)

    metadata = build_metadata(shots, places, objects, captions)

    upsert_milvus(metadata, embed_siglip.result(), embed_metaclip.result(), embed_video.result())
    upsert_elastic(metadata, asr_task.result(), ocr_task.result(), captions.result())
```

## 7. Performance targets

For a 1M-keyframe / 800-hour-video dataset on a single A6000 (48 GB):

| Stage | Wall time | Bottleneck |
|---|---|---|
| TransNet shot detect | 6 h | GPU |
| Keyframe extract | 4 h | Disk I/O |
| SigLIP-2 | 12 h | GPU |
| Meta CLIP 2 | 16 h | GPU |
| InternVideo2 | 24 h | GPU |
| PaddleOCR | 6 h | CPU OK; can parallelise |
| PhoWhisper + WhisperX | 32 h | GPU |
| Qwen-VL captioning (1/8) | 20 h | GPU |
| YOLO + Places + ADL | 4 h | GPU |
| Milvus + Elastic ingest | 4 h | Disk |
| **Total wall time** | **~5-7 days** | -- |

Acceptable. If we need to re-index after a model swap mid-competition, we can parallelise across 4 GPUs cloud-burst and finish in 1 day.

## 8. Storage footprint

| Asset | Size for 1M frames |
|---|---|
| Keyframes (jpg @ 80% quality, 384px) | 30 GB |
| SigLIP-2 embeddings (fp16, 1152-d) | 2.3 GB |
| Meta CLIP 2 embeddings | 2 GB |
| InternVideo2 embeddings (per-shot, 768-d) | 0.5 GB |
| CLAP embeddings | 1 GB |
| OCR JSON | 0.5 GB |
| ASR JSON | 0.3 GB |
| Captions JSON | 0.4 GB |
| Metadata Parquet | 0.2 GB |
| Milvus index files | 8 GB |
| Elasticsearch indexes | 5 GB |
| **Total** | **~50 GB** |

Easily fits one NVMe SSD per machine.

## 9. Incremental indexing

Once initial index built:
- New videos arrive as a folder.
- Run `index_video` Prefect flow per file.
- Each file is fully self-contained; safe to retry.
- Milvus + Elasticsearch upsert by `frame_id` (idempotent).

## 10. Data quality checks

After indexing, run automated checks:
- **OCR sanity**: fraction of frames with at least 1 detected text > 30% (Vietnamese news has lots of overlays).
- **ASR sanity**: fraction of videos with at least 1 segment > 95%.
- **Embedding sanity**: random 100-query smoke test: top-1 self-retrieval = 1.0.
- **Caption sanity**: fraction of non-empty captions on sample > 95%; mean Vietnamese token count 30-100.

Failures generate an alert in our internal Slack and the offending video_id is re-queued.
