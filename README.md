# AIC2026 - SCOAI

> **EN:** Strategy, research, implementation specs, and early infrastructure for
> competing in **AI Challenge HCMC 2026**, targeting 1st prize in Bang A.
>
> **VI:** Kho chiến lược, nghiên cứu, đặc tả triển khai và hạ tầng ban đầu cho
> **Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026**, mục tiêu giải nhất Bảng A.

**Competition site / Trang cuộc thi:** <https://aichallenge.hochiminhcity.gov.vn/>

**Current status / Tình trạng hiện tại:** as of 2026-06-24, the plan is still
feasible but compressed. The official schedule still points to the preliminary
content and requirements release on 2026-06-25. See
[`docs/strategy/01-feasibility-audit-2026-06-24.md`](docs/strategy/01-feasibility-audit-2026-06-24.md).

**Problem / Bài toán:** build an intelligent assistant for deep retrieval over
large multimedia data: images, audio, and text. The format follows
Lifelog Search Challenge (LSC) and Video Browser Showdown (VBS), with both an
interactive track and a new automatic-agent track.

---

## Read This First / Đọc Trước

**EN**

1. [`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md)
   explains why this competition is winnable, what we are building, and where
   the schedule is tight.
2. [`docs/strategy/01-feasibility-audit-2026-06-24.md`](docs/strategy/01-feasibility-audit-2026-06-24.md)
   is the current feasibility checkpoint.
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md) define the
   Spec-Driven Development workflow.
4. [`docs/proposals/08-original-contributions.md`](docs/proposals/08-original-contributions.md)
   explains what is actually novel in our system.

**VI**

1. [`docs/strategy/00-master-strategy.md`](docs/strategy/00-master-strategy.md)
   giải thích vì sao mục tiêu khả thi, hệ thống cần xây gì, và phần nào đang
   bị nén thời gian.
2. [`docs/strategy/01-feasibility-audit-2026-06-24.md`](docs/strategy/01-feasibility-audit-2026-06-24.md)
   là bản kiểm toán khả thi mới nhất.
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) và [`AGENTS.md`](AGENTS.md) là quy trình
   Spec-Driven Development bắt buộc.
4. [`docs/proposals/08-original-contributions.md`](docs/proposals/08-original-contributions.md)
   mô tả phần đóng góp mới thật sự của đội.

Visual companions / Hình minh họa:
[`docs/illustrations/`](docs/illustrations/README.md).

![System Architecture](docs/illustrations/aic2026-system-architecture.png)

---

## Feasibility Snapshot / Ảnh Chụp Khả Thi

**EN:** Feasible, but compressed. The repo has a strong planning and
infrastructure base: SDD workflow, eval harness, remote GPU/R2 artifact flow,
and a mature C1 DiacriticBERT workstream. The main risk is that the
competition-facing retrieval product is not integrated yet: data ingestion,
Milvus/Elasticsearch, DRES submit, React operator console, planner, reranker,
and trace logger are still mostly specs or reserved specs.

**VI:** Khả thi, nhưng thời gian đã bị nén. Kho hiện có nền tảng tốt: quy trình
SDD, bộ eval, luồng chạy GPU từ xa với R2, và nhánh C1 DiacriticBERT khá chín.
Rủi ro chính là sản phẩm truy hồi để thi vẫn chưa được tích hợp: nhập dữ liệu,
Milvus/Elasticsearch, submit DRES, giao diện React cho operator, planner,
reranker và trace logger vẫn chủ yếu nằm ở mức spec hoặc spec đã giữ chỗ.

Immediate critical path / Đường găng trước mắt:

1. 2026-06-25 to 2026-06-27: validate released rules and dataset shape, then
   approve SPEC-0003.
2. 2026-06-27 to 2026-07-03: build one real-data image retrieval lane before
   expanding model breadth.
3. 2026-07-03 to 2026-07-10: add text lanes and DRES submission.
4. 2026-07-10 to 2026-07-17: ship the operator loop: UI, neighbour inspection,
   submission verification, and trace logging.

---

## Winning Hypothesis / Giả Thuyết Thắng

**EN**

- **Floor:** reproduce the 2026 finalist-grade Vietnamese multimodal stack:
  SigLIP-2, Meta CLIP 2, InternVideo2, Vintern, BGE-M3, Milvus,
  Elasticsearch, OCR/ASR, planner, reranker, and DRES flow.
- **Edge:** ship at least two of the three primary original contributions:
  C1 DiacriticBERT, C2 learned per-task fusion, and C4 agent self-distillation.
- **Moat:** operator drills plus a submission-verification panel. Operator
  skill is a documented score lever in LSC/VBS-style competitions.

**VI**

- **Nền:** tái tạo stack đa phương thức tiếng Việt đủ sức vào chung kết:
  SigLIP-2, Meta CLIP 2, InternVideo2, Vintern, BGE-M3, Milvus,
  Elasticsearch, OCR/ASR, planner, reranker và luồng DRES.
- **Lợi thế:** ship ít nhất hai trong ba đóng góp chính: C1 DiacriticBERT,
  C2 học cách fusion theo loại task, và C4 tự chưng cất agent từ trace của
  operator.
- **Hào lũy:** luyện operator nghiêm túc cộng với panel xác minh trước khi
  submit. Kỹ năng operator là đòn bẩy điểm số đã được ghi nhận trong LSC/VBS.

Primary original contributions / Đóng góp chính:

| ID | EN | VI | Status |
|---|---|---|---|
| C1 | DiacriticBERT for Vietnamese ASR/OCR noise | DiacriticBERT cho nhiễu dấu tiếng Việt từ ASR/OCR | Implementing |
| C2 | Per-task learned fusion with RRF fallback | Fusion học theo loại task, fallback về RRF | Reserved |
| C4 | Agent self-distillation from operator traces | Tự chưng cất agent từ trace operator | Reserved |

---

## Repository Map / Cấu Trúc Kho

```text
CONTRIBUTING.md                       # SDD workflow for humans
AGENTS.md                             # SDD workflow for AI assistants
pyproject.toml                        # uv-managed Python 3.11+ project
ruff.toml                             # lint and format config
uv.lock                               # pinned lockfile

src/aic2026/
  models/                             # Pydantic task, submission, metrics models
  harness/                            # eval backend protocol, runner, scoring
  reporting/                          # metrics.json, report.html, provenance
  embedding/                          # Embedder protocol, dummy, SigLIP-2 wrapper
  train/                              # C1 DiacriticBERT training utilities
  eval/                               # C1 and harness eval helpers
  remote/                             # remote GPU runner and R2 artifact support
  cli/                                # eval, embed, train, remote CLIs

tests/
  unit/                               # acceptance-criterion unit tests
  integration/                        # smoke eval subprocess tests
  mock_tasks/                         # 20-task smoke corpus

docs/
  strategy/                           # master strategy and feasibility audits
  proposals/                          # architecture-level proposals
  specs/                              # component-level behavior contracts
  adr/                                # accepted architectural decisions
  research-notes/                     # cited background research
  papers/                             # downloaded reference papers

infra/remote/                         # GPU lease helper scripts
eval-results/                         # generated eval outputs, per-run dirs ignored
experiments/                          # experiment workspaces
```

---

## What Is Implemented / Phần Đã Có

**EN**

- `eval` CLI: runs the smoke task set against a deterministic stub backend and
  writes `metrics.json`, `report.html`, and provenance.
- Ranking metrics: R@1, R@5, R@10, MRR, NDCG@10, task slices, latency fields.
- Embedding skeleton: `Embedder`, `DummyEmbedder`, SigLIP-2 wrapper, and
  `embed` CLI.
- C1 tooling: noise functions, corpus builder, head training/eval helpers,
  C1 demo, and remote jobs.
- Remote execution: `remote` CLI, SSH launchers, Cloudflare R2 client,
  manifest ledger, cache restore/provisioning path.

**VI**

- CLI `eval`: chạy bộ smoke task với stub backend xác định và ghi
  `metrics.json`, `report.html`, cùng provenance.
- Metric truy hồi: R@1, R@5, R@10, MRR, NDCG@10, lát cắt theo task, trường
  latency.
- Khung embedding: `Embedder`, `DummyEmbedder`, wrapper SigLIP-2 và CLI `embed`.
- Công cụ C1: hàm tạo nhiễu, tạo corpus, train/eval head, demo C1 và remote job.
- Chạy từ xa: CLI `remote`, launcher SSH/SLURM, client Cloudflare R2, ledger
  manifest, và luồng restore cache/provision.

---

## What Is Still Spec-Only / Phần Còn Ở Mức Spec

**EN:** The following are critical and not yet integrated runtime: SPEC-0003
data ingestion, SPEC-0005 OCR/ASR ingestion, SPEC-0006 Milvus, SPEC-0007
Elasticsearch, SPEC-0008 planner, SPEC-0010 reranker, SPEC-0012 React console,
SPEC-0013 submission verification, SPEC-0018 DRES integration, and SPEC-0019
operator trace logger.

**VI:** Các phần sau là đường găng nhưng chưa thành runtime tích hợp:
SPEC-0003 nhập dữ liệu, SPEC-0005 OCR/ASR, SPEC-0006 Milvus, SPEC-0007
Elasticsearch, SPEC-0008 planner, SPEC-0010 reranker, SPEC-0012 giao diện React,
SPEC-0013 xác minh submit, SPEC-0018 tích hợp DRES, và SPEC-0019 trace logger
cho operator.

---

## Getting Started / Bắt Đầu Phát Triển

Canonical setup / Cách chuẩn:

```bash
uv sync --dev
uv run pytest -q
uv run eval --tasks tests/mock_tasks/smoke_20.jsonl --system my-experiment --no-latency-sim
```

Optional extras / Gói tùy chọn:

```bash
uv sync --extra embedding   # real image embedding backbones
uv sync --extra train       # C1 DiacriticBERT training path
```

Quality gates / Cổng chất lượng:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run eval --tasks tests/mock_tasks/smoke_20.jsonl --system "local-smoke"
```

**EN:** If `uv` is not on `PATH`, install/configure `uv` first. Do not commit
local virtualenvs, generated eval outputs, model weights, credentials, or
datasets.

**VI:** Nếu `uv` chưa có trong `PATH`, hãy cài/cấu hình `uv` trước. Không commit
virtualenv cục bộ, output eval, model weight, credential hoặc dataset.

---

## Remote GPU Work / Chạy GPU Từ Xa

**EN:** Heavy offline work runs on ephemeral GPU leases and persists artifacts
to Cloudflare R2, per ADR-0011. Configure a gitignored `.env.remote` from
`.env.remote.example`, then use:

**VI:** Việc nặng chạy trên GPU lease tạm thời và lưu artifact bền vững lên
Cloudflare R2 theo ADR-0011. Tạo `.env.remote` từ `.env.remote.example` (file
này bị gitignore), rồi dùng:

```bash
uv run remote setup
uv run remote provision --sha <git-sha>
uv run remote run <job> --dry-run
uv run remote list
uv run remote pull <run-id>
```

Never commit `.env.remote`, SSH keys, API tokens, model weights, or downloaded
competition data.

Không bao giờ commit `.env.remote`, SSH key, API token, model weight hoặc dữ
liệu cuộc thi đã tải.

---

## How To Contribute / Cách Đóng Góp

**EN**

This repo follows **Spec-Driven Development**:

1. Find the matching spec in [`docs/specs/INDEX.md`](docs/specs/INDEX.md).
2. If no spec exists, write the spec first using
   [`docs/specs/template.md`](docs/specs/template.md).
3. Read related ADRs before implementation.
4. Name tests after acceptance criteria, for example `test_..._AC2`.
5. Branch as `spec/NNNN-short-name`.
6. Use commit subjects like `[SPEC-NNNN] short description`.

**VI**

Kho này theo **Spec-Driven Development**:

1. Tìm spec phù hợp trong [`docs/specs/INDEX.md`](docs/specs/INDEX.md).
2. Nếu chưa có spec, viết spec trước bằng
   [`docs/specs/template.md`](docs/specs/template.md).
3. Đọc ADR liên quan trước khi triển khai.
4. Đặt tên test theo acceptance criteria, ví dụ `test_..._AC2`.
5. Tạo branch dạng `spec/NNNN-short-name`.
6. Dùng commit subject dạng `[SPEC-NNNN] mô tả ngắn`.

The one-line rule / Quy tắc một dòng:

> No code without a spec. No decision without a record.
>
> Không viết code khi chưa có spec. Không quyết định kiến trúc khi chưa có
> bản ghi.

---

## License / Giấy Phép

**EN:** Internal team artifact. Cited papers remain the property of their
authors; consult each paper for its license.

**VI:** Đây là tài liệu nội bộ của đội. Các paper được trích dẫn thuộc quyền
của tác giả tương ứng; hãy xem từng paper để biết giấy phép cụ thể.
