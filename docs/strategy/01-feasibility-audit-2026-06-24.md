# Feasibility Audit - 2026-06-24

> Scope: Feasibility check for the AIC2026-SCOAI plan as of 2026-06-24
> (Asia/Saigon), using the local repository state at `31f36f7` and the
> official competition website.

## Verdict

**Feasible, but compressed.** The target is still credible if the team treats
the 2026-06-25 dataset/rules release as the start of a focused Phase 1
implementation sprint. The repo has a useful spec-driven foundation, a working
evaluation harness, remote GPU/R2 infrastructure, and a materially advanced C1
DiacriticBERT workstream. The critical gap is that the competition-facing
retrieval product is still mostly architecture and reserved specs rather than
integrated runtime.

## External Schedule Check

The official competition page still lists:

- Registration from launch until 2026-06-15.
- Preliminary-round content and requirements expected on 2026-06-25.
- Training sessions expected in June and July 2026.
- Preliminary round expected in August 2026.
- Preliminary results expected on 2026-08-30.
- Finals expected from 2026-09-12 through 2026-09-26.

Source: <https://aichallenge.hochiminhcity.gov.vn/>, section "Thoi gian, tien
do du kien trien khai cac vong thi" and "Noi dung", checked 2026-06-24.

## Repository Evidence

### What is already real

- **SDD workflow is strong.** `AGENTS.md`, `CONTRIBUTING.md`, spec registry,
  ADR registry, and source headers make scope control unusually good for a
  competition repo.
- **Eval harness exists.** `bin/eval` / `aic2026.cli.eval` runs the 20-task
  smoke set against a deterministic stub backend and emits `metrics.json`,
  `report.html`, and provenance.
- **Ranking metrics exist.** SPEC-0020 adds NDCG@10, which is necessary for the
  C2 learned-fusion gate.
- **Remote execution is unusually mature.** `bin/remote`, R2 artifact sync,
  warm-cache restore, run manifests, and job registry reduce the risk of
  losing work on ephemeral GPU leases.
- **C1 has momentum.** SPEC-0014 includes noise schedules, corpus building,
  head training, offline eval, R2-backed remote jobs, and demo tooling.

### What is not yet real

- **No end-to-end retrieval system yet.** Milvus, Elasticsearch, OCR/ASR
  ingestion, DRES integration, planner, reranker, UI, and operator-trace logger
  are still reserved or draft specs, not integrated runtime.
- **No real-data ingestion contract yet.** SPEC-0003 is reserved; the June 25
  dataset must be turned into an approved ingestion spec immediately.
- **C2 and C4 are blocked.** C2 needs ranked lists over a dev set; C4 needs
  operator traces. Neither can progress meaningfully until ingestion, UI, and
  trace logging exist.
- **Local developer ergonomics need cleanup.** The WSL shell lacks `uv` on
  `PATH`; the existing Windows `.venv` works, but moto/R2 tests fail locally
  because Windows cannot connect to `0.0.0.0:<port>`.

## Feasibility By Workstream

| Workstream | Status | Feasibility | Audit note |
|---|---|---|---|
| Strategy and governance | Strong | High | SDD and ADR discipline are already in place. |
| Eval harness | Partial but usable | High | Good enough for smoke and C1/C2 gates, but SPEC-0001 full AC5-AC8 remain unfinished. |
| Remote GPU/R2 | Implementing | High | Strong competitive advantage if credentials and lease playbooks stay current. |
| C1 DiacriticBERT | Implementing | High | Most mature technical edge. Keep it, but do not let it crowd out ingestion/UI. |
| Data ingestion | Reserved | Medium | Tomorrow's dataset release must produce SPEC-0003 and a minimal importer within 48 hours. |
| Milvus/Elasticsearch retrieval | Reserved | Medium | Core product risk. Needs first thin baseline before adding all model lanes. |
| DRES/submission path | Draft | Medium | SPEC-0018 exists; wire early to avoid finals-day surprises. |
| UI/operator console | Reserved | Medium-low | Operator moat depends on SPEC-0012 and SPEC-0013; these cannot wait until late Phase 2. |
| Planner/agent | Reserved | Medium-low | Bakeoff is useful, but a deterministic retrieval baseline should precede agent work. |
| C2 learned fusion | Reserved | Medium | Feasible after ranked-list logs exist; otherwise becomes paper-only. |
| C4 self-distillation | Reserved | Medium-low | Feasible only if trace logging starts during Phase 1 practice. |

## Critical Path

1. **2026-06-25 to 2026-06-27: dataset/rules intake.**
   Write the dataset-shape delta, approve SPEC-0003, and run a loader over a
   tiny sample.
2. **2026-06-27 to 2026-07-03: thin retrieval baseline.**
   Use organiser keyframes and metadata first; compute one SigLIP-2 lane; store
   vectors in a simple local baseline if Milvus is not ready.
3. **2026-07-03 to 2026-07-10: text lanes and submission path.**
   Add OCR/ASR/description indexing and wire SPEC-0018 DRES submit. This is
   more important than adding a second image encoder.
4. **2026-07-10 to 2026-07-17: operator loop.**
   Ship a minimal React console plus submission verification and trace logging.
5. **2026-07-17 to mid-August: quality work.**
   Add Meta CLIP 2, InternVideo2, C1 lane, reranker, C2, planner, and C4 in
   that order, gated by eval evidence.

## Go / No-Go Gates

| Date | Gate | Pass condition |
|---|---|---|
| 2026-06-27 | Data intake | SPEC-0003 approved and a sample loader produces canonical frame records. |
| 2026-07-03 | Retrieval smoke | One image lane returns top-10 for at least 20 real-data sanity queries. |
| 2026-07-10 | Text + submit | OCR/ASR/metadata lane exists and DRES submit can be tested end to end. |
| 2026-07-17 | Operator loop | UI can search, inspect neighbours, submit, and log traces. |
| 2026-08-01 | Differentiation | C1 integrated or explicitly deferred; C2 training data captured; C4 traces accumulating. |
| 2026-08-15 | Preliminary readiness | Full baseline beats the internal baseline by 20% or the team narrows scope for reliability. |

## Recommended Scope Control

- Build the first real-data baseline with **one image lane + organiser metadata**
  before adding every model in proposal 01.
- Treat **submission verification** as part of the MVP, not polish.
- Treat **operator-trace logging** as part of Phase 1, because C4 cannot be
  recovered late.
- Keep RRF as the live fallback until C2 has real held-out evidence.
- Do not start C3 or C5 unless Phase 1 gates are green.

## Documentation Refresh Actions

- Update `README.md` to reflect the real implemented surface and provide
  bilingual English/Vietnamese onboarding.
- Update `docs/strategy/00-master-strategy.md` so "today" and Phase 0 language
  no longer imply May 24.
- Keep accepted ADR decisions unchanged; create a future ADR only if the team
  changes a binding architectural choice.
