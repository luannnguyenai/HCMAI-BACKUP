---
id: ADR-0011
title: Cloudflare R2 as the artifact store; four-tier persistence across lease rollovers
status: Accepted
decided_on: 2026-05-29
deciders:
  - team lead
related_adrs:
  - ADR-0003
---

# ADR-0011 - Cloudflare R2 as the artifact store; four-tier persistence across lease rollovers

## Status

Accepted.

## Context

[ADR-0003](ADR-0003-rtx5070-finals-gh200-offline.md) places all heavy compute - image-tower embedding extraction, PhoWhisper / PaddleOCR, Qwen2.5-VL-72B captioning, C1/C2/C4 training, INT4/FP4 quantisation calibration - on a remote GPU box (GH200-class). In our case the remote is a **university / NVIDIA-internal allocation with a 1-2 week lease window**. When the lease ends the whole box is wiped; any artifact left only on the cluster is gone.

This forces a question the existing ADRs do not answer: **where do the things we produce on the remote actually live?** Trained LoRA adapters, embedding `.npy` files, captioning outputs, eval reports, and run-trace Parquet must survive across leases - otherwise we have to redo work every 1-2 weeks. The 2026-05-29 chat-channel walk-through enumerated four candidates: AWS S3, GCS, Cloudflare R2, self-hosted MinIO on a VPS, or no third-party store (SSH-pull tarballs into the local laptop). Each fails differently:

- **SSH-pull only**: if the lease ends mid-pull, or before we remember to pull, the artifact is lost.
- **AWS S3 / GCS**: mature, but **egress fees** bite specifically the pattern we will hit hardest - pulling artifacts back to laptops dozens of times during paper-writing and ablation iteration.
- **MinIO VPS**: full control, but adds one more babysat box to a team already stretched.
- **R2**: S3-compatible API (`boto3` with `endpoint_url=...`), **free egress**, ~$0.015/GB stored.

The repo also has a strong existing discipline against committing secrets ([AGENTS.md](../../AGENTS.md) SS "Things you must not do" item 5; `.gitignore` already excludes `.env*`, `secrets/`, `*.pem`, `*.key`). The artifact-store choice must extend that discipline rather than break it.

## Decision

We adopt **Cloudflare R2** as the canonical artifact store, and a **four-tier persistence model** across lease rollovers:

| Tier | Lives on | Survives a lease rollover? | What goes here |
|---|---|---|---|
| **Repo** | git + GitHub | yes, forever | code, specs, ADRs, `pyproject.toml`, `uv.lock`, small mock tasks, config templates |
| **Artifact store** | Cloudflare R2 bucket | yes | embedding `.npy` files, LoRA adapters, training logs, `metrics.json`, run-trace Parquet, the run ledger |
| **Remote-ephemeral** | leased cluster | no - wiped when lease ends | `uv` venv, intermediate dataset copies, scratch dirs, HuggingFace cache |
| **Local dev** | laptop | yes | `.env.remote` (gitignored), pulled-back artifacts, the `bin/remote` CLI |

R2 is accessed via `boto3` with `endpoint_url=https://<account>.r2.cloudflarestorage.com`; the existing `.gitignore` patterns already cover the credentials surface. Run provenance is captured in an append-only ledger at `s3://<bucket>/manifest/` (one object per ledger entry, rather than a single JSONL file, to avoid append races between concurrent jobs). Each ledger entry pins `(run_id, git_sha, job, started_at, finished_at, exit_code, blobs[])` so any artifact can be re-derived from `git_sha` + the same input data.

Credentials propagation: SSH access via the existing `~/.ssh/config` `Host aic2026-gpu` alias (no new key dance); R2 keys and HF token via a local `.env.remote` (gitignored); both are exported to the remote at job-launch time over SSH and never written to disk on the cluster.

The companion implementation lives in [SPEC-0022](../specs/SPEC-0022-remote-gpu-runner.md).

## Consequences

### Positive

- **Free egress** removes the cost-of-iteration penalty for pulling artifacts back to laptops, which is exactly what we will do during ablation iteration and paper-writing.
- **S3-compatible API** means we are not locked into Cloudflare - if R2 ever becomes a problem we can move to any other S3 backend by swapping `endpoint_url`.
- **One persistent store across multiple leases** is the actual mechanism that lets us accumulate work without re-doing it every 1-2 weeks.
- **Single ledger schema** for all jobs (embedding extraction, captioning, training, eval) keeps `bin/remote list` / `pull` uniform across job types.
- **Clean separation of secrets from repo**: extends the existing `.env` / `secrets/` gitignore patterns; no new pattern needed.

### Negative

- **One new external account** to provision and rotate keys for. A team-wide R2 token rotation policy is a follow-up.
- **One new Python dependency** (`boto3`, ~7 MB). Light, but non-zero CI cost.
- **No transactional guarantees** across the ledger and the artifact blobs - a job can crash after uploading some blobs and before appending the ledger entry, leaving orphan blobs. We accept this and add a periodic `bin/remote prune` follow-up to GC unreferenced prefixes.
- We are coupling to Cloudflare's R2 availability for read-back; for finals-day demos we will pull the production hot-path weights to the laptop ahead of time (the 12 GB RTX 5070 path is offline anyway per ADR-0003).

### Neutral / observable

- The cluster login node needs outbound HTTPS to `*.r2.cloudflarestorage.com` and `huggingface.co`; this is the same egress allow-list a normal `pip install` already needs.
- R2 cost is observable but not free: at ~1 TB stored, ~$15/month. We will track this manually for the first two months and revisit if it grows.
- The ledger lives under `manifest/` as one object per entry (timestamp + run_id in the key) so concurrent jobs do not race on a JSONL append. The "single JSONL file" form is a future optimisation if list cost ever matters.
- **`bin/remote` becomes the standard interface** for any future GPU job (Meta CLIP 2 extraction, InternVideo2 extraction, C1 DiacriticBERT training, Qwen-VL captioning); the API contract in SPEC-0022 should be respected by all of them.

## Alternatives considered

- **AWS S3** - mature, well-known, team has prior credentials - **rejected because egress fees** dominate the iteration pattern (laptop pulls), and the operational gain over R2 is minimal.
- **Google Cloud Storage** - mature, fast, good Python SDK - **rejected for the same egress reason** as S3.
- **Self-hosted MinIO on a small VPS** - cheap, fully under team control, S3 API - **rejected because** it adds one more box to babysit (security patching, uptime monitoring, capacity) on a team that is stretched and on a deadline.
- **SSH-pull tarballs into the laptop, no third-party store** - zero new accounts, uses the existing SSH stack - **rejected because** the 1-2 week lease window means any artifact we forget to pull dies with the lease; the whole point of the store is to outlive the lease.

## References

- [`SPEC-0022`](../specs/SPEC-0022-remote-gpu-runner.md) - the implementation spec.
- [`ADR-0003`](ADR-0003-rtx5070-finals-gh200-offline.md) - the hardware split that makes a separate offline-artifact tier necessary.
- Cloudflare R2 pricing: <https://developers.cloudflare.com/r2/pricing/>
- `boto3` against R2 endpoint: <https://developers.cloudflare.com/r2/api/s3/api/>
- `.gitignore` of this repo - existing secrets-handling discipline this ADR extends.
