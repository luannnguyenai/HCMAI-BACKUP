---
id: SPEC-0024
title: One-command provisioning - R2 warm-cache restore + lease-hardening
status: Implementing
owner: unassigned
created: 2026-05-30
updated: 2026-05-30
implements_proposal: docs/proposals/05-evaluation-harness.md SS 5
related_adrs:
  - ADR-0003
  - ADR-0011
depends_on:
  - SPEC-0001
  - SPEC-0004
  - SPEC-0022
---

# SPEC-0024 - One-command provisioning: R2 warm-cache restore + lease-hardening

> Turn a fresh, ephemeral GPU lease into a ready-to-run box with a single `bin/remote provision` command, by restoring warm caches (uv wheels + model weights) from R2 instead of re-downloading. Also fixes every merged-code bug the first real H200 lease surfaced. Closes the loop on ADR-0011: R2 is the only thing that survives a lease, so it must hold everything needed to rebuild a box fast.

## 1. Context

The first real lease (8x H200, 2026-05-30) proved the SPEC-0022 runner works but exposed that a fresh box still costs ~90 min of plumbing + downloads, and surfaced several merged-code bugs that CI structurally cannot catch (the GPU path is `importorskip`-skipped). The lease confirmed (research-note context, PR #9):

- `~` on the lease is wiped on rollover; **no persistent cluster storage** (user-confirmed). R2 is the only durable layer.
- **No container runtime** available (user-confirmed). So the answer is warm-cache restore, not images.
- The repeated cost that matters is `uv sync` re-downloading GBs of CUDA wheels; weights are already R2-backed by `cache-weights` (SPEC-0022 AC8).

This spec packages provisioning so the next lease is `provision` -> ready in minutes, and folds in the lease-discovered fixes.

## 2. Scope

### 2.1 In scope

- **`R2Client` checksum-compat fix** for real Cloudflare R2: construct the boto3 client with `request_checksum_calculation="when_required"` + `response_checksum_validation="when_required"`. Without this, `list_objects_v2` 404s with `NoSuchKey` on botocore >= 1.36 (Cloudflare-documented). This unblocks `bin/remote list`/`pull` and the restore path.
- **`cache-env` job**: mirror the uv wheel cache (`~/.cache/uv`) to R2 under `env-cache/uv-<arch>/` (arch-tagged: `x86_64` vs `aarch64` never clash).
- **Hardened `provision`** - one idempotent command:
  - push code from the laptop via **scoped** `git archive` (private-repo safe, no 208 MB of papers)
  - **PATH-prepend** (`export PATH=$HOME/.local/bin:$PATH`) on every remote command (uv not on the non-interactive PATH)
  - restore the uv cache from R2 (when present) via `uvx --from awscli aws s3 sync`, so `uv sync` is a cache hit
  - `uv sync --frozen --extra embedding`
  - optional `--restore-weights`: restore `weights/<repo>/` from R2 to a local weights dir
  - credentials passed to the box via `ssh_exec(env=...)` (shlex-quoted, never written to remote disk) - NOT fragile inline strings
- **`remote-job-exec` run_id auto-generation**: if `--run-id` is omitted, build a valid one (`<7hex>-<job>-<utcstamp>`) so manual invocation can't hit the `RunContext` format trap.
- **`cache-weights` default-list update**: add Meta CLIP 2 (`facebook/metaclip-2-worldwide-huge-quickgelu`, verified id), drop the SigLIP-2 "verify" caveat (the `timm/...` id worked on the lease), annotate InternVideo2 as gated (pre-accept HF terms).

### 2.2 Out of scope

- Container images (no runtime on the cluster).
- Caching the raw HF hub cache layout for fully transparent offline `from_pretrained` - this spec restores weights to a local dir; transparent-offline wiring (HF_HUB_OFFLINE + cache reconstruction) is SS 9 Q1.
- `bin/remote prune` (orphan-blob GC) - still a follow-up.
- Multi-tenant R2 prefixes.

## 3. API contract / interface

```python
# aic2026/remote/r2.py  (R2Client.__init__ change)
# boto3.client("s3", ..., config=Config(
#     request_checksum_calculation="when_required",
#     response_checksum_validation="when_required",
# ))

# aic2026/remote/jobs/cache_env.py
@register("cache-env")
def cache_env(ctx: RunContext, config: dict[str, Any], *, uv_cache_dir: Path | None = None) -> None:
    """Mirror the uv wheel cache to env-cache/uv-<arch>/ on R2."""
```

```
bin/remote provision --sha <sha>
                     [--restore-weights]      # also pull weights/<repo>/ from R2
                     [--no-restore-env]        # skip uv-cache restore (force fresh uv sync)
                     [--dry-run]

remote-job-exec <job> [--run-id <id>] <out_dir> [--config k=v ...]
                      # --run-id optional; auto-generated when omitted
```

## 4. Behaviour

- **provision, normal**: scoped archive push -> restore uv cache from `env-cache/uv-<arch>/` (skip if absent) -> `uv sync --frozen --extra embedding` (cache hit) -> optionally restore weights -> print "ready". Idempotent: re-running re-syncs without harm.
- **provision, `--dry-run`**: print the plan (ssh/scp/sync/uv-sync commands) and exit 0; no side effects.
- **provision, no uv cache in R2 yet**: falls back to a normal (slower) `uv sync`; logs that the cache was absent. So the first lease still works; subsequent ones are fast once `cache-env` has run.
- **cache-env**: detect arch (`platform.machine()`), locate the uv cache (`uv cache dir` or `~/.cache/uv`), mirror it to `env-cache/uv-<arch>/`, write a `.cache-meta.json`.
- **R2Client**: all operations (list/upload/download/get/put) go through one client carrying the checksum-compat Config. `list()` now works against real R2.
- **remote-job-exec**: `--run-id` omitted -> `make_run_id(git_sha_here_or_zeros, job_name)`.

## 5. Acceptance criteria

- **AC1**: `R2Client` is constructed with the checksum-compat `Config`; `R2Client.list(prefix)` returns keys against a moto backend (regression guard) and is documented as the real-R2 `NoSuchKey` fix. Verified in `tests/unit/test_remote_r2_checksum_AC1.py`.
- **AC2**: `cache-env` mirrors a fake uv-cache dir to `env-cache/uv-<arch>/` on R2 with a `.cache-meta.json`; arch-tagged prefix. Verified in `tests/unit/test_remote_cache_env_AC2.py` (moto + injected cache dir).
- **AC3**: `remote-job-exec` with no `--run-id` generates a `RunContext`-valid run_id; with `--run-id` it uses the passed value. Verified in `tests/unit/test_remote_job_exec_runid_AC3.py`.
- **AC4**: `bin/remote provision --dry-run` exits 0, prints the planned push/restore/sync steps, and performs no ssh/R2/scp side effects. Verified in `tests/unit/test_remote_provision_dry_run_AC4.py`.
- **AC5** (manual; pasted into PR): on a fresh lease, `bin/remote provision --sha <sha>` brings the box to ready (uv cache restored, `uv sync` a cache hit) in < 5 min, and `bin/remote list` works against real R2.

## 6. Non-functional requirements

- **Latency**: with a warm uv cache in R2, provision (excluding weight restore) completes in < 5 min on a fresh lease.
- **Security**: credentials reach the box only via `ssh_exec(env=...)` at exec time (shlex-quoted), never written to remote disk.
- **Compatibility**: works on x86_64 and aarch64 leases (arch-tagged cache; uv re-downloads any missing-arch wheels gracefully).

## 7. Dependencies

- **Internal**: SPEC-0001, SPEC-0004, SPEC-0022 (this hardens SPEC-0022's `provision` + extends its R2Client + job registry).
- **External**: `awscli` is invoked ephemerally via `uvx --from awscli` on the box (not a committed dependency); `moto[server]` (dev) for tests.

## 8. Test plan

- Unit (`tests/unit/`): `test_remote_r2_checksum_AC1.py`, `test_remote_cache_env_AC2.py`, `test_remote_job_exec_runid_AC3.py`, `test_remote_provision_dry_run_AC4.py`. All CPU-safe, moto + subprocess mocks.
- Manual (cluster): AC5 evidence pasted into the PR on the next lease.

## 9. Open questions

- **Q1**: Transparent offline weight loading. This spec restores `weights/<repo>/` to a local dir; making `from_pretrained(repo)` / `snapshot_download(repo)` resolve to it offline (HF_HUB_OFFLINE + cache reconstruction, or an `AIC2026_WEIGHTS_DIR` the encoders consult) is deferred. The flat mirror is sufficient for path-based loading now.
- **Q2**: uv-cache size management - the cache can grow; a periodic prune of `env-cache/` is a follow-up.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-30 | implementer (user-directed) | Created; Draft -> Approved -> Implementing in one pass per CONTRIBUTING. Packages provisioning into one command + R2 warm-cache restore; fixes the `R2Client.list()` R2-checksum bug + the run_id format trap surfaced on the H200 lease. |
