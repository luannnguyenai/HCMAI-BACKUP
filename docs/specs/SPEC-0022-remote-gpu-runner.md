---
id: SPEC-0022
title: Remote GPU job runner with Cloudflare R2 artifact sync
status: Implementing
owner: unassigned
created: 2026-05-29
updated: 2026-05-29
implements_proposal: docs/proposals/05-evaluation-harness.md SS 5
related_adrs:
  - ADR-0003
  - ADR-0011
depends_on:
  - SPEC-0001
  - SPEC-0004
---

# SPEC-0022 - Remote GPU job runner with Cloudflare R2 artifact sync

> Ships `bin/remote`: the single CLI that packages a job, runs it on an ephemeral (1-2 week) leased GPU box, and syncs the results to Cloudflare R2 so they survive the lease rollover. This is the substrate every offline-heavy spec (SPEC-0004 image embedding, future SPEC-0014 C1 training, captioning, quantisation) plugs into.

## 1. Context

[ADR-0003](../adr/ADR-0003-rtx5070-finals-gh200-offline.md) places image-tower embedding, PhoWhisper, PaddleOCR, Qwen-VL captioning, C1/C2/C4 training, and INT4/FP4 quantisation calibration on a remote GPU box. The team's allocation comes as a **1-2 week lease** that is then revoked; any artifact left only on the lease box is gone. [ADR-0011](../adr/ADR-0011-r2-artifact-store-and-lease-rollover.md) commits us to Cloudflare R2 + a four-tier persistence model (git, R2, cluster-ephemeral, local) and this spec is the implementation.

[SPEC-0001](SPEC-0001-evaluation-harness.md) gives us the `bin/eval` pattern + provenance discipline (every run pinned to `git_sha`, deterministic `run_id`, machine-readable `metrics.json`). [SPEC-0004](SPEC-0004-image-embedding-service.md) gives us a real `SigLip2Embedder` and `extract_image_embeddings` that produce `.npy` + manifest. This spec stitches them together for execution on the remote.

## 2. Scope

### 2.1 In scope

- A new `bin/remote` CLI with six subcommands: `setup`, `provision`, `run`, `pull`, `list`, `teardown`.
- A `RunContext` Pydantic model with stable `run_id` and provenance fields.
- A `ManifestEntry` Pydantic model + append-only R2 ledger at `manifest/<key>.json` (one object per entry to avoid append races).
- An `R2Client` wrapping `boto3` against R2's `endpoint_url`, with `upload_dir / list / download_dir`.
- An `ssh_exec` wrapper around `subprocess` for cluster ops.
- Four launchers: `srun` (default; SLURM interactive), `sbatch` (SLURM queued, via templated `infra/remote/slurm.sbatch.tpl`), `ssh` (login-node only), `local` (no remote at all - testing).
- A job registry (`@register("name")`) + one job (`extract-siglip`) wiring SPEC-0004's `SigLip2Embedder` end to end.
- `.env.remote.example` committed; real `.env.remote` gitignored.
- Six CPU-safe AC tests using `moto` for R2 mocking and `subprocess.run` mocks for SSH.

### 2.2 Out of scope

- Meta CLIP 2 / InternVideo2 / Qwen-VL / training jobs - each lands as a follow-up job behind the same `bin/remote run <job>` interface.
- Cluster-discovery or lease-acquisition automation - we expect a working `~/.ssh/config Host aic2026-gpu` alias before `bin/remote setup`.
- Multi-tenant R2 (per-user prefixes, quotas) - single shared bucket for the team for now (ADR-0011 flags as follow-up).
- Cost accounting and orphan-blob GC - tracked manually; `bin/remote prune` is a follow-up spec.
- Cross-region replication of the bucket - R2 already handles durability; we do not double up.
- Submission to DRES (SPEC-0018) - separate path.

## 3. API contract / interface

```python
# aic2026/remote/context.py

class RunContext(BaseModel):
    """All the provenance + paths needed to execute one job."""

    run_id: str             # f"{git_sha[:7]}-{job_name}-{utc_ts}"
    git_sha: str            # full 40-char SHA from `git rev-parse HEAD`
    job_name: str
    started_at: datetime    # UTC
    local_run_dir: Path     # eval-results/remote/<run_id>/ (on the launching laptop)
    remote_run_dir: PurePosixPath   # ~/aic2026/<git_sha[:7]>/runs/<run_id>/ on cluster
    r2_prefix: str          # runs/<run_id>/   (no leading slash; bucket-relative)
```

```python
# aic2026/remote/manifest.py

class ManifestEntry(BaseModel):
    run_id: str
    git_sha: str
    job_name: str
    started_at: datetime
    finished_at: datetime | None
    exit_code: int | None
    r2_prefix: str
    blobs: list[str]        # bucket-relative keys, e.g. ["runs/<id>/v.npy", ...]
    env: dict[str, str]     # whitelisted env captured (e.g. SLURM_JOB_ID); never secrets

def append_to_r2(client: R2Client, entry: ManifestEntry) -> None: ...
def read_all(client: R2Client, *, limit: int = 100) -> list[ManifestEntry]: ...
```

```python
# aic2026/remote/r2.py

class R2Client:
    """Thin boto3 wrapper. Config from env or explicit kwargs:
       R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET."""

    def upload_dir(self, local: Path, prefix: str) -> list[str]: ...   # returns keys
    def list(self, prefix: str) -> list[str]: ...
    def download_dir(self, prefix: str, local: Path) -> list[Path]: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
```

```python
# aic2026/remote/registry.py

JobFn = Callable[[RunContext, dict[str, Any]], None]

def register(name: str) -> Callable[[JobFn], JobFn]: ...
def resolve(name: str) -> JobFn: ...   # raises KeyError(<helpful list>)
```

```
bin/remote setup
bin/remote provision --sha <git-sha>
bin/remote run <job> [--launcher srun|sbatch|ssh|local]
                     [--dry-run]
                     [--config key=value ...]
bin/remote pull <run_id> [--filter glob]
bin/remote list [--job NAME] [--limit N]
bin/remote teardown --confirm
```

## 4. Behaviour

- **`setup`**: read `.env.remote`; assert required keys present; `ssh <host> 'true'` round-trip; R2 `list_buckets()` succeeds; print a one-line diagnostic; exit 0.
- **`provision --sha <sha>`**: `ssh <host>` then `git clone --depth 1 --branch <sha-or-branch>` into `~/aic2026/<sha[:7]>/`, run `uv sync --frozen --extra embedding`, optionally pre-warm HF cache (`huggingface-cli download` for the SigLIP-2 checkpoint). Idempotent: if the directory already exists, exit 0 with a "already provisioned" note.
- **`run <job> --launcher srun [--dry-run] [--config k=v]`**: resolve job from the registry; construct `RunContext`; if `--dry-run`, **print** the ssh/srun/R2 actions that *would* be taken and exit 0 (no side effects). Otherwise: launch the job via the chosen launcher, capture `exit_code`, `upload_dir(local_run_dir, r2_prefix)`, append the `ManifestEntry`. Exit code follows the job's exit code.
- **`pull <run_id> [--filter glob]`**: list the ledger; find the entry; `download_dir(entry.r2_prefix, eval-results/remote/<run_id>/)`. Filter applies to bucket keys before download.
- **`list [--job NAME] [--limit N]`**: read the ledger; optionally filter; pretty-print the last N entries.
- **`teardown --confirm`**: ssh in; `rm -rf ~/aic2026/`. Refuses without `--confirm`.

**Failure modes:**

- **`.env.remote` missing or incomplete**: `setup` and `run` exit non-zero with a clear "missing key X; see `.env.remote.example`".
- **SSH unreachable**: `ssh_exec` raises with stderr; CLI catches and exits non-zero.
- **R2 unreachable**: same pattern; `boto3` exceptions surfaced.
- **Job fails on the remote**: the job's non-zero exit propagates through the launcher; we **still** upload whatever made it to `local_run_dir` (partial artifacts) and **still** append a `ManifestEntry` with `exit_code != 0` and `finished_at` set. Avoids "the job failed silently and we lost the partial work" pattern.
- **Crash between upload and ledger append**: the blobs land in R2 but no ledger entry references them. Accepted; `bin/remote prune` is a follow-up.

**Determinism:**

- `run_id` is pinned to `(git_sha[:7], job_name, utc_ts)` where `utc_ts` is generated once at the start of `run` and threaded through every step.
- Within `--dry-run`, `utc_ts` is fixed to a sentinel (`"DRYRUN"`) so the printed plan is reproducible.

## 5. Acceptance criteria

- **AC1**: `RunContext` produces a stable `run_id` matching `r"^[0-9a-f]{7}-[a-z0-9_-]+-\d{8}T\d{6}Z$"` and round-trips through `model_dump_json()` -> `model_validate_json()`. Verified in `tests/unit/test_remote_context_AC1.py`.
- **AC2**: `ManifestEntry`s appended via `append_to_r2(...)` are returned by `read_all(...)` in the same order they were added; verified against `moto` in `tests/unit/test_remote_manifest_AC2.py`.
- **AC3**: `R2Client.upload_dir` writes the expected keys to a `moto` bucket; `list` returns them; `download_dir` produces byte-identical local files; verified in `tests/unit/test_remote_r2_AC3.py`.
- **AC4**: `ssh_exec` returns a `CompletedProcess` on success; on non-zero exit it raises with the remote stderr in the message. Verified in `tests/unit/test_remote_ssh_AC4.py` via `subprocess.run` mocks.
- **AC5**: `registry.resolve(name)` returns the registered callable; an unknown name raises `KeyError` whose message contains the available names. Verified in `tests/unit/test_remote_registry_AC5.py`.
- **AC6**: `bin/remote run extract-siglip --dry-run` exits 0, prints the planned `ssh`/`srun`/R2 actions to stdout, and writes nothing to disk or the network. Verified in `tests/unit/test_remote_cli_dry_run_AC6.py`.
- **AC7** (manual; pasted into the PR by the user after a real cluster run): `bin/remote run extract-siglip --launcher srun --input-dir <small-sample-dir>` produces `runs/<run_id>/v.npy` + `runs/<run_id>/v.manifest.jsonl` in R2 plus a ledger entry; `bin/remote pull <run_id>` retrieves them locally.

## 6. Non-functional requirements

- **Latency**: harness overhead per run (excluding the job itself) <= 60 s wall-clock total: ssh handshake + R2 upload of <=1 GB of `.npy` artifacts.
- **Memory**: CLI < 200 MB resident.
- **Compatibility**: Python 3.11+, `boto3 >= 1.34`, `moto >= 5` (dev only). Cluster side requires `ssh`, `uv` (installed by `bootstrap.sh`).
- **Security**: secrets are never written to remote disk; they are exported to the remote process at launch time via `ssh <host> "VAR=val command..."` and live only in process env.

## 7. Dependencies

- **Internal**: SPEC-0001 (provenance pattern), SPEC-0004 (`SigLip2Embedder` + `extract_image_embeddings`).
- **External**: `boto3 >= 1.34` (core dep, ~7 MB); `moto >= 5` (dev group, for S3 mocking in tests).
- **Data**: none committed; tests use `tmp_path` fake files; AC7 uses a tiny (~20-file) sample directory the user supplies.

## 8. Test plan

- **Unit tests** (`tests/unit/`):
  - `test_remote_context_AC1.py` - schema, run_id format, round-trip.
  - `test_remote_manifest_AC2.py` - append + read against moto.
  - `test_remote_r2_AC3.py` - upload/list/download against moto.
  - `test_remote_ssh_AC4.py` - success path + non-zero exit path with mocked `subprocess.run`.
  - `test_remote_registry_AC5.py` - register / resolve / unknown-key error.
  - `test_remote_cli_dry_run_AC6.py` - Typer `CliRunner` for the dry-run path; assert exit 0 + plan in stdout + no side effects.
- **Manual (cluster)**: AC7 evidence pasted into the PR.

## 9. Open questions

- **Q1**: SLURM `srun --gpus=` syntax varies across cluster admin policies. The default in `launchers.launch_srun` is `--gpus=1 --time=02:00:00`; override via `--config srun_args="..."`. Cluster-specific defaults should live in `.env.remote` once the user confirms which cluster.
- **Q2**: Bucket layout: we use `runs/<run_id>/` for artifacts and `manifest/<run_id>.json` for ledger. If the team later wants per-user scoping (`users/<gh_handle>/runs/...`), add a `--prefix-root` flag - non-breaking.
- **Q3**: `provision` HF cache pre-warm is opt-in (`--prewarm`) for now because HF downloads are slow and the user may want to skip them on a quick provision. Default: do not pre-warm; rely on first-job lazy download.

## 10. Changelog

| Date | Author | Change |
|---|---|---|
| 2026-05-29 | implementer (user-directed) | Created; Draft -> Approved -> Implementing in one pass for solo work per CONTRIBUTING. Slice (skeleton + extract-siglip) ships in branch `spec/0022-remote-gpu-runner`. AC7 (real cluster run) pasted into the PR by the user. |
