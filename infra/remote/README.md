# infra/remote/ - runbook for a fresh GPU lease

> Lives outside `src/` because these are operations scripts, not application
> code. The SDD rule "anything in `src/` is spec-gated" does not apply here;
> the lifecycle they orchestrate is captured in [SPEC-0022](../../docs/specs/SPEC-0022-remote-gpu-runner.md).

## When you get a new 1-2 week GPU lease

1. **Add the cluster to your local `~/.ssh/config`** as alias `aic2026-gpu`
   (or whatever you set `AIC2026_REMOTE_SSH_HOST` to). Confirm with
   `ssh aic2026-gpu echo ok`.

2. **One-time bootstrap on the login node**:

   ```bash
   scp infra/remote/bootstrap.sh aic2026-gpu:~/bootstrap.sh
   ssh aic2026-gpu bash ~/bootstrap.sh
   ```

   This installs `uv` if missing, makes `~/aic2026/` writable, and exits.
   No secrets are stored on the cluster by this script.

3. **From your laptop**:

   ```bash
   cp .env.remote.example .env.remote
   $EDITOR .env.remote      # fill in R2_*, HF_TOKEN, AIC2026_REMOTE_REPO_URL
   set -a && source .env.remote && set +a

   ./bin/remote setup
   ./bin/remote provision --sha $(git rev-parse HEAD)
   ./bin/remote run extract-siglip --launcher srun --config input_dir=/scratch/sample_frames
   ./bin/remote list --limit 5
   ./bin/remote pull <run_id>
   ```

## When the lease ends

```bash
./bin/remote teardown --confirm
```

(R2 artifacts persist by design - ADR-0011 four-tier model. Nothing on the
cluster survives.)

## Files in this directory

- `bootstrap.sh` - one-time login-node setup; idempotent.
- `slurm.sbatch.tpl` - template for `--launcher sbatch` (queued submission).
- `README.md` - this file.

## Cross-references

- [SPEC-0022](../../docs/specs/SPEC-0022-remote-gpu-runner.md) - the spec.
- [ADR-0011](../../docs/adr/ADR-0011-r2-artifact-store-and-lease-rollover.md) - the artifact-store decision.
- [ADR-0003](../../docs/adr/ADR-0003-rtx5070-finals-gh200-offline.md) - the underlying hardware split.
