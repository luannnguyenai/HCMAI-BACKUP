#!/usr/bin/env bash
# One-off C1 (SPEC-0014) smoke runner for the GPU box.
# Avoids pasting long commands over SSH (spaces before `--` flags get mangled);
# scp this file to the box and run it instead:
#   scp infra/remote/c1_smoke.sh aic2026-gpu:c1_smoke.sh
#   ssh -o ServerAliveInterval=30 aic2026-gpu 'bash c1_smoke.sh'
# Optional first arg overrides the provisioned dir's short SHA (default 7f18c88).

SHA7="${1:-7f18c88}"
DIR="$HOME/aic2026/$SHA7"

export PATH="$HOME/.local/bin:$PATH"
# Disable the Rust tokenizers thread pool: it races the GIL during interpreter
# shutdown (Fatal Python error: PyGILState_Release ... finalizing).
export TOKENIZERS_PARALLELISM=false

cd "$DIR" || { echo "ERROR: no provisioned dir $DIR"; exit 1; }

# Export env (HF_TOKEN, R2_*) for the training subprocesses. Sourcing tolerates
# the .env.remote line-20 quirk; we don't want it to abort the run.
set -a
. ./.env.remote 2>/dev/null || true
set +a

# Corpus is deterministic + already built on the first run; skip the rebuild.
if [ -f /tmp/c1/pairs.parquet ]; then
  echo "== c1-corpus (cached: /tmp/c1/pairs.parquet already exists) =="
else
  echo "== c1-corpus =="
  # Same at-exit GIL/tokenizers race as c1-fit: the corpus prints `OK ...` and
  # writes the parquet, then non-zero-exits during interpreter cleanup. Trust
  # the parquet file's existence, not the exit code.
  uv run train c1-corpus --out /tmp/c1/pairs.parquet --max-per-source 5000 || true
  echo "== c1-corpus parquet check =="
  ls -la /tmp/c1/pairs.parquet 2>/dev/null || { echo "corpus FAILED (no parquet)"; exit 1; }
fi

echo "== c1-fit =="
uv run train c1-fit --pairs /tmp/c1/pairs.parquet --out-dir /tmp/c1/run --max-steps 2000 --batch-size 128
fit_rc=$?
echo "== c1-fit exit code: $fit_rc =="

# Always report the checkpoint state, even if the process crashed at exit:
# a finalization-time crash still leaves head.pt + train_meta.json written.
echo "== /tmp/c1/run =="
ls -la /tmp/c1/run 2>/dev/null || echo "(no /tmp/c1/run)"
echo "== train_meta.json =="
cat /tmp/c1/run/train_meta.json 2>/dev/null || echo "(no train_meta.json - training did not finish)"
