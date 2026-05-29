#!/usr/bin/env bash
#SBATCH --job-name=aic2026
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# SPEC-0022 - SLURM batch template. `bin/remote run --launcher sbatch`
# renders this file by replacing `{{ CMD }}` (and any other variables) with
# the actual job command via plain str.replace. Keep the placeholders
# `{{ NAME }}`-style and do NOT introduce a Jinja2 dep on the cluster.
set -euo pipefail

echo "[sbatch] host:        $(hostname)"
echo "[sbatch] SLURM_JOB_ID: ${SLURM_JOB_ID:-(none)}"
echo "[sbatch] starting at:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

{{ CMD }}

echo "[sbatch] finished at:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
