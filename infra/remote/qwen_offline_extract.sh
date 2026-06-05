#!/usr/bin/env bash
# Implements SPEC-0004 SS 3 (offline extraction CLI) + ADR-0012.
# Offline visual-document lane runner for the GPU box: clones the official
# QwenLM/Qwen3-VL-Embedding repo (its `.process()` API, not AutoModel) and runs
# `bin/embed images --encoder qwen3vl` over a keyframe dir, writing the
# pre-indexed dense lane (.npy + manifest). Qwen3-VL-Embedding-2B is OFFLINE
# ONLY (never the online query encoder); see ADR-0012 + SPEC-0025.
#
#   scp infra/remote/qwen_offline_extract.sh aic2026-gpu:.
#   ssh aic2026-gpu 'bash qwen_offline_extract.sh <project-dir> <kf-dir> <out-base> [out-dim]'
#
# Args / env:
#   $1 / DIR      = project dir with the bench code + `embedding` extra synced
#                   (default ~/aic2026/HEAD)
#   $2 / KF_DIR   = keyframe input dir (*.jpg/jpeg/png/webp)
#                   (default /tmp/aic2025/kf)
#   $3 / OUT_BASE = output base path; writes <base>.npy + <base>.manifest.jsonl
#                   (default /tmp/aic2025/qwen/qwen3vl)
#   $4 / OUT_DIM  = optional MRL truncation width (default: native 2048)
#
# Prereq once per dir: `cd <dir> && uv sync --extra embedding` (torch/
# transformers/pillow) plus `uv pip install qwen-vl-utils`.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

DIR="${1:-${DIR:-$HOME/aic2026/HEAD}}"
KF_DIR="${2:-${KF_DIR:-/tmp/aic2025/kf}}"
OUT_BASE="${3:-${OUT_BASE:-/tmp/aic2025/qwen/qwen3vl}}"
OUT_DIM="${4:-${OUT_DIM:-}}"
QWEN_SRC="${QWEN_SRC:-$HOME/Qwen3-VL-Embedding}"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found on PATH"; exit 1; }
[ -d "$DIR" ] || { echo "ERROR: no project dir $DIR"; exit 1; }
[ -d "$KF_DIR" ] || { echo "ERROR: no keyframe dir $KF_DIR"; exit 1; }

# --- Qwen3-VL-Embedding official repo (its .process() API, not AutoModel) ---
if [ ! -d "$QWEN_SRC" ]; then
  echo "== cloning QwenLM/Qwen3-VL-Embedding -> $QWEN_SRC =="
  git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding "$QWEN_SRC"
fi
cd "$DIR" && uv pip install -q qwen-vl-utils 2>/dev/null || true
echo "== qwen impl src: $QWEN_SRC =="

# --- run the offline extraction ---
mkdir -p "$(dirname "$OUT_BASE")"
OUT_DIM_ARG=()
[ -n "$OUT_DIM" ] && OUT_DIM_ARG=(--out-dim "$OUT_DIM")

echo "== embed images (encoder=qwen3vl, kf=$KF_DIR, out=$OUT_BASE) =="
./bin/embed images \
  --input "$KF_DIR" \
  --output "$OUT_BASE" \
  --encoder qwen3vl \
  --impl-src "$QWEN_SRC" \
  "${OUT_DIM_ARG[@]}"
rc=$?

echo "== extract exit code: $rc =="
echo "== outputs: ${OUT_BASE}.npy + ${OUT_BASE}.manifest.jsonl =="
echo "NOTE: the .npy + manifest are the input to SPEC-0006 ingestion (the"
echo "      optional qwen3vl dense field for the offline visual-document lane)."
exit $rc
