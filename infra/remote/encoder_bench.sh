#!/usr/bin/env bash
# Encoder bake-off (SPEC-0025) runner for the GPU box.
# Unzips the AIC2025 query texts + provided-CLIP features, then runs
# `bin/embed bench` over a keyframe sample with the 4 encoders -> a
# deployability table + an HTML contact sheet.
#
#   scp -r src/aic2026/embedding src/aic2026/eval src/aic2026/cli/embed.py \
#       aic2026-gpu:aic2026/<dir>/src/aic2026/...   # get the bench code onto the box
#   scp infra/remote/encoder_bench.sh aic2026-gpu:.
#   ssh aic2026-gpu 'bash encoder_bench.sh <project-dir> [n_docs] [encoders]'
#
# Args:
#   $1 = project dir with the bench code + `embedding` extra synced
#        (default ~/aic2026/HEAD)
#   $2 = n_docs keyframe sample (default 20000)
#   $3 = encoders csv (default siglip2,metaclip2,qwen3vl,provided)
#
# Prereq once per dir: `cd <dir> && uv sync --extra embedding` (torch/open_clip/
# transformers/bitsandbytes). If Qwen fails to load, bump transformers per the
# HF model card (SPEC-0025 Q2) and drop qwen3vl from the encoders csv to proceed.

set -u
DIR="${1:-$HOME/aic2026/HEAD}"
N_DOCS="${2:-20000}"
ENCODERS="${3:-siglip2,metaclip2,qwen3vl,provided}"
ROOT=/tmp/aic2025

export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found on PATH"; exit 1; }
[ -d "$DIR" ] || { echo "ERROR: no project dir $DIR"; exit 1; }

# --- prepare query texts: unzip query-p*.zip -> a dir of *kis*.txt ---
QTXT="$ROOT/query_txt"
mkdir -p "$QTXT"
shopt -s nullglob
for z in "$ROOT"/query/query-p*.zip; do unzip -oq "$z" -d "$QTXT" || true; done
echo "== query texts: $(find "$QTXT" -name '*kis*.txt' | wc -l) KIS files =="

# --- prepare provided-CLIP features: unzip the features zip ---
PROVIDED="$ROOT/provided_clip"
if [ ! -d "$PROVIDED" ]; then
  mkdir -p "$PROVIDED"
  for z in "$ROOT"/video_batch_1/clip-features-32-aic25-b1.zip; do
    [ -f "$z" ] && { echo "== unzip $z =="; unzip -oq "$z" -d "$PROVIDED" || true; }
  done
fi
echo "== provided-CLIP layout (first entries) =="
find "$PROVIDED" -maxdepth 2 | head -10

# --- Qwen3-VL-Embedding official repo (its .process() API, not AutoModel) ---
QWEN_SRC="$HOME/Qwen3-VL-Embedding"
QWEN_ARG=()
if printf '%s' "$ENCODERS" | grep -q qwen3vl; then
  [ -d "$QWEN_SRC" ] || git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding "$QWEN_SRC" || true
  cd "$DIR" && uv pip install -q qwen-vl-utils 2>/dev/null || true
  QWEN_ARG=(--qwen-impl-src "$QWEN_SRC")
  echo "== qwen impl src: $QWEN_SRC =="
fi

# --- run the bench ---
cd "$DIR" || exit 1
mkdir -p "$ROOT/bench"
echo "== embed bench (encoders=$ENCODERS, n_docs=$N_DOCS) =="
uv run embed bench \
  --kf-root "$ROOT/kf" \
  --queries "$QTXT" \
  --encoders "$ENCODERS" \
  --provided-features "$PROVIDED" \
  "${QWEN_ARG[@]}" \
  --n-docs "$N_DOCS" \
  --top-k 5 \
  --max-queries 20 \
  --device cuda \
  --out "$ROOT/bench"
rc=$?
echo "== bench exit code: $rc =="
echo "== deployability.json =="
cat "$ROOT/bench/deployability.json" 2>/dev/null || echo "(none)"
echo "== report: $ROOT/bench/bench_report.html =="
exit $rc
