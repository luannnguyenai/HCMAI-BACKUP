#!/usr/bin/env bash
# Download (optional) + unzip + profile the AIC2025 proxy corpus (research-note 07).
#
#   scp infra/remote/profile_aic2025.{sh,py} aic2026-gpu:.
#   ssh aic2026-gpu 'bash profile_aic2025.sh "<drive-folder-url>"'
#
# Args:
#   $1 = Google Drive folder URL (optional; if omitted, assumes data already at $ROOT)
#   $2 = ROOT dir (default /tmp/aic2025)
#
# Notes:
#  * `gdown --folder` fetches top-level files (the 8 Keyframes_*.zip). Nested
#    subfolders (query/, video_batch_*/) may need their own --folder URLs; grab
#    at least query/ for the C1 calibration (research-note 07 / SPEC-0014 Q2).
#  * Pillow gives resolution stats; ffprobe gives video durations. Both optional.

set -u
URL="${1:-}"
ROOT="${2:-/tmp/aic2025}"
mkdir -p "$ROOT"

if [ -n "$URL" ]; then
  echo "== fetching $URL -> $ROOT =="
  uv run pip install -q gdown pillow 2>/dev/null || pip install -q gdown pillow
  uv run gdown --folder "$URL" -O "$ROOT" || gdown --folder "$URL" -O "$ROOT"
fi

# Unzip any keyframe archives that haven't been expanded yet.
shopt -s nullglob
for z in "$ROOT"/Keyframes_*.zip; do
  stem="$(basename "$z" .zip)"
  if [ ! -d "$ROOT/kf/$stem" ]; then
    echo "== unzip $z =="
    mkdir -p "$ROOT/kf/$stem"
    unzip -q "$z" -d "$ROOT/kf/$stem" || echo "  (unzip failed for $z)"
  fi
done

echo "== profiling =="
uv run python profile_aic2025.py --root "$ROOT" --out "$ROOT/profile.json"
rc=$?
echo "== profile exit code: $rc =="
exit $rc
