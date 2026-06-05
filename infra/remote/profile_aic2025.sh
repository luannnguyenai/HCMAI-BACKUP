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

# Non-interactive ssh (`ssh host 'bash profile_aic2025.sh ...'`) doesn't load the
# login PATH, so uv (installed at ~/.local/bin) isn't found. Export it explicitly,
# matching c1_eval.sh / c1_smoke.sh / c1_demo.sh.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on PATH; expected ~/.local/bin/uv" >&2
  exit 1
fi

mkdir -p "$ROOT"

if [ -n "$URL" ]; then
  echo "== fetching $URL -> $ROOT =="
  # `uv run --with` resolves gdown into an ephemeral env (no global pip install,
  # so no PEP 668 externally-managed-environment failure on the box's system pip).
  uv run --with gdown gdown --folder "$URL" -O "$ROOT"
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
# pillow is optional in the .py; --with enables the resolution histogram.
uv run --with pillow --with openpyxl python profile_aic2025.py --root "$ROOT" --out "$ROOT/profile.json"
rc=$?
echo "== profile exit code: $rc =="
exit $rc
