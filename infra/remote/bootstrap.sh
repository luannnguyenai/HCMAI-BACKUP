#!/usr/bin/env bash
# SPEC-0022 - one-time login-node bootstrap for a fresh GPU lease.
#
# Idempotent: run as many times as you like. Installs `uv` if missing,
# ensures ~/aic2026 exists and is writable. Does NOT write secrets to disk -
# credentials come over ssh at job-launch time only.
set -euo pipefail

AIC2026_BASE="${AIC2026_REMOTE_BASE:-$HOME/aic2026}"

echo "[bootstrap] target base dir: $AIC2026_BASE"
mkdir -p "$AIC2026_BASE"

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv (no sudo)"
  # The official installer drops `uv` into ~/.local/bin; that needs to be on
  # the user's PATH in their .bashrc / .zshrc on the cluster.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bootstrap] uv version: $(uv --version)"

if ! command -v git >/dev/null 2>&1; then
  echo "[bootstrap] WARNING: git not found on PATH. Please ask the cluster admin to provide it."
  exit 1
fi

echo "[bootstrap] hostname:   $(hostname)"
echo "[bootstrap] whoami:     $(whoami)"
echo "[bootstrap] python:     $(command -v python3 || echo 'not on PATH')"
echo "[bootstrap] free disk under base:"
df -h "$AIC2026_BASE" | tail -1

echo "[bootstrap] OK. Next: from your laptop, run \`./bin/remote provision --sha <sha>\`."
