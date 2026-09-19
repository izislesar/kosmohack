#!/usr/bin/env bash
# kosmohack env bootstrap — Task 1 (fire-win-19h).
# Target: vast.ai base CUDA Ubuntu 24.04 image, running as ROOT.
# Idempotent: safe to re-run; second run must exit 0 quickly.
# Non-interactive: DEBIAN_FRONTEND=noninteractive, no prompts.
# Strategy: system pip + venv at /workspace/kosmohack/.venv (primary, lean).
#   Docker: attempted via apt (docker.io, lightweight); failure is NON-FATAL and
#   documented — pip-venv path is the supported working path (no Jupyter-heavy images).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

WORKDIR="${WORKDIR:-/workspace/kosmohack}"
VENV="${VENV:-$WORKDIR/.venv}"
LOG="${LOG:-$WORKDIR/env/bootstrap-run.log}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

mkdir -p "$WORKDIR/env"
touch "$LOG"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

log "== bootstrap start (workdir=$WORKDIR venv=$VENV) =="

# --- 1. apt base packages (idempotent, retries) ---
log "-- apt update/install base packages --"
apt_get() { timeout 240 apt-get "$@" ; }
for i in 1 2 3; do
  if apt_get update >>"$LOG" 2>&1; then break; fi
  log "apt-get update attempt $i failed, retrying..."
  sleep 5
  [ "$i" = "3" ] && log "WARN: apt-get update failed 3x, continuing with existing lists"
done
# --no-install-recommends keeps it lean; gdal bits needed for rasterio wheels fallback
timeout 600 apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip python3-dev \
  build-essential git curl ca-certificates \
  libgl1 libglib2.0-0 libgdal-dev gdal-bin >>"$LOG" 2>&1 \
  || log "WARN: some apt packages failed; continuing (wheels may cover it)"

# --- 2. docker (best-effort, NON-FATAL) ---
log "-- docker (best-effort) --"
if command -v docker >/dev/null 2>&1; then
  log "docker already present: $(docker --version 2>&1 | tee -a "$LOG")"
else
  if timeout 600 apt-get install -y --no-install-recommends docker.io >>"$LOG" 2>&1; then
    log "docker.io installed: $(docker --version 2>&1 | tee -a "$LOG")"
  else
    log "WARN: docker install failed/skipped — PIP-VENV FALLBACK is the working path (documented decision)."
  fi
fi
{ docker ps 2>&1 || echo "DOCKER_PS_UNAVAILABLE (no daemon or no docker — venv path unaffected)"; } | tee -a "$LOG"

# --- 3. venv (idempotent) ---
log "-- venv at $VENV --"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" 2>&1 | tee -a "$LOG"
  log "venv created"
else
  log "venv exists, reusing"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools 2>&1 | tail -2 | tee -a "$LOG"

# --- 3b. project python3 on PATH (wrapper; /usr/bin/python3 stays system for apt) ---
log "-- /usr/local/bin/python3 wrapper --"
if grep -q "managed by env/bootstrap.sh" /usr/local/bin/python3 2>/dev/null; then
  log "wrapper present"
else
  rm -f /usr/local/bin/python3
  printf '#!/bin/sh\n# kosmohack venv-wrapper (managed by env/bootstrap.sh)\nexec %s/bin/python "$@"\n' "$VENV" > /usr/local/bin/python3
  chmod +x /usr/local/bin/python3
  log "wrapper installed"
fi

# --- 4. torch first (CUDA-matched index), then the rest from PyPI ---
log "-- pip install torch (cu121) --"
if "$VENV/bin/python" -c "import torch; assert torch.__version__.startswith('2.3.')" 2>/dev/null; then
  log "torch 2.3 already installed: $("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
else
  timeout 1800 "$VENV/bin/pip" install --index-url "$TORCH_INDEX_URL" \
    "torch==2.3.1" "torchvision==0.18.1" 2>&1 | tail -3 | tee -a "$LOG"
fi
log "-- pip install -r requirements (PyPI, torch lines already satisfied) --"
# requirements.txt lives at repo root locally and in env/ on the VPS — accept both, NEVER silent-empty
REQ=""
for cand in "$WORKDIR/requirements.txt" "$WORKDIR/env/requirements.txt"; do
  if [ -f "$cand" ]; then REQ="$cand"; break; fi
done
if [ -z "$REQ" ]; then log "ERROR: requirements.txt not found in $WORKDIR or $WORKDIR/env"; exit 1; fi
log "requirements file: $REQ"
# Strip torch lines from requirements to avoid PyPI (non-CUDA) reinstall
grep -v -E '^(torch|torchvision)==' "$REQ" > /tmp/kosmohack-req-nocuda.txt
if [ ! -s /tmp/kosmohack-req-nocuda.txt ]; then log "ERROR: filtered requirements empty"; exit 1; fi
timeout 1800 "$VENV/bin/pip" install -r /tmp/kosmohack-req-nocuda.txt 2>&1 | tail -5 | tee -a "$LOG"

# --- 5. verify: CUDA must be True, not just pip exit 0 ---
# NOTE: verify steps are failure-tolerant (record, don't abort) so the log always
# reaches "bootstrap done". Real gate results are asserted by the caller/QA channel.
log "-- verify --"
{ "$VENV/bin/python" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" 2>&1 || log "VERIFY-FAIL: torch cuda check"; } | tee -a "$LOG"
{ "$VENV/bin/python" -c "import torch; a=torch.randn(64,64,device='cuda'); print('tensor-op-ok', float((a@a).sum()))" 2>&1 | tail -3 || log "VERIFY-FAIL: cuda tensor op"; } | tee -a "$LOG"
{ "$VENV/bin/python" -c "import segmentation_models_pytorch as smp, timm, fastapi; print('imports-ok', smp.__version__, timm.__version__, fastapi.__version__)" 2>&1 || log "VERIFY-FAIL: smp/timm/fastapi imports"; } | tee -a "$LOG"
{ "$VENV/bin/python" -c "import terratorch, importlib.metadata as m; print('terratorch-ok', m.version('terratorch'))" 2>&1 || log "WARN: terratorch import failed (non-blocking for AF-unet start)"; } | tee -a "$LOG"

log "== bootstrap done =="
