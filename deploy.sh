#!/usr/bin/env bash
# ==============================================================================
# deploy.sh
# Production Deployment & Live Service Orchestrator for Chatterbox Multilingual TTS
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "    CHATTERBOX MULTILINGUAL TTS - PRODUCTION DEPLOYMENT   "
echo "=========================================================="

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

# 1. Environment Verification
echo "[1/4] Auditing virtual environment..."
if [ ! -d ".venv" ]; then
    echo "[!] .venv missing. Running initial environment bootstrap with uv..."
    uv venv --python 3.11
fi

PYTHON=".venv/bin/python"
UVICORN=".venv/bin/uvicorn"

if [ ! -f "$UVICORN" ]; then
    echo "[!] uvicorn missing. Installing production server stack..."
    uv pip install uvicorn fastapi soundfile
fi

# 2. Run Subsystem Health and Verification Audit
echo "[2/4] Executing pre-flight audit suite..."
$PYTHON tests/test_audit.py
if [ $? -ne 0 ]; then
    echo "[FATAL] Pre-flight audit failed! Aborting deployment."
    exit 1
fi
echo "[+] Pre-flight audit passed 100%."

# 3. Create necessary runtime directories
echo "[3/4] Ensuring static cache and upload directories..."
mkdir -p static/audio_output
mkdir -p static/uploads
mkdir -p logs

# 4. Launch Production Service
echo "[4/4] Launching Chatterbox Live Service on http://${HOST}:${PORT}..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PORT="${PORT}"

exec $UVICORN server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level info \
    --access-log
