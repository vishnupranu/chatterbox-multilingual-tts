#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "========================================================"
echo "    CHATTERBOX MULTILINGUAL TTS - PLATFORM AUDIT SUITE  "
echo "========================================================"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "[ERROR] Python 3 not found!"
    exit 1
fi

echo "[*] Python executable: $($PYTHON --version)"
echo "[*] Target device: $($PYTHON -c "import tts_engine; print(tts_engine.get_target_device())")"
echo "[*] Running comprehensive audit across 20 subsystems..."
echo ""

$PYTHON tests/test_audit.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "  [AUDIT PASSED] ALL 20 SUBSYSTEM TESTS VERIFIED 100%   "
    echo "========================================================"
else
    echo ""
    echo "========================================================"
    echo "  [AUDIT FAILED] SUBSYSTEM AUDIT TESTS ENCOUNTERED ERRORS"
    echo "========================================================"
    exit $EXIT_CODE
fi
