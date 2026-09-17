#!/usr/bin/env bash
# ==============================================================================
# launch_desktop.sh
# Native Desktop Application Launcher for Chatterbox Multilingual TTS
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "[*] Launching Chatterbox Desktop Application..."
exec $PYTHON desktop_app.py "$@"
