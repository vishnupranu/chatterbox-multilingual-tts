#!/usr/bin/env bash
set -e

export PYTORCH_ENABLE_MPS_FALLBACK=1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "[!] 'uv' is not installed. Please install uv (e.g. brew install uv or curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi

PYTHON=".venv/bin/python"

COMMAND="${1:-app}"

case "$COMMAND" in
    app|"")
        echo "[*] Launching Chatterbox Z.ai Multilingual Platform on http://localhost:8000..."
        "$PYTHON" server.py
        ;;
    ui|gradio)
        echo "[*] Launching Gradio UI on http://localhost:7860..."
        "$PYTHON" app.py
        ;;
    cli)
        shift
        "$PYTHON" cli.py "$@"
        ;;
    languages)
        "$PYTHON" cli.py --list-languages
        ;;
    *)
        echo "Usage: $0 {app|ui|cli|languages} [arguments...]"
        echo "Examples:"
        echo "  $0                                             # Launch Z.ai Platform (http://localhost:8000)"
        echo "  $0 app                                         # Launch Z.ai Platform (Chat, Code, Workers)"
        echo "  $0 ui                                          # Launch Gradio Web UI (http://localhost:7860)"
        echo "  $0 cli -t 'Hello world' -l en -o speech.wav   # Synthesize English via CLI"
        echo "  $0 languages                                   # List all 23 supported languages"
        exit 1
        ;;
esac
