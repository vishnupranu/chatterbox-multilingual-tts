"""
desktop_app.py
Native Desktop Application for Chatterbox Multilingual Voice Platform.
Provides cross-platform native macOS / Windows / Linux window wrapper with WebKit rendering,
integrated AITalk copilot, and background server orchestration.
"""
import os
import sys
import time
import socket
import threading
import urllib.request
import urllib.error

# Enforce MPS fallback for macOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def is_server_running(host: str, port: int) -> bool:
    """Checks if server is already responding to HTTP requests."""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/status")
        with urllib.request.urlopen(req, timeout=1.5) as res:
            return res.status == 200
    except Exception:
        return False


def start_background_server():
    """Starts the FastAPI backend in a background daemon thread if not already running."""
    import uvicorn
    from server import app
    print(f"[*] Starting local backend daemon on {BASE_URL}...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


def main():
    print("=========================================================")
    print("   CHATTERBOX MULTILINGUAL TTS - DESKTOP APPLICATION     ")
    print("=========================================================")

    # 1. Start or connect to backend server
    if not is_server_running(SERVER_HOST, SERVER_PORT):
        server_thread = threading.Thread(target=start_background_server, daemon=True)
        server_thread.start()

        # Wait up to 10 seconds for startup
        retries = 20
        while retries > 0 and not is_server_running(SERVER_HOST, SERVER_PORT):
            time.sleep(0.5)
            retries -= 1

        if not is_server_running(SERVER_HOST, SERVER_PORT):
            print("[ERROR] Failed to start backend server. Aborting.")
            sys.exit(1)
        print(f"[+] Backend server operational on {BASE_URL}")
    else:
        print(f"[+] Connected to running backend on {BASE_URL}")

    # 2. Launch Native Desktop Window
    try:
        import webview
        print("[*] Launching Native Desktop Window via WebKit...")
        window = webview.create_window(
            title="Chatterbox Multilingual TTS Studio",
            url=BASE_URL,
            width=1320,
            height=880,
            min_size=(1024, 700),
            background_color="#090a0f",
            text_select=True,
            confirm_close=False,
        )
        webview.start(debug=False)
    except ImportError:
        print("[!] pywebview not found. Falling back to default desktop web browser...")
        import webbrowser
        webbrowser.open(BASE_URL)
    except Exception as e:
        print(f"[!] Desktop window error ({e}). Opening system browser...")
        import webbrowser
        webbrowser.open(BASE_URL)


if __name__ == "__main__":
    main()
