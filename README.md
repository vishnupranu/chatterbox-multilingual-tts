# Chatterbox Z.ai Multilingual Platform

An advanced, flexible Text-to-Speech (TTS) platform inspired by **[chat.z.ai](https://chat.z.ai/)**, powered by **Resemble AI's Chatterbox Multilingual V3/V2** models.

Features a sleek dark glassmorphism UI with **three interactive workspaces**:
1. 💬 **Chat**: Conversational AI voice assistant with 23-language speech synthesis, zero-shot voice cloning, audio waveforms, and prompt suggestions.
2. 💻 **Code**: Interactive developer workspace with syntax-highlighted code editor, multi-language snippets (Python SDK, cURL, Node.js, CLI), and a live script execution runner.
3. ⚡ **Workers**: Autonomous background agents for batch TTS generation, cross-lingual localization/dubbing, voice timbre profiling, and real-time job queue tracking.

---

## Workspace Tour

### 1. 💬 Chat Workspace
- **Instant Multilingual Voice**: Synthesize natural speech across 23 languages (English, Spanish, French, German, Chinese, Japanese, Hindi, Arabic, Russian, Portuguese, etc.).
- **Zero-Shot Voice Cloning**: Upload or record a reference audio clip (10 seconds) to clone vocal timbre, prosody, and style across all languages.
- **Waveform Audio Player**: Integrated audio controls with wave visualizer, playback rate controls, and one-click WAV download.
- **Accent Decoupling**: Set CFG/Pace Weight to `0.0` for cross-lingual accent transfer without reference accent leakage.

### 2. 💻 Code Workspace
- **Pre-built Integration Snippets**:
  - Python SDK (`from chatterbox.mtl_tts import ChatterboxMultilingualTTS`)
  - cURL REST API (`POST /api/chat`)
  - Modern JavaScript Fetch client
  - CLI usage (`./run.sh cli`)
- **Live Script Runner**: Test generation parameters and run requests directly against the local engine with console output.

### 3. ⚡ Autonomous Workers Workspace
- **Batch Audio Generator**: Input multi-line documents, articles, or scripts to generate sequential audio files with a progress bar and batch downloads.
- **Cross-Lingual Dubbing Worker**: Automatically renders synchronized audio in multiple selected target languages preserving speaker vocal tone.
- **Voice Profiler**: Upload voice clips to estimate fundamental frequency ($f_0$), pitch, duration, and optimal synthesis parameters.
- **Live Queue Manager**: Real-time status badges (`QUEUED`, `RUNNING`, `COMPLETED`), progress bars, and streaming worker logs.

---

## Quick Start

### Launch the Platform
```bash
./run.sh
```
Open your browser at **`http://localhost:8000`**.

### Alternative Modes
- **Gradio Classic UI**:
  ```bash
  ./run.sh ui
  # Runs on http://localhost:7860
  ```
- **CLI Terminal Synthesis**:
  ```bash
  ./run.sh cli --text "Hello world from terminal" --lang en --output speech.wav
  ```
- **List All 23 Supported Languages**:
  ```bash
  ./run.sh languages
  ```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Primary Z.ai Single Page Application |
| `GET` | `/api/status` | Engine status, hardware device (`mps`/`cpu`), model version |
| `GET` | `/api/languages` | Supported languages and sample prompt texts |
| `POST` | `/api/chat` | Main conversational TTS synthesis endpoint |
| `POST` | `/api/upload-voice` | Upload audio file for voice cloning |
| `POST` | `/api/code/run` | Execute code test request and return logs & audio |
| `POST` | `/api/workers/batch` | Enqueue batch audio generation job |
| `POST` | `/api/workers/dubbing` | Enqueue cross-lingual dubbing job |
| `GET` | `/api/workers/jobs` | Get list of all worker jobs and statuses |
| `GET` | `/api/workers/jobs/{id}`| Get details and logs of a specific worker job |

---

## Supported Languages (23 Total)

`ar` (Arabic) • `da` (Danish) • `de` (German) • `el` (Greek) • `en` (English) • `es` (Spanish) • `fi` (Finnish) • `fr` (French) • `he` (Hebrew) • `hi` (Hindi) • `it` (Italian) • `ja` (Japanese) • `ko` (Korean) • `ms` (Malay) • `nl` (Dutch) • `no` (Norwegian) • `pl` (Polish) • `pt` (Portuguese) • `ru` (Russian) • `sv` (Swedish) • `sw` (Swahili) • `tr` (Turkish) • `zh` (Chinese)

---

## Hardware Optimization (Apple Silicon & Intel Mac)

- Automatically detects Apple Silicon Metal Performance Shaders (`mps`) or NVIDIA CUDA.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is pre-configured to handle FFT and complex tensor operations safely.
- Remaps weights via `mac_patch.py` to prevent CUDA serialization errors on macOS.
