# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml README.md ./
COPY resemble_chatterbox ./resemble_chatterbox

RUN uv pip install --system --only-binary llvmlite -e resemble_chatterbox
RUN uv pip install --system fastapi uvicorn soundfile librosa gradio requests

COPY . .

EXPOSE 8080

CMD ["python", "server.py"]
