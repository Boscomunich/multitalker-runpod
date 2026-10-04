FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CUDA runtime libraries are supplied by the PyTorch wheels. Run the image
# with the NVIDIA container runtime; no system Python/driver mix is needed.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers 1 --ws-max-size 65536 --ws-max-queue 16"]
