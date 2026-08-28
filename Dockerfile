FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python3.11 -m uvicorn server:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]