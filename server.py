from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from inference import InferenceService

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("multitalker")
API_KEY = os.getenv("MULTITALKER_API_KEY", "")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_BUFFER = int(os.getenv("MAX_STREAM_BUFFER_MB", "16")) * 1024 * 1024
DEVICE = os.getenv("MODEL_DEVICE", "cuda")
service = InferenceService(DEVICE)
gpu_lock = asyncio.Semaphore(1)


class TranscriptSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptResponse(BaseModel):
    segments: list[TranscriptSegment]
    duration: float
    processing_time: float
    rtf: float | None


class HealthResponse(BaseModel):
    status: str
    gpu: bool
    models_loaded: bool
    device: str
    gpu_name: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(service.load)
    yield


app = FastAPI(title="Multitalker inference service", lifespan=lifespan)


def authorized(value: str | None) -> bool:
    configured_key = os.getenv("MULTITALKER_API_KEY", API_KEY)
    return bool(configured_key) and value == f"Bearer {configured_key}"


@app.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    gpu_name = None
    try:
        import torch
        gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu else None
    except ImportError:
        gpu = False
    if not service.ready:
        response.status_code = 503
    return HealthResponse(status="ok" if service.ready else "starting", gpu=gpu,
                          models_loaded=service.ready, device=DEVICE, gpu_name=gpu_name)


@app.get("/ready", response_model=HealthResponse)
async def ready() -> HealthResponse:
    result = await health(Response())
    if not service.ready:
        raise HTTPException(status_code=503, detail="Models are not initialized")
    return result


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile = File(...), authorization: str | None = Header(default=None)) -> TranscriptResponse:
    if not authorized(authorization):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not service.ready:
        raise HTTPException(status_code=503, detail="Models are not initialized")
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in {".mp4", ".m4a", ".wav", ".webm", ".mp3", ".ogg"}:
        raise HTTPException(status_code=422, detail="Unsupported audio format")
    request_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="multitalker-") as directory:
        source = Path(directory) / f"source{suffix}"
        target = Path(directory) / "audio.wav"
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Audio file too large")
        source.write_bytes(data)
        try:
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", str(source), "-ac", "1", "-ar", "16000", "-f", "wav", str(target)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            async with gpu_lock:
                result = await asyncio.to_thread(service.transcribe_file, target)
        except subprocess.CalledProcessError as error:
            LOGGER.warning("request=%s audio conversion failed", request_id)
            raise HTTPException(status_code=422, detail="Audio conversion failed") from error
        except Exception as error:
            LOGGER.exception("request=%s inference failed", request_id)
            raise HTTPException(status_code=500, detail="Inference failed") from error
    return TranscriptResponse(segments=result.segments, duration=result.duration,
                              processing_time=result.processing_time, rtf=result.rtf)


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = None
    buffered = 0
    try:
        start = await websocket.receive_json()
        if not authorized(start.get("authorization") or websocket.headers.get("authorization")):
            await websocket.send_json({"type": "error", "code": "unauthorized", "message": "Invalid API key"})
            await websocket.close(code=1008)
            return
        if start.get("type") != "start" or any((start.get("sample_rate") != 16000,
                start.get("channels") != 1, start.get("encoding") != "pcm_s16le")):
            await websocket.send_json({"type": "error", "code": "invalid_start", "message": "Expected 16 kHz mono pcm_s16le"})
            await websocket.close(code=1003)
            return
        session = service.new_stream()
        await websocket.send_json({"type": "ready", "session_id": session_id})
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                buffered += len(chunk)
                if buffered > MAX_BUFFER:
                    await websocket.send_json({"type": "error", "code": "backpressure", "message": "Stream buffer limit exceeded"})
                    await websocket.close(code=1009)
                    return
                async with gpu_lock:
                    segments = await asyncio.to_thread(session.push, chunk)
                buffered = 0
                await websocket.send_json({"type": "transcript", "segments": segments, "final": False})
            elif message.get("text") and json.loads(message["text"]).get("type") == "stop":
                async with gpu_lock:
                    segments = await asyncio.to_thread(session.finish)
                await websocket.send_json({"type": "transcript", "segments": segments, "final": True})
                return
    except WebSocketDisconnect:
        LOGGER.info("session=%s disconnected", session_id)
    except Exception:
        LOGGER.exception("session=%s streaming failed", session_id)
        try:
            await websocket.send_json({"type": "error", "code": "inference_failed", "message": "Streaming inference failed"})
        except Exception:
            pass