from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import tempfile
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from anyio import CancelScope

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from inference import InferenceService

load_dotenv(Path(__file__).with_name(".env"), override=False)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("multitalker")
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_BUFFER = int(os.getenv("MAX_STREAM_BUFFER_BYTES", "960000"))
MAX_CHUNK = int(os.getenv("MAX_STREAM_CHUNK_BYTES", "65536"))
MAX_QUEUED_PACKETS = 512
MAX_ACTIVE_STREAMS = int(os.getenv("MAX_ACTIVE_STREAMS", "1"))
MAX_DURATION = float(os.getenv("MAX_AUDIO_DURATION_SECONDS", "3600"))
START_TIMEOUT = float(os.getenv("STREAM_START_TIMEOUT_SECONDS", "10"))
CONVERSION_TIMEOUT = float(os.getenv("CONVERSION_TIMEOUT_SECONDS", "60"))
DEVICE = os.getenv("MODEL_DEVICE", "cuda")
service = InferenceService(DEVICE, max_speakers=int(os.getenv("MAX_SPEAKERS", "2")))
gpu_lock = asyncio.Semaphore(1)
active_streams = 0


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
    key = os.getenv("MULTITALKER_API_KEY", "")
    if not key or key == "replace-me":
        raise RuntimeError("Set a non-placeholder MULTITALKER_API_KEY before starting")
    limits = (MAX_UPLOAD, MAX_BUFFER, MAX_CHUNK, MAX_ACTIVE_STREAMS,
              MAX_DURATION, START_TIMEOUT, CONVERSION_TIMEOUT)
    if any(not math.isfinite(value) or value <= 0 for value in limits) or MAX_CHUNK > MAX_BUFFER:
        raise RuntimeError("Audio limits must be positive; chunk limit must fit the buffer")
    await asyncio.to_thread(service.load)
    yield


app = FastAPI(title="Multitalker inference service", lifespan=lifespan)


def authorized(value: str | None) -> bool:
    key = os.getenv("MULTITALKER_API_KEY", "")
    return (bool(key) and key != "replace-me" and isinstance(value, str) and
            hmac.compare_digest(value.encode(), f"Bearer {key}".encode()))


async def run_thread(function: Callable[..., Any], *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling to_thread does not stop its underlying thread. Keep the
        # semaphore and caller-owned files/session alive until it finishes.
        with CancelScope(shield=True):
            # Repeated task cancellation must not cancel the shielded worker
            # future and release the semaphore while its thread still runs.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled() and task.exception() is not None:
                LOGGER.error("Worker failed during cancellation",
                             exc_info=task.exception())
        raise


async def run_gpu(function: Callable[..., Any], *args: Any) -> Any:
    async with gpu_lock:
        return await run_thread(function, *args)


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
async def ready(response: Response) -> HealthResponse:
    return await health(response)


async def convert_audio(source: Path, target: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-t", str(MAX_DURATION + 1), "-f", "wav", str(target),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        await asyncio.wait_for(process.communicate(), CONVERSION_TIMEOUT)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise HTTPException(status_code=422, detail="Audio conversion failed")
    with wave.open(str(target), "rb") as converted:
        duration = converted.getnframes() / converted.getframerate()
    if not duration:
        raise HTTPException(status_code=422, detail="Audio is empty")
    if duration > MAX_DURATION:
        raise HTTPException(status_code=413, detail="Audio duration limit exceeded")


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
        try:
            uploaded = 0
            with source.open("wb") as output:
                while data := await file.read(1024 * 1024):
                    uploaded += len(data)
                    if uploaded > MAX_UPLOAD:
                        raise HTTPException(status_code=413, detail="Audio file too large")
                    await run_thread(output.write, data)
            if not uploaded:
                raise HTTPException(status_code=422, detail="Audio is empty")
            await convert_audio(source, target)
            result = await run_gpu(service.transcribe_file, target)
        except HTTPException:
            raise
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="Audio conversion timed out") from error
        except Exception as error:
            LOGGER.exception("request=%s transcription failed", request_id)
            raise HTTPException(status_code=500, detail="Inference failed") from error
        finally:
            await file.close()
    return TranscriptResponse(segments=result.segments, duration=result.duration,
                              processing_time=result.processing_time, rtf=result.rtf)


class StreamProtocolError(Exception):
    def __init__(self, code: str, message: str, close_code: int = 1003):
        super().__init__(message)
        self.code = code
        self.close_code = close_code


def parse_control(message: dict[str, Any]) -> dict[str, Any]:
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if not isinstance(text, str) or len(text.encode()) > 4096:
        raise StreamProtocolError("invalid_control", "Expected a small JSON control message")
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as error:
        raise StreamProtocolError("invalid_control", "Invalid control JSON") from error
    if not isinstance(value, dict):
        raise StreamProtocolError("invalid_control", "Control message must be an object")
    return value


def wire_segments(segments: list[dict[str, Any]], session_id: str, final: bool) -> list[dict[str, Any]]:
    result = []
    occurrences: dict[str, int] = {}
    for segment in segments:
        start_ms = round(segment["start"] * 1000)
        key = f'{segment["speaker"]}:{start_ms}'
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        result.append({**segment, "id": f"{session_id}:{key}:{occurrence}",
                       "startMs": start_ms, "endMs": round(segment["end"] * 1000),
                       "isFinal": final})
    return result


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    global active_streams
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = None
    created_sessions = []
    tasks: list[asyncio.Task] = []
    admitted = False
    buffered = 0
    total_bytes = 0
    close_code = 1000
    try:
        try:
            start = parse_control(await asyncio.wait_for(websocket.receive(), START_TIMEOUT))
        except TimeoutError as error:
            raise StreamProtocolError("start_timeout", "Start handshake timed out", 1008) from error
        if not authorized(websocket.headers.get("authorization") or start.get("authorization")):
            raise StreamProtocolError("unauthorized", "Invalid API key", 1008)
        if (start.get("type") != "start" or type(start.get("sample_rate")) is not int or
                start.get("sample_rate") != 16000 or type(start.get("channels")) is not int or
                start.get("channels") != 1 or start.get("encoding") != "pcm_s16le"):
            raise StreamProtocolError("invalid_start", "Expected 16 kHz mono pcm_s16le")
        if not service.ready:
            raise StreamProtocolError("unavailable", "Models are not initialized", 1013)
        if active_streams >= MAX_ACTIVE_STREAMS:
            raise StreamProtocolError("busy", "Streaming session capacity reached", 1013)
        active_streams += 1
        admitted = True
        def create_session():
            created_sessions.append(service.new_stream())
            return created_sessions[0]

        session = await run_gpu(create_session)
        await websocket.send_json({"type": "ready", "session_id": session_id})
        # Reserve one entry for the stop sentinel, even at the packet limit.
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=MAX_QUEUED_PACKETS + 1)

        async def receive_audio() -> None:
            nonlocal buffered, total_bytes
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))
                chunk = message.get("bytes")
                if chunk is None:
                    control = parse_control(message)
                    if control.get("type") != "stop":
                        raise StreamProtocolError("invalid_control", "Only stop is supported after start")
                    queue.put_nowait(None)
                    return
                if not chunk or len(chunk) % 2:
                    raise StreamProtocolError("invalid_pcm", "Expected complete PCM16 samples")
                if total_bytes + len(chunk) > MAX_DURATION * 16000 * 2:
                    raise StreamProtocolError("duration_limit", "Audio duration limit exceeded", 1009)
                if (len(chunk) > MAX_CHUNK or buffered + len(chunk) > MAX_BUFFER or
                        queue.qsize() >= MAX_QUEUED_PACKETS):
                    raise StreamProtocolError("backpressure", "Stream buffer limit exceeded", 1009)
                # Includes work currently executing as well as queued packets.
                buffered += len(chunk)
                total_bytes += len(chunk)
                queue.put_nowait(chunk)

        async def consume_audio() -> None:
            nonlocal buffered
            previous = []
            while True:
                chunk = await queue.get()
                if chunk is None:
                    segments = await run_gpu(session.finish)
                    await websocket.send_json({"type": "transcript", "session_id": session_id,
                        "segments": wire_segments(segments, session_id, True),
                        "snapshot": True, "final": True})
                    await websocket.send_json({"type": "ended", "session_id": session_id})
                    return
                try:
                    segments = await run_gpu(session.push, chunk)
                finally:
                    buffered -= len(chunk)
                if segments != previous:
                    previous = segments
                    await websocket.send_json({"type": "transcript", "session_id": session_id,
                        "segments": wire_segments(segments, session_id, False),
                        "snapshot": True, "final": False})

        receiver = asyncio.create_task(receive_audio())
        worker = asyncio.create_task(consume_audio())
        tasks = [receiver, worker]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        # Receiving stop queues a sentinel behind all accepted PCM. Await the
        # worker's final transcript/ended messages before closing the connection.
        await worker
    except WebSocketDisconnect:
        LOGGER.info("session=%s disconnected", session_id)
    except StreamProtocolError as error:
        close_code = error.close_code
        await websocket.send_json({"type": "error", "code": error.code, "message": str(error)})
    except Exception:
        close_code = 1011
        LOGGER.exception("session=%s streaming failed", session_id)
        try:
            await websocket.send_json({"type": "error", "code": "inference_failed",
                                       "message": "Streaming inference failed"})
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        # ASGI shutdown/disconnect may cancel the request's AnyIO scope. Cleanup
        # must survive that cancellation until worker threads release the GPU.
        with CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                if created_sessions:
                    await run_gpu(created_sessions[0].close)
            finally:
                if admitted:
                    active_streams -= 1
            try:
                await websocket.close(code=close_code)
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass
