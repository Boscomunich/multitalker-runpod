# Multitalker GPU inference service

This is a long-running FastAPI service for a RunPod NVIDIA Pod. It loads the NVIDIA Multitalker Parakeet and Streaming Sortformer models once during startup and exposes recorded and incremental streaming transcription.

## Run

Use Python 3.11, a CUDA-compatible NVIDIA driver, FFmpeg, and a NeMo-compatible PyTorch installation. Install `requirements.txt`, copy `.env.example` to `.env`, set `MULTITALKER_API_KEY`, then run:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The first startup downloads several GB of model weights. `/health` reports `starting` until both models load; `/ready` returns 503 during that period.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/transcribe -H "Authorization: Bearer YOUR_KEY" -F "file=@test.wav"
```

## WebSocket protocol

Connect to `/stream`, send an initial JSON message containing `authorization`, `type: start`, `sample_rate: 16000`, `channels: 1`, and `encoding: pcm_s16le`, then send binary signed-16-bit PCM frames. Send `{"type":"stop"}` to flush the final result. The service uses NeMo's `CacheAwareStreamingAudioBuffer` and `SpeakerTaggedASR` per connection; it does not batch the whole socket into one request. A single semaphore serializes GPU work, and the configured buffer limit closes an overflowing connection.

## RunPod and NestJS

Expose TCP port 8000. The proxy URL is `https://<POD_ID>-8000.proxy.runpod.net/transcribe`. NestJS should send its `Buffer` as multipart field `file` with `Authorization: Bearer <MULTITALKER_API_KEY>`. For streaming, its gateway should proxy the documented WebSocket handshake and binary PCM frames to `wss://<POD_ID>-8000.proxy.runpod.net/stream`.

Speaker labels are arrival-order IDs such as `speaker_0` and `speaker_1`; the service never identifies clinician or patient from voice characteristics.

## Verification and limitations

The lightweight tests cover readiness and authentication without downloading models. Actual NeMo imports, model weights, CUDA memory use, FFmpeg conversion, timestamps, and real-time factor require a GPU integration test. NeMo `main` is used because these newly released model cards depend on current helper APIs; pin a dated NeMo commit after validating the target RunPod CUDA image.
