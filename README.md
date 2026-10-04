# Multitalker GPU inference service

A long-running FastAPI service for a RunPod NVIDIA Pod. It loads NVIDIA Multitalker Parakeet and Streaming Sortformer once, and exposes recorded and live transcription. Speaker IDs describe arrival order; they do not identify clinician/patient roles.

## Run

Use Python 3.11, FFmpeg, and a CUDA-compatible NVIDIA driver. Install matching PyTorch/Torchaudio CUDA wheels before the remaining dependencies:

```bash
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
cp .env.example .env
# Set MULTITALKER_API_KEY to a non-placeholder secret.
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 --ws-max-size 65536 --ws-max-queue 16
```

The service loads its adjacent `.env` without overriding deployment environment variables. Models load during application startup, before requests are served, and may download several GB on the first run. Startup fails if the API key is missing/placeholder or CUDA is requested but unavailable. Once running, `/health` and `/ready` return 503 if models are unavailable.

For Docker:

```bash
docker build -t multitalker .
docker run --gpus all --env-file .env -p 8000:8000 multitalker
```

The image uses Python 3.11 and CUDA 12.6 PyTorch wheels, which include their CUDA runtime libraries. The host provides the NVIDIA driver/container runtime. Use one Uvicorn worker per GPU; additional workers each load separate models and bypass the process-local inference lock/session limit.

## Recorded transcription

```bash
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/transcribe -H "Authorization: Bearer YOUR_KEY" -F "file=@test.wav"
```

Supported extensions: WAV, WebM, MP3, MP4, M4A, OGG. FFmpeg converts recorded files to mono 16 kHz WAV asynchronously. Upload size, decoded duration, and conversion time are bounded. The response retains the recorded API's format:

```json
{
  "segments": [{"speaker": "speaker_0", "start": 0.1, "end": 1.7, "text": "Knee pain."}],
  "duration": 2.5,
  "processing_time": 0.6,
  "rtf": 0.24
}
```

Times are seconds. Duration includes silence and comes from the decoded audio, not the last spoken word. RTF covers model processing only; conversion/queue/network latency are additional.

## WebSocket protocol

NestJS connects to `wss://<POD_ID>-8000.proxy.runpod.net/stream` with `Authorization: Bearer <MULTITALKER_API_KEY>`, then sends:

```json
{"type":"start","sample_rate":16000,"channels":1,"encoding":"pcm_s16le"}
```

The first message must arrive within `STREAM_START_TIMEOUT_SECONDS`. For compatibility, the key can also be supplied in the start message's `authorization` field. Wait for:

```json
{"type":"ready","session_id":"SERVER_SESSION_ID"}
```

Then send **raw binary mono PCM16 little-endian at 16,000 Hz**, with complete two-byte samples. Browser/NestJS 80 ms frames contain 2,560 bytes. There is no live resampling, MediaRecorder, or Base64 conversion in this service.

PCM packet boundaries do not determine model steps. A bounded PCM window retains STFT context and generates each feature frame once. NeMo's `CacheAwareStreamingAudioBuffer` waits for full model-sized chunks, preserves pre-encoder context, and normalizes those feature views. Each connection has its own `SpeakerTaggedASR` cache. Partial model chunks are flushed only on stop; minimal model padding is excluded from output duration. Consumed feature history is trimmed.

Transcript messages contain a **complete current snapshot**, sent when text/timing changes:

```json
{
  "type": "transcript",
  "session_id": "SERVER_SESSION_ID",
  "snapshot": true,
  "final": false,
  "segments": [{
    "id": "SERVER_SESSION_ID:speaker_0:100:0",
    "speaker": "speaker_0",
    "text": "Knee pain",
    "start": 0.1,
    "end": 1.7,
    "startMs": 100,
    "endMs": 1700,
    "isFinal": false
  }]
}
```

IDs stay stable when an utterance's text/end time changes; start time and speaker determine identity. `start`/`end` retain the previous API's second-based fields. `startMs`/`endMs`/`id`/`isFinal` match PhysioAssistant's transcript segment fields. Treat each snapshot as the current transcript, replacing previous snapshots; do not append the entire list repeatedly. Speaker assignment/boundaries can be revised by the model before stop.

Send `{"type":"stop"}` after the last binary frame. The service drains accepted PCM in order, emits a final transcript with `final: true` and `isFinal: true`, then `{"type":"ended","session_id":"..."}`, and closes with code 1000. An empty stream produces an empty final snapshot.

Errors are `{"type":"error","code":"...","message":"..."}` followed by closure: invalid controls/PCM 1003, authentication/start timeout 1008, buffer/duration limit 1009, unavailable/busy 1013, inference failure 1011. Disconnect/error/stop releases session caches and capacity. Cancelled requests wait for their underlying worker thread before releasing the GPU lock or deleting files.

## Capacity and configuration

- `MAX_STREAM_BUFFER_BYTES=960000`: 30 seconds of pending PCM, including the packet executing inference. This is queue capacity, not a 30-second recording limit. Pending packets are also capped at 512 to bound queue metadata.
- `MAX_STREAM_CHUNK_BYTES=65536`: maximum individual PCM packet. Keep Uvicorn's `--ws-max-size` at least this large; `--ws-max-queue 16` also bounds its transport queue.
- `MAX_ACTIVE_STREAMS=1`: conservative default for session cache memory. Additional sessions receive `busy`; increase only after measuring GPU memory and inference throughput.
- `MAX_SPEAKERS=2`: configurable from one to four.
- `MAX_AUDIO_DURATION_SECONDS=3600`: maximum recorded/live audio duration, separate from queued buffer capacity.
- `MAX_UPLOAD_MB=100`, `CONVERSION_TIMEOUT_SECONDS=60`, `STREAM_START_TIMEOUT_SECONDS=10`.

GPU work is serialized across recorded and live requests. Long recorded jobs can delay streaming; use separate service instances for recorded/live traffic if that latency is unacceptable.

## PhysioAssistant integration boundary

This is a Pod HTTP/WebSocket service, not RunPod's serverless `/run` + `/status` job API. PhysioAssistant's current `RunpodRecordedProvider` calls the latter, and its live `createStream()` remains a stub. It therefore needs a Pod provider adapter before connecting to this service. Recorded requests must use multipart field `file`; live starts must translate `sampleRate` to `sample_rate`, and upstream `ready` to the frontend's `started` acknowledgement.

The NestJS gateway should forward transcript snapshots to the frontend and reconcile them by segment ID before persistence. It must also handle upstream errors/disconnects and wait for `ended` on stop. Disable PhysioAssistant's `AUDIO_STREAM_DISCARD` only once that adapter is connected. Keep the service API key in NestJS, never the browser.

## Verification

Install `pytest`, `httpx`, and `numpy` for local tests, then run `python -m pytest -q`. API tests use fake models; inference tests check packet-independent feature windows, model chunk gating, context retention, final flushing, SegLST normalization, and cancellation/cleanup without loading model weights.

NeMo is pinned to commit `1688cc3d6a9ade854f544987810c53f605dc86fc` rather than a moving `main`. The adapter follows NVIDIA's [cache-aware buffer](https://github.com/NVIDIA/NeMo/blob/1688cc3d6a9ade854f544987810c53f605dc86fc/nemo/collections/asr/parts/utils/streaming_utils.py) and [multitalker helpers](https://github.com/NVIDIA/NeMo/blob/1688cc3d6a9ade854f544987810c53f605dc86fc/nemo/collections/asr/parts/utils/multispk_transcribe_utils.py).

These checks do not establish GPU readiness. Before deployment, build the image on RunPod, load the actual pinned models, stream known single/overlapping-speaker clips with varying packet sizes, compare complete transcripts and timestamps, measure real-time factor/memory across a long session, and verify stop/disconnect behavior. No local CUDA inference or container build is claimed.
