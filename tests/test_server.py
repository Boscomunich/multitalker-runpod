import asyncio
import threading
from contextlib import suppress
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from inference import InferenceResult

START = {"type": "start", "sample_rate": 16000, "channels": 1,
         "encoding": "pcm_s16le"}
AUTH = {"authorization": "Bearer test"}


class FakeSession:
    def __init__(self):
        self.chunks = []
        self.closed = False
        self.finished = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def push(self, chunk):
        self.entered.set()
        if self.block:
            assert self.release.wait(5), "Test worker was not released"
        if self.fail:
            raise RuntimeError("GPU failed")
        self.chunks.append(chunk)
        return [{"speaker": "speaker_0", "start": 0.0, "end": 0.08,
                 "text": "Knee" if len(self.chunks) == 1 else "Knee pain"}]

    def finish(self):
        self.finished = True
        return [{"speaker": "speaker_0", "start": 0.0, "end": 0.16,
                 "text": "Knee pain."}] if self.chunks else []

    def close(self):
        self.closed = True


class FakeService:
    ready = True

    def __init__(self):
        self.sessions = []
        self.next_session = None

    def load(self):
        pass

    def new_stream(self):
        session = self.next_session or FakeSession()
        self.next_session = None
        self.sessions.append(session)
        return session

    def transcribe_file(self, path):
        assert Path(path).exists()
        return InferenceResult([{"speaker": "speaker_0", "start": 0,
                                 "end": 1, "text": "Hello"}], 2.0, 0.5)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("MULTITALKER_API_KEY", "test")
    fake = FakeService()
    monkeypatch.setattr(server, "service", fake)
    monkeypatch.setattr(server, "gpu_lock", asyncio.Semaphore(1))
    monkeypatch.setattr(server, "active_streams", 0)
    with TestClient(server.app) as client:
        yield client, fake
    assert server.active_streams == 0
    assert all(session.closed for session in fake.sessions)


def begin(socket):
    socket.send_json(START)
    return socket.receive_json()


def test_health_and_ready_without_model(api):
    client, fake = api
    fake.ready = False
    for path in ("/health", "/ready"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["models_loaded"] is False


def test_transcribe_requires_auth(api):
    client, _ = api
    response = client.post("/transcribe", files={"file": ("test.wav", b"RIFF")})
    assert response.status_code == 401


@pytest.mark.parametrize("data,filename,status", [
    (b"", "empty.wav", 422), (b"abc", "test.exe", 422),
])
def test_invalid_upload(api, data, filename, status):
    client, _ = api
    response = client.post("/transcribe", headers=AUTH, files={"file": (filename, data)})
    assert response.status_code == status


def test_upload_limit(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(server, "MAX_UPLOAD", 3)
    response = client.post("/transcribe", headers=AUTH, files={"file": ("test.wav", b"four")})
    assert response.status_code == 413


def test_recorded_transcript_and_temporary_file_cleanup(api, monkeypatch):
    client, _ = api
    paths = []

    async def convert(source, target):
        paths.extend([source, target])
        target.write_bytes(b"fake")

    monkeypatch.setattr(server, "convert_audio", convert)
    response = client.post("/transcribe", headers=AUTH, files={"file": ("test.wav", b"RIFF")})
    assert response.status_code == 200
    assert response.json()["duration"] == 2.0
    assert response.json()["rtf"] == 0.25
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("control", [[], None, "string", {"type": "start"}])
def test_invalid_start_is_protocol_error(api, control):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        socket.send_json(control)
        assert socket.receive_json()["code"] in {"invalid_control", "invalid_start"}
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1003


def test_bad_json_is_protocol_error(api):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        socket.send_text("{")
        assert socket.receive_json()["code"] == "invalid_control"


def test_stream_requires_auth(api):
    client, _ = api
    with client.websocket_connect("/stream") as socket:
        socket.send_json(START)
        assert socket.receive_json()["code"] == "unauthorized"


def test_stream_checks_readiness(api):
    client, fake = api
    fake.ready = False
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        socket.send_json(START)
        assert socket.receive_json()["code"] == "unavailable"
    assert not fake.sessions


def test_stream_revisions_stop_and_cleanup(api):
    client, fake = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        ack = begin(socket)
        assert ack["type"] == "ready"
        socket.send_bytes(b"\x01\x00" * 1280)
        interim = socket.receive_json()
        socket.send_bytes(b"\x02\x00" * 1280)
        updated = socket.receive_json()
        assert interim["segments"][0]["id"] == updated["segments"][0]["id"]
        assert updated["segments"][0]["text"] == "Knee pain"
        assert updated["segments"][0]["startMs"] == 0
        assert updated["segments"][0]["endMs"] == 80
        assert updated["segments"][0]["isFinal"] is False
        socket.send_json({"type": "stop"})
        final = socket.receive_json()
        assert final["final"] is True
        assert final["snapshot"] is True
        assert final["segments"][0]["isFinal"] is True
        assert final["segments"][0]["id"] == interim["segments"][0]["id"]
        assert socket.receive_json()["type"] == "ended"
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1000
    assert fake.sessions[0].finished
    assert fake.sessions[0].chunks == [b"\x01\x00" * 1280, b"\x02\x00" * 1280]


def test_empty_stream_finishes_without_inventing_speech(api):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_json({"type": "stop"})
        assert socket.receive_json()["segments"] == []
        assert socket.receive_json()["type"] == "ended"


def test_odd_pcm_is_rejected(api):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_bytes(b"\x00")
        assert socket.receive_json()["code"] == "invalid_pcm"


def test_slow_inference_has_real_queue_limit(api, monkeypatch):
    client, fake = api
    monkeypatch.setattr(server, "MAX_BUFFER", 8)
    monkeypatch.setattr(server, "MAX_CHUNK", 8)
    session = FakeSession()
    session.block = True
    fake.next_session = session
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_bytes(b"\x00" * 6)
        assert session.entered.wait(2)
        socket.send_bytes(b"\x00" * 6)
        try:
            assert socket.receive_json()["code"] == "backpressure"
        finally:
            session.release.set()
    assert session.chunks == [b"\x00" * 6]


def test_capacity_is_released_after_disconnect(api):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as first:
        begin(first)
        with client.websocket_connect("/stream", headers=AUTH) as second:
            second.send_json(START)
            assert second.receive_json()["code"] == "busy"
    with client.websocket_connect("/stream", headers=AUTH) as next_socket:
        assert begin(next_socket)["type"] == "ready"


def test_inference_failure_closes_session(api):
    client, fake = api
    fake.next_session = FakeSession()
    fake.next_session.fail = True
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_bytes(b"\x00\x00")
        assert socket.receive_json()["code"] == "inference_failed"


@pytest.mark.parametrize("cancel_twice", [False, True])
def test_cancelled_inference_keeps_gpu_serialized(monkeypatch, cancel_twice):
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def first_work():
        entered.set()
        assert release.wait(5)

    async def run():
        monkeypatch.setattr(server, "gpu_lock", asyncio.Semaphore(1))
        first = asyncio.create_task(server.run_gpu(first_work))
        await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        if cancel_twice:
            await asyncio.sleep(0.02)
            first.cancel()
        second = asyncio.create_task(server.run_gpu(second_entered.set))
        await asyncio.sleep(0.03)
        assert not second_entered.is_set()
        release.set()
        with suppress(asyncio.CancelledError):
            await first
        await second
        assert second_entered.is_set()

    asyncio.run(run())


def test_duration_limit_is_separate_from_queue_capacity(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(server, "MAX_DURATION", 1 / 16000)
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_bytes(b"\x00" * 4)
        assert socket.receive_json()["code"] == "duration_limit"


def test_start_timeout(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(server, "START_TIMEOUT", 0.02)
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        assert socket.receive_json()["code"] == "start_timeout"


def test_unknown_control_does_not_become_inference_error(api):
    client, _ = api
    with client.websocket_connect("/stream", headers=AUTH) as socket:
        begin(socket)
        socket.send_json({"type": "pause"})
        assert socket.receive_json()["code"] == "invalid_control"
