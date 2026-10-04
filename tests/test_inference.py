from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from inference import StreamingSession


class Features:
    def __init__(self, values):
        self.values = list(values)

    def size(self, _):
        return len(self.values)

    def __getitem__(self, key):
        return Features(self.values[key[2]])

    def clone(self):
        return Features(self.values)


class Length:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value

    def __iadd__(self, amount):
        self.value += amount
        return self

    def __isub__(self, amount):
        self.value -= amount
        return self


class ModelBuffer:
    def __init__(self, values, padded=False):
        self.buffer = Features(values)
        self.buffer_idx = 0
        self.streams_length = Length(len(values))
        self.streaming_cfg = SimpleNamespace(
            chunk_size=[12, 8], shift_size=[8, 8], pre_encode_cache_size=[0, 2])
        self.sampling_frames = [1, 1]
        self.step = 0
        self.padded = padded

    def append(self, values):
        self.buffer.values.extend(values)
        self.streams_length += len(values)

    def iter_with_right_context(self, right_context):
        assert right_context == 0
        while self.buffer_idx < len(self.buffer.values):
            first = self.buffer_idx == 0
            count = 12 if first and not self.padded else 8
            cache = 0 if first else 2
            audio = self.buffer.values[max(0, self.buffer_idx - cache):self.buffer_idx + count]
            self.buffer_idx += 8
            self.step += 1
            yield audio, len(audio), audio, len(audio)


@pytest.fixture
def fake_torch(monkeypatch):
    module = ModuleType("torch")
    module.inference_mode = nullcontext
    module.nn = SimpleNamespace(functional=SimpleNamespace(
        pad=lambda features, padding: Features(features.values + [0] * padding[1])))
    monkeypatch.setitem(__import__("sys").modules, "torch", module)


def session_for(buffer):
    session = StreamingSession.__new__(StreamingSession)
    session.buffer = buffer
    session.step = 0
    session.cfg = SimpleNamespace(pad_and_drop_preencoded=buffer.padded, diar_right_context=0)
    session.service = SimpleNamespace(
        asr_model=SimpleNamespace(encoder=SimpleNamespace(
            streaming_cfg=SimpleNamespace(drop_extra_pre_encoded=2))),
        diar_model=SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8)))
    calls = []
    session.streamer = SimpleNamespace(
        perform_parallel_streaming_stt_spk=lambda **kwargs: calls.append(kwargs))
    return session, calls


def test_live_model_waits_for_full_chunks_and_preserves_cache(fake_torch):
    buffer = ModelBuffer(range(8))
    session, calls = session_for(buffer)
    session._drain(final=False)
    assert not calls  # An 80 ms network packet is not an entire model chunk.
    buffer.append(range(8, 14))
    session._drain(final=False)
    assert len(calls) == 1
    assert calls[0]["chunk_audio"] == list(range(12))
    assert calls[0]["is_buffer_empty"] is False
    buffer.append(range(14, 18))
    session._drain(final=False)
    assert calls[1]["chunk_audio"] == list(range(6, 16))
    assert calls[1]["step_num"] == 1
    assert calls[1]["drop_extra_pre_encoded"] == 2


def test_stop_processes_short_tail_and_marks_only_last_block_final(fake_torch):
    session, calls = session_for(ModelBuffer(range(19)))
    session._drain(final=True)
    assert len(calls) == 3
    assert [call["is_buffer_empty"] for call in calls] == [False, False, True]
    assert calls[-1]["chunk_audio"] == list(range(14, 19))


def test_feature_stacking_drops_initial_padding(fake_torch):
    session, calls = session_for(ModelBuffer(range(8), padded=True))
    session._drain(final=False)
    assert len(calls) == 1
    assert calls[0]["drop_extra_pre_encoded"] == 2


def test_feature_buffer_does_not_grow_with_session_length(fake_torch):
    buffer = ModelBuffer([])
    session, calls = session_for(buffer)
    for packet in range(1500):
        buffer.append(range(packet * 8, packet * 8 + 8))
        session._drain(final=False)
        assert buffer.buffer.size(-1) <= 14
    assert len(calls) >= 1498
    assert calls[-1]["step_num"] == len(calls) - 1


@pytest.fixture
def nemo_contract(monkeypatch, fake_torch):
    import sys
    import inference

    np = pytest.importorskip("numpy")

    def stft(audio):
        # Independent centered STFT reference. Check left pre-emphasis context
        # as well as right analysis context at every packet boundary.
        signal = np.concatenate((audio[:1], audio[1:] - 0.97 * audio[:-1]))
        padded = np.pad(signal, (256, 256), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 512)[::160]
        return np.fft.rfft(frames * np.hanning(512), axis=-1)

    class NeMoBuffer(ModelBuffer):
        def __init__(self, model, online_normalization, pad_and_drop_preencoded):
            super().__init__([], pad_and_drop_preencoded)
            self.buffer = None

        def preprocess_audio(self, audio):
            # This enforces the actual NeMo NumPy input contract.
            assert isinstance(audio, np.ndarray)
            return Features(stft(audio)), None

        def append_processed_signal(self, features, stream_id=-1):
            if self.buffer is None:
                assert stream_id == -1
                self.buffer = features
            else:
                assert stream_id == 0
                self.buffer.values.extend(features.values)
            self.streams_length += len(features.values)

        def append_audio_file(self, audio_filepath, stream_id):
            import wave
            with wave.open(audio_filepath) as audio_file:
                audio = np.frombuffer(audio_file.readframes(audio_file.getnframes()),
                                      dtype="<i2").astype(np.float32) / 32768.0
            self.append_processed_signal(Features(stft(audio)), stream_id)

        def is_buffer_empty(self):
            return self.buffer_idx >= self.streams_length[0]

    class Streamer:
        def __init__(self, *args):
            self.calls = []
            self.raw = [
                {"speaker": "speaker_0", "start_time": 0.0, "end_time": 0.4, "words": "Hello."},
                {"speaker": "speaker_1", "start_time": 0.5, "end_time": 0.9, "words": "Knee pain."},
            ]
            self.instance_manager = SimpleNamespace(
                batch_asr_states=[SimpleNamespace(seglsts=self.raw)])
            self.final_calls = 0

        def perform_parallel_streaming_stt_spk(self, **kwargs):
            self.calls.append(kwargs)

        def generate_seglst_dicts_from_parallel_streaming(self, **kwargs):
            self.final_calls += 1
            return self.raw

    streaming = ModuleType("nemo.collections.asr.parts.utils.streaming_utils")
    streaming.CacheAwareStreamingAudioBuffer = NeMoBuffer
    helpers = ModuleType("nemo.collections.asr.parts.utils.multispk_transcribe_utils")
    helpers.SpeakerTaggedASR = Streamer
    monkeypatch.setitem(sys.modules, streaming.__name__, streaming)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)

    def config(*args, **kwargs):
        return SimpleNamespace(pad_and_drop_preencoded=False, diar_right_context=0)

    monkeypatch.setattr(inference, "build_config", config)
    class Config(dict):
        __getattr__ = dict.__getitem__
    model_cfg = Config(preprocessor=Config(
        window_stride=0.01, window_size=0.025, n_fft=512))
    service = inference.InferenceService(device="cpu")
    service.device = "cpu"
    service.cfg = config()
    service.ready = True
    service.asr_model = SimpleNamespace(cfg=model_cfg, encoder=SimpleNamespace(
        streaming_cfg=SimpleNamespace(drop_extra_pre_encoded=2)))
    service.diar_model = SimpleNamespace(encoder=SimpleNamespace(subsampling_factor=8))
    return np, stft, service


def test_actual_push_features_match_batch_stft_across_packet_sizes(nemo_contract):
    np, stft, service = nemo_contract
    integers = np.random.default_rng(7).integers(-30000, 30000, 16000 * 2 + 77, dtype=np.int16)
    pcm = integers.astype("<i2").tobytes()
    reference = stft(integers.astype(np.float32) / 32768.0)

    def run(packet_samples):
        session = service.new_stream()
        for offset in range(0, len(pcm), packet_samples * 2):
            session.push(pcm[offset:offset + packet_samples * 2])
        final = session.finish()
        assert len(final) == 2  # Full flat NeMo output survives normalization.
        assert session.finish() == final
        assert session.streamer.final_calls == 1
        chunks = [np.asarray(call["chunk_audio"]) for call in session.streamer.calls]
        session.close()
        assert not session.pcm.pcm
        assert session.streamer is None
        return chunks

    small, large = run(1280), run(2240)
    assert len(small) == len(large)
    for step, (a, b) in enumerate(zip(small, large)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        start = max(0, step * 8 - (0 if step == 0 else 2))
        np.testing.assert_allclose(a, reference[start:start + len(a)], rtol=1e-5, atol=1e-6)


def test_empty_stream_and_one_sample_flush(nemo_contract):
    _, _, service = nemo_contract
    empty = service.new_stream()
    assert empty.finish() == []
    short = service.new_stream()
    assert short.push(b"\x01\x00") == []
    assert short.finish()[0]["end"] <= 1 / 16000
    empty.close()
    short.close()


def test_recorded_path_preserves_all_segments_and_silent_duration(nemo_contract, tmp_path):
    import wave
    _, _, service = nemo_contract
    path = tmp_path / "recorded.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 32000)
    result = service.transcribe_file(path)
    assert [segment["text"] for segment in result.segments] == ["Hello.", "Knee pain."]
    assert result.duration == 2.0  # Last spoken word ends before the trailing silence.
