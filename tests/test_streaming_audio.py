import math
import struct

import pytest

from inference import normalize_segments
from streaming_audio import PCMFeatureBuffer


def test_nemo_seglst_words_and_flat_output_are_preserved():
    raw = [{"speaker": "speaker_1", "start_time": 1.2, "end_time": 2.3,
            "words": " Patient reports knee pain. "}]
    assert normalize_segments(raw) == [
        {"speaker": "speaker_1", "start": 1.2, "end": 2.3, "text": "Patient reports knee pain."}]
    assert normalize_segments(raw[0]) == normalize_segments(raw)
    assert normalize_segments([]) == []


def test_model_padding_does_not_extend_timestamps():
    assert normalize_segments([{"words": "Hello", "start_time": 0, "end_time": 5}], 2)[0]["end"] == 2


@pytest.mark.parametrize("end", [math.nan, math.inf, -1])
def test_invalid_model_timestamps_are_rejected(end):
    with pytest.raises(ValueError):
        normalize_segments([{"words": "Hello", "start_time": 0, "end_time": end}])


def test_feature_windows_are_packet_independent_and_bounded():
    pcm = b"".join(struct.pack("<h", i % 32768) for i in range(16000 * 4 + 51))

    def collect(packet_samples):
        buffer = PCMFeatureBuffer(160, 256)
        windows = []
        for offset in range(0, len(pcm), packet_samples * 2):
            buffer.append(pcm[offset:offset + packet_samples * 2])
            window = buffer.window()
            if window:
                origin = buffer.start_sample // 160
                for local_frame in range(window.first_frame, window.last_frame):
                    # The complete STFT neighborhood is identical across packets.
                    center = local_frame * 160
                    neighborhood = window.pcm[max(0, center - 256) * 2:(center + 256) * 2]
                    windows.append((origin + local_frame, neighborhood))
                buffer.commit(window)
            assert len(buffer.pcm) <= (packet_samples + 2 * buffer.context_samples + 160) * 2
        final = buffer.window(final=True)
        assert final is not None
        assert final.emitted_end == (buffer.total_samples + 159) // 160
        return windows

    small = collect(1280)  # Browser's 80 ms packets.
    large = collect(3200)
    # Different boundaries can defer a different number of trailing frames.
    count = min(len(small), len(large))
    assert small[:count] == large[:count]
    assert [frame for frame, _ in small] == list(range(len(small)))


def test_short_audio_flush_and_partial_pcm_rejection():
    buffer = PCMFeatureBuffer(160, 256)
    buffer.append(b"\x00\x00")
    assert buffer.window() is None
    assert buffer.window(final=True).last_frame == 1
    with pytest.raises(ValueError):
        buffer.append(b"\x01")
