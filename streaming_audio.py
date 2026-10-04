"""Bounded PCM/STFT windows; network packet boundaries do not define model steps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureWindow:
    pcm: bytes
    first_frame: int
    last_frame: int
    emitted_end: int


class PCMFeatureBuffer:
    def __init__(self, hop_samples: int, context_samples: int) -> None:
        if hop_samples <= 0 or context_samples <= 0:
            raise ValueError("Feature hop and context must be positive")
        self.hop = hop_samples
        self.context_frames = (context_samples + hop_samples - 1) // hop_samples + 1
        self.context_samples = self.context_frames * hop_samples
        self.pcm = bytearray()
        self.start_sample = 0
        self.total_samples = 0
        self.emitted_end = 0

    def append(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            raise ValueError("PCM16 frames must contain complete two-byte samples")
        self.pcm.extend(pcm)
        self.total_samples += len(pcm) // 2

    def window(self, final: bool = False) -> FeatureWindow | None:
        # Delay live features until their centered STFT windows have real right
        # context. On stop, the model preprocessor supplies its normal padding.
        end = ((self.total_samples + self.hop - 1) // self.hop if final else
               max(0, (self.total_samples - self.context_samples) // self.hop))
        if end <= self.emitted_end:
            return None
        offset = self.start_sample // self.hop
        return FeatureWindow(bytes(self.pcm), self.emitted_end - offset,
                             end - offset, end)

    def commit(self, window: FeatureWindow) -> None:
        self.emitted_end = window.emitted_end
        # Keep enough left context for the next STFT and pre-emphasis sample,
        # aligned to the feature hop so recomputing does not shift timestamps.
        keep_from = max(0, (self.emitted_end - self.context_frames) * self.hop)
        discard = max(0, keep_from - self.start_sample)
        del self.pcm[:discard * 2]
        self.start_sample += discard

    def clear(self) -> None:
        self.pcm.clear()


def step_size(value: int | list[int], first: bool) -> int:
    return int(value[0 if first else 1] if isinstance(value, (list, tuple)) else value)
