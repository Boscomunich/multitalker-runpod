from __future__ import annotations

import logging
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multitalker_transcript_config import MultitalkerTranscriptionConfig
from streaming_audio import PCMFeatureBuffer, step_size

LOGGER = logging.getLogger(__name__)
ASR_MODEL_NAME = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
DIAR_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"
SAMPLE_RATE = 16000


def build_config(device: str, audio_file: str | None = None, max_speakers: int = 2) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.asr_model = ASR_MODEL_NAME
    cfg.diar_pretrained_name = DIAR_MODEL_NAME
    cfg.device = device
    cfg.audio_file = audio_file
    cfg.batch_size = 1
    cfg.max_num_of_spks = max_speakers
    cfg.deploy_mode = True
    cfg.generate_realtime_scripts = False
    cfg.output_path = None
    cfg.print_time = False
    return cfg


@dataclass
class InferenceResult:
    segments: list[dict[str, Any]]
    duration: float
    processing_time: float

    @property
    def rtf(self) -> float | None:
        return self.processing_time / self.duration if self.duration else None


class InferenceService:
    def __init__(self, device: str = "cuda", max_speakers: int = 2) -> None:
        if not 1 <= max_speakers <= 4:
            raise ValueError("MAX_SPEAKERS must be between one and four")
        self.device_name = device
        self.max_speakers = max_speakers
        self.device: Any = None
        self.asr_model: Any = None
        self.diar_model: Any = None
        self.cfg: Any = None
        self.ready = False

    def load(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming, validate_feature_frame_strides,
        )

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        self.device = torch.device(self.device_name)
        self.cfg = build_config(str(self.device), max_speakers=self.max_speakers)
        LOGGER.info("Loading NeMo models on %s", self.device)
        self.diar_model = SortformerEncLabelModel.from_pretrained(
            self.cfg.diar_pretrained_name).eval().to(self.device)
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.cfg.asr_model).eval().to(self.device)
        self.asr_model.encoder.set_default_att_context_size(
            att_context_size=self.cfg.att_context_size)
        validate_feature_frame_strides(self.asr_model, self.diar_model)
        diar_chunk_len = (self.asr_model.encoder.streaming_cfg.valid_out_len +
                          self.asr_model.encoder.streaming_cfg.cache_drop_size)
        configure_diar_streaming(self.diar_model, self.cfg,
                                 self.asr_model.encoder.subsampling_factor, diar_chunk_len)
        self.diar_model.rttms_mask_mats = None
        if isinstance(self.diar_model.encoder.pre_encode, FeatureStacking):
            self.cfg.pad_and_drop_preencoded = True
        self.ready = True
        LOGGER.info("NeMo models initialized")

    def new_stream(self) -> "StreamingSession":
        self._require_ready()
        return StreamingSession(self)

    def transcribe_file(self, audio_path: Path) -> InferenceResult:
        self._require_ready()
        import torch
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        started = time.perf_counter()
        cfg = build_config(str(self.device), str(audio_path), self.max_speakers)
        cfg.pad_and_drop_preencoded = self.cfg.pad_and_drop_preencoded
        with wave.open(str(audio_path), "rb") as audio_file:
            duration = audio_file.getnframes() / audio_file.getframerate()
        with torch.inference_mode():
            buffer = CacheAwareStreamingAudioBuffer(
                model=self.asr_model, online_normalization=False,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded)
            buffer.append_audio_file(audio_filepath=str(audio_path), stream_id=-1)
            streamer = SpeakerTaggedASR(cfg, self.asr_model, self.diar_model)
            right_context = cfg.diar_right_context * self.diar_model.encoder.subsampling_factor
            for step, (audio, lengths, diar_audio, diar_lengths) in enumerate(
                    buffer.iter_with_right_context(right_context)):
                drop = (0 if step == 0 and not cfg.pad_and_drop_preencoded else
                        self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded)
                streamer.perform_parallel_streaming_stt_spk(
                    step_num=step, chunk_audio=audio, chunk_lengths=lengths,
                    diar_chunk_audio=diar_audio, diar_chunk_lengths=diar_lengths,
                    is_buffer_empty=buffer.is_buffer_empty(), drop_extra_pre_encoded=drop)
            # NeMo returns a flat SegLST list, not one nested list per sample.
            raw = streamer.generate_seglst_dicts_from_parallel_streaming(
                samples=[{"audio_filepath": str(audio_path)}])
            segments = normalize_segments(raw, duration)
        return InferenceResult(segments, duration, time.perf_counter() - started)

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError("Inference models are not initialized")


class StreamingSession:
    def __init__(self, service: InferenceService) -> None:
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self.service = service
        self.cfg = build_config(str(service.device), max_speakers=service.max_speakers)
        self.cfg.pad_and_drop_preencoded = service.cfg.pad_and_drop_preencoded
        # Normalize model-sized feature views, never individual network packets.
        self.buffer = CacheAwareStreamingAudioBuffer(
            model=service.asr_model, online_normalization=True,
            pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded)
        self.streamer = SpeakerTaggedASR(self.cfg, service.asr_model, service.diar_model)
        preprocessor = service.asr_model.cfg.preprocessor
        if preprocessor.get("exact_pad", False):
            raise ValueError("The streaming frontend requires centered STFT padding")
        hop = round(preprocessor.window_stride * SAMPLE_RATE)
        window = round(preprocessor.window_size * SAMPLE_RATE)
        n_fft = preprocessor.get("n_fft") or 2 ** math.ceil(math.log2(window))
        self.pcm = PCMFeatureBuffer(hop, n_fft // 2)
        self.step = 0
        self.closed = False
        self.final_segments: list[dict[str, Any]] | None = None

    def push(self, pcm_s16le: bytes) -> list[dict[str, Any]]:
        if self.closed or self.final_segments is not None:
            raise RuntimeError("Stream is closed")
        import torch

        with torch.inference_mode():
            self.pcm.append(pcm_s16le)
            self._append_features()
            self._drain(final=False)
        return self._segments()

    def _append_features(self, final: bool = False) -> None:
        import numpy as np

        window = self.pcm.window(final)
        if window is None:
            return
        audio = np.frombuffer(window.pcm, dtype="<i2").astype(np.float32) / 32768.0
        # Very short final recordings still need enough samples for reflect pad.
        if final and len(audio) <= self.pcm.context_samples:
            audio = np.pad(audio, (0, self.pcm.context_samples + 1 - len(audio)))
        # CacheAwareStreamingAudioBuffer.preprocess_audio expects NumPy, not Tensor.
        features, _ = self.buffer.preprocess_audio(audio)
        if features.size(-1) < window.last_frame:
            raise RuntimeError("Model preprocessor produced too few feature frames")
        fresh = features[:, :, window.first_frame:window.last_frame]
        self.buffer.append_processed_signal(
            fresh, stream_id=-1 if self.buffer.buffer is None else 0)
        self.pcm.commit(window)

    def _drain(self, final: bool) -> None:
        import torch

        if self.buffer.buffer is None:
            return
        valid_end = int(self.buffer.streams_length[0])
        cfg = self.buffer.streaming_cfg
        right_context = self.cfg.diar_right_context * self.service.diar_model.encoder.subsampling_factor
        while self.buffer.buffer_idx < valid_end:
            first = self.step == 0 and not self.cfg.pad_and_drop_preencoded
            required = step_size(cfg.chunk_size, first) + right_context
            available = valid_end - self.buffer.buffer_idx
            if not final and available < required:
                break
            if final and self.buffer.sampling_frames is not None:
                minimum = step_size(self.buffer.sampling_frames, self.step == 0)
                missing = minimum - available
                if missing > 0:
                    self.buffer.buffer = torch.nn.functional.pad(self.buffer.buffer, (0, missing))
                    self.buffer.streams_length += missing
            item = next(self.buffer.iter_with_right_context(right_context), None)
            if item is None:
                break
            audio, lengths, diar_audio, diar_lengths = item
            drop = (0 if first else
                    self.service.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded)
            self.streamer.perform_parallel_streaming_stt_spk(
                step_num=self.step, chunk_audio=audio, chunk_lengths=lengths,
                diar_chunk_audio=diar_audio, diar_chunk_lengths=diar_lengths,
                is_buffer_empty=final and self.buffer.buffer_idx >= valid_end,
                drop_extra_pre_encoded=drop)
            self.step += 1
        if not final:
            # NeMo's simulation buffer retains the whole feature history by
            # default. Keep only the pre-encoder cache and unprocessed features.
            keep = max(1, step_size(cfg.pre_encode_cache_size, False))
            discard = max(0, self.buffer.buffer_idx - keep)
            if discard:
                self.buffer.buffer = self.buffer.buffer[:, :, discard:].clone()
                self.buffer.buffer_idx -= discard
                self.buffer.streams_length -= discard

    def _segments(self) -> list[dict[str, Any]]:
        if self.step == 0:
            return []
        return normalize_segments(self.streamer.instance_manager.batch_asr_states[0].seglsts,
                                  self.pcm.total_samples / SAMPLE_RATE)

    def finish(self) -> list[dict[str, Any]]:
        if self.final_segments is not None:
            return self.final_segments
        if self.closed:
            raise RuntimeError("Stream is closed")
        import torch

        with torch.inference_mode():
            self._append_features(final=True)
            self._drain(final=True)
            raw = (self.streamer.generate_seglst_dicts_from_parallel_streaming(
                samples=[{"audio_filepath": "streaming_session.wav"}]) if self.step else [])
            self.final_segments = normalize_segments(raw, self.pcm.total_samples / SAMPLE_RATE)
        return self.final_segments

    def close(self) -> None:
        self.closed = True
        self.pcm.clear()
        self.buffer = None
        self.streamer = None


def normalize_segments(value: Any, duration: float | None = None) -> list[dict[str, Any]]:
    items = (value if isinstance(value, list) else value.get("segments", [value])
             if isinstance(value, dict) else [])
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("words", item.get("text", ""))
        if not isinstance(text, str) or not text.strip():
            continue
        start = float(item.get("start", item.get("start_time", 0.0)))
        end = float(item.get("end", item.get("end_time", 0.0)))
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Invalid model transcript timestamps")
        if duration is not None:
            start, end = min(start, duration), min(end, duration)
        result.append({"speaker": str(item.get("speaker", item.get("speaker_id", "speaker_0"))),
                       "start": start, "end": end, "text": text.strip()})
    return sorted(result, key=lambda segment: segment["start"])
