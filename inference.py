from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multitalker_transcript_config import MultitalkerTranscriptionConfig

LOGGER = logging.getLogger(__name__)
ASR_MODEL_NAME = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
DIAR_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def build_config(device: str, audio_file: str | None = None) -> Any:
	from omegaconf import OmegaConf

	cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
	cfg.asr_model = ASR_MODEL_NAME
	cfg.diar_pretrained_name = DIAR_MODEL_NAME
	cfg.device = device
	cfg.audio_file = audio_file
	cfg.batch_size = 1
	cfg.max_num_of_spks = 2
	cfg.deploy_mode = True
	cfg.generate_realtime_scripts = False
	cfg.output_path = None
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
		from nemo.collections.asr.parts.utils.multispk_transcribe_utils import configure_diar_streaming, validate_feature_frame_strides

		if self.device_name.startswith("cuda") and not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but no CUDA device is available")
		self.device = torch.device(self.device_name)
		self.cfg = build_config(str(self.device))
		LOGGER.info("CUDA available=%s device=%s", torch.cuda.is_available(), self.device)
		if torch.cuda.is_available():
			LOGGER.info("GPU=%s CUDA=%s", torch.cuda.get_device_name(0), torch.version.cuda)
		LOGGER.info("Loading diarization model")
		self.diar_model = SortformerEncLabelModel.from_pretrained(self.cfg.diar_pretrained_name).eval().to(self.device)
		LOGGER.info("Loading multitalker ASR model")
		self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.cfg.asr_model).eval().to(self.device)
		self.asr_model.encoder.set_default_att_context_size(att_context_size=self.cfg.att_context_size)
		validate_feature_frame_strides(self.asr_model, self.diar_model)
		diar_chunk_len = self.asr_model.encoder.streaming_cfg.valid_out_len + self.asr_model.encoder.streaming_cfg.cache_drop_size
		configure_diar_streaming(self.diar_model, self.cfg, self.asr_model.encoder.subsampling_factor, diar_chunk_len)
		if isinstance(self.diar_model.encoder.pre_encode, FeatureStacking):
			self.cfg.pad_and_drop_preencoded = True
		self.ready = True
		LOGGER.info("NeMo models initialized")

	def new_stream(self) -> "StreamingSession":
		self._require_ready()
		return StreamingSession(self)

	def transcribe_file(self, audio_path: Path) -> InferenceResult:
		self._require_ready()
		from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
		from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
		started = time.perf_counter()
		cfg = build_config(str(self.device), str(audio_path))
		cfg.pad_and_drop_preencoded = self.cfg.pad_and_drop_preencoded
		buffer = CacheAwareStreamingAudioBuffer(model=self.asr_model, online_normalization=False,
																												 pad_and_drop_preencoded=cfg.pad_and_drop_preencoded)
		buffer.append_audio_file(audio_filepath=str(audio_path), stream_id=-1)
		streamer = SpeakerTaggedASR(cfg, self.asr_model, self.diar_model)
		for step, (audio, lengths, diar_audio, diar_lengths) in enumerate(buffer.iter_with_right_context(0)):
			drop = 0 if step == 0 else self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
			streamer.perform_parallel_streaming_stt_spk(step_num=step, chunk_audio=audio, chunk_lengths=lengths,
				diar_chunk_audio=diar_audio, diar_chunk_lengths=diar_lengths,
				is_buffer_empty=buffer.is_buffer_empty(), drop_extra_pre_encoded=drop)
		raw = streamer.generate_seglst_dicts_from_parallel_streaming(samples=[{"audio_filepath": str(audio_path)}])[0]
		segments = normalize_segments(raw)
		duration = max((item["end"] for item in segments), default=0.0)
		return InferenceResult(segments, duration, time.perf_counter() - started)

	def _require_ready(self) -> None:
		if not self.ready:
			raise RuntimeError("Inference models are not initialized")


class StreamingSession:
	def __init__(self, service: InferenceService) -> None:
		import numpy as np

		self.service = service
		self.audio = np.empty(0, dtype=np.float32)
		self.streamer: Any = None
		self.buffer: Any = None
		self.step = 0
		self.cfg = build_config(str(service.device))
		self.cfg.max_num_of_spks = service.max_speakers
		self.cfg.pad_and_drop_preencoded = service.cfg.pad_and_drop_preencoded
		self.pending = b""

	def push(self, pcm_s16le: bytes) -> list[dict[str, Any]]:
		import numpy as np
		import torch
		from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
		from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
		data = self.pending + pcm_s16le
		usable_length = len(data) - (len(data) % 2)
		self.pending = data[usable_length:]
		if usable_length == 0:
			return []
		self.audio = np.concatenate((self.audio, np.frombuffer(data[:usable_length], dtype="<i2").astype(np.float32) / 32768.0))
		is_first_chunk = self.streamer is None
		if self.streamer is None:
			self.buffer = CacheAwareStreamingAudioBuffer(model=self.service.asr_model, online_normalization=False,
																												 pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded)
			self.streamer = SpeakerTaggedASR(self.cfg, self.service.asr_model, self.service.diar_model)
		self.buffer.append_audio(torch.from_numpy(self.audio), stream_id=-1 if is_first_chunk else 0)
		self.audio = np.empty(0, dtype=np.float32)
		output = []
		for audio, lengths, diar_audio, diar_lengths in self.buffer.iter_with_right_context(0):
			drop = 0 if self.step == 0 else self.service.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
			self.streamer.perform_parallel_streaming_stt_spk(step_num=self.step, chunk_audio=audio, chunk_lengths=lengths,
				diar_chunk_audio=diar_audio, diar_chunk_lengths=diar_lengths,
				is_buffer_empty=self.buffer.is_buffer_empty(), drop_extra_pre_encoded=drop)
			self.step += 1
			output = normalize_segments(self.streamer.instance_manager.batch_asr_states[0].seglsts)
		return output

	def finish(self) -> list[dict[str, Any]]:
		if self.pending:
			self.push(b"\x00")
		if self.streamer is None:
			return []
		self.push(b"\x00" * 32000)
		return normalize_segments(self.streamer.generate_seglst_dicts_from_parallel_streaming(
			samples=[{"audio_filepath": "streaming_session.wav"}])[0])


def normalize_segments(value: Any) -> list[dict[str, Any]]:
	items = value if isinstance(value, list) else value.get("segments", []) if isinstance(value, dict) else []
	return [{"speaker": str(item.get("speaker", item.get("speaker_id", "speaker_0"))),
			 "start": float(item.get("start", item.get("start_time", 0.0))),
			 "end": float(item.get("end", item.get("end_time", 0.0))), "text": str(item["text"])}
			for item in items if isinstance(item, dict) and item.get("text")]
