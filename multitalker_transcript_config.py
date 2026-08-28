from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MultitalkerTranscriptionConfig:
    diar_model: Optional[str] = None
    diar_pretrained_name: Optional[str] = None
    max_num_of_spks: Optional[int] = 4
    parallel_speaker_strategy: bool = True
    masked_asr: bool = False
    mask_preencode: bool = False
    cache_gating: bool = True
    cache_gating_buffer_size: int = 2
    single_speaker_mode: bool = False
    feat_len_sec: float = 0.01
    session_len_sec: float = -1
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = True
    precision: str = "bf16"
    streaming_mode: bool = True
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    diar_right_context: int = 0
    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"
    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    use_amp: bool = True
    debug_mode: bool = False
    deploy_mode: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 5
    online_normalization: bool = False
    output_path: Optional[str] = None
    diar_output_rttm_dir: Optional[str] = None
    diar_collar: float = 0.0
    diar_ignore_overlap: bool = False
    pad_and_drop_preencoded: bool = False
    generate_realtime_scripts: bool = False
    spk_supervision: str = "diar"
    binary_diar_preds: bool = True
    verbose: bool = False
    word_window: int = 50
    sent_break_sec: float = 1.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None
    ignored_initial_frame_steps: int = 5
    finetune_realtime_ratio: float = 0.01
