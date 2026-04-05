from typing import Callable
from music.common.cnn import Conv1dArch, target_tps_out_channel_conv1d_arch
from music.common.hyperparameters import HyperParameters
from music.common.schedule import ConstantSchedule, Schedule
import torch


class TokeniserHP(HyperParameters):
    rec_loss_weight: Schedule
    commitment_loss_weight: Schedule
    codebook_loss_weight: Schedule
    learning_rate: Schedule

    n_transformer_blocks: int
    transformer_nheads: int
    transformer_ff_depth: int
    transformer_ff_activation: Callable[[], torch.nn.Module]

    codebook_size: int
    num_codebooks: int

    enc_conv_arch: Conv1dArch
    data_path: str
    update_steps: int
    batch_size: int
    window_len: int
    backprop_steps_per_update: int
    grad_accumulate_steps: int
    batch_sample_frequency: int
