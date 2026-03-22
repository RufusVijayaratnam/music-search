from typing import Callable
from music.common.cnn import Conv1dArch, target_tps_out_channel_conv1d_arch
from music.common.hyperparameters import HyperParameters
from music.common.schedule import ConstantSchedule, Schedule
import torch


class TokeniserHP(HyperParameters):
    rec_loss_weight: Schedule = ConstantSchedule(start_value=1.0)
    commitment_loss_weight: Schedule = ConstantSchedule(start_value=0.25)
    codebook_loss_weight: Schedule = ConstantSchedule(start_value=1.0)
    learning_rate: Schedule = ConstantSchedule(start_value=0.001)

    n_transformer_blocks: int
    transformer_nheads: int
    transformer_ff_depth: int
    transformer_ff_activation: Callable[[], torch.nn.Module]

    codebook_size: int
    num_codebooks: int

    enc_conv_arch: Conv1dArch = target_tps_out_channel_conv1d_arch(
        sample_rate=41000, target_tps=50, in_channels=2, out_channels=128, n_layers=3
    )
    data_path: str = "./data/train"
    update_steps: int = 1000
    batch_size: int = 256
    window_len: int = 82000
    backprop_steps_per_update: int = 32
    grad_accumulate_steps: int = 1
    batch_sample_frequency: int = 20
