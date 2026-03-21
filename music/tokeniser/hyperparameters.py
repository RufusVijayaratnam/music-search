from typing import Callable
from music.common.cnn import Conv1dArch, target_tps_out_channel_conv1d_arch
from music.common.hyper_parameters import HyperParameters
import torch


class TokeniserHP(HyperParameters):
    target_tps: int
    enc_conv_arch: Conv1dArch = target_tps_out_channel_conv1d_arch(
        sample_rate=41000, target_tps=10, in_channels=2, out_channels=512, n_layers=5
    )
    n_transformer_blocks: int
    transformer_nheads: int
    transformer_ff_depth: int
    transformer_ff_activation: Callable[[], torch.nn.Module]

    codebook_size: int
