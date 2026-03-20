from typing import Callable
from music.common.hyper_parameters import HyperParameters
import torch

from music.common.mlp import MlpArchitecture


class TokeniserHP(HyperParameters):
    target_tps: int
    conv_n_layers: int
    conv_base_channels: int
    conv_overlap_factor: float
    n_transformer_blocks: int
    transformer_nheads: int
    transformer_ff_depth: int
    transformer_ff_activation: Callable[[], torch.nn.Module]
