import torch
from music.common.transformer import Transformer, PreNormTransformerBlock
from music.common.attention import RoPEMultiHeadSelfAttention
from music.common.mlp import MlpArchitecture, create_mlp
from music.tokeniser.hyperparameters import TokeniserHP


def create_encoder_conv_layer(
    sample_rate: int,
    target_tps: int,
    n_layers: int,
    base_channels: int,
    overlap_factor: float,
    audio_channels: int,
):
    total_stride = sample_rate // target_tps
    stride_per_layer = round(total_stride ** (1 / n_layers))

    strides = [stride_per_layer] * n_layers
    strides[-1] = total_stride // (stride_per_layer ** (n_layers - 1))

    channels = [base_channels * 2**i for i in range(n_layers)]

    kernels = [round(s * overlap_factor) for s in strides]
    paddings = [(k - s) // 2 for k, s in zip(kernels, strides)]

    layers: list[torch.nn.Module] = []
    in_channels = audio_channels

    for s, c, k, p in zip(strides, channels, kernels, paddings):
        layers.append(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=c,
                kernel_size=k,
                stride=s,
                padding=p,
            )
        )
        layers.append(torch.nn.GroupNorm(num_groups=8, num_channels=c))
        layers.append(torch.nn.GELU())
        in_channels = c
    out_channels = c

    encoder = torch.nn.Sequential(*layers)
    return encoder, out_channels


def create_encoder_transformer_layer(
    transformer_model_dim: int,
    n_tranformer_blocks: int,
    nheads: int,
    ff_arch: MlpArchitecture,
    device: torch.device,
    dtype: torch.dtype,
):
    blocks = []
    for _ in range(n_tranformer_blocks):
        attn = RoPEMultiHeadSelfAttention(transformer_model_dim, nheads, True, device, dtype)
        norm1 = torch.nn.LayerNorm(transformer_model_dim)
        norm2 = torch.nn.LayerNorm(transformer_model_dim)
        ff = create_mlp(transformer_model_dim, ff_arch)
        block = PreNormTransformerBlock(attn, norm1, norm2, ff)
        blocks.append(block)
    transformer = Transformer(blocks)
    return transformer


def create_tokeniser_encoder(sample_rate: int, audio_channels, hp: TokeniserHP):
    conv_layer, conv_out_channels = create_encoder_conv_layer(
        sample_rate=sample_rate,
        target_tps=hp.target_tps,
        n_layers=hp.conv_n_layers,
        base_channels=hp.conv_n_layers,
        overlap_factor=hp.conv_overlap_factor,
        audio_channels=audio_channels,
    )

    ff_arch = MlpArchitecture(
        hidden_sizes=[conv_out_channels] * hp.transformer_ff_depth,
        activation=hp.transformer_ff_activation,
    )

    transformer_layer = create_encoder_transformer_layer(
        transformer_model_dim=conv_out_channels,
        n_tranformer_blocks=hp.n_transformer_blocks,
        nheads=hp.transformer_nheads,
        ff_arch=ff_arch,
        device=hp.device,
        dtype=hp.dtype,
    )
