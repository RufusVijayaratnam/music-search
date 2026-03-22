import torch
import torch.nn as nn
from music.common.cnn import (
    Conv1dArch,
    create_conv1d,
    create_conv1d_transpose,
    create_conv1d_transpose_arch,
)
from music.common.quantiser import AbstractQuantiser, ResidualVectorQuantiser
from music.common.transformer import Transformer, PreNormTransformerBlock
from music.common.attention import RoPEMultiHeadSelfAttention
from music.common.mlp import MlpArchitecture, create_mlp
from music.tokeniser.hyperparameters import TokeniserHP


class _Transpose12(nn.Module):
    """Swap dims 1 and 2: [B, C, L] <-> [B, L, C]"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


def create_encoder_conv_layer(arch: Conv1dArch):
    return create_conv1d(arch)


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


def create_tokeniser_encoder(hp: TokeniserHP) -> torch.nn.Module:
    conv_arch = hp.enc_conv_arch
    conv_layer = create_encoder_conv_layer(conv_arch).to(device=hp.device)

    ff_arch = MlpArchitecture(
        hidden_sizes=[conv_arch.out_channels] * hp.transformer_ff_depth,
        activation=hp.transformer_ff_activation,
    )

    transformer_layer = create_encoder_transformer_layer(
        transformer_model_dim=conv_arch.out_channels,
        n_tranformer_blocks=hp.n_transformer_blocks,
        nheads=hp.transformer_nheads,
        ff_arch=ff_arch,
        device=hp.device,
        dtype=hp.dtype,
    ).to(device=hp.device)

    return torch.nn.Sequential(conv_layer, _Transpose12(), transformer_layer)


def create_tokeniser_quantiser(hp: TokeniserHP) -> AbstractQuantiser:
    token_dim = hp.enc_conv_arch.out_channels
    simple_quantiser = ResidualVectorQuantiser(
        codebook_size=hp.codebook_size,
        token_dim=token_dim,
        num_codebooks=hp.num_codebooks,
        device=hp.device,
    )
    return simple_quantiser.to(hp.device)


def create_tokeniser_decoder(hp: TokeniserHP) -> torch.nn.Module:
    enc_conv_arch = hp.enc_conv_arch

    ff_arch = MlpArchitecture(
        hidden_sizes=[enc_conv_arch.out_channels] * hp.transformer_ff_depth,
        activation=hp.transformer_ff_activation,
    )

    transformer_layer = create_encoder_transformer_layer(
        transformer_model_dim=enc_conv_arch.out_channels,
        n_tranformer_blocks=hp.n_transformer_blocks,
        nheads=hp.transformer_nheads,
        ff_arch=ff_arch,
        device=hp.device,
        dtype=hp.dtype,
    ).to(hp.device)

    dec_conv_arch = create_conv1d_transpose_arch(enc_conv_arch)
    dec_conv_transpose = create_conv1d_transpose(dec_conv_arch).to(hp.device)

    return torch.nn.Sequential(transformer_layer, _Transpose12(), dec_conv_transpose)


def create_tokeniser_networks(
    hp: TokeniserHP,
) -> tuple[torch.nn.Module, AbstractQuantiser, torch.nn.Module]:
    encoder = create_tokeniser_encoder(hp)
    quantiser = create_tokeniser_quantiser(hp)
    decoder = create_tokeniser_decoder(hp)
    return encoder, quantiser, decoder
