from music.tokeniser.hyperparameters import TokeniserHP
from music.common.schedule import Schedule, ConstantSchedule
from typing import Callable
from music.common.cnn import Conv1dArch, target_tps_out_channel_conv1d_arch
import torch


def make_tokeniser_hp(
    rec_loss_weight: Schedule,
    commitment_loss_weight: Schedule,
    codebook_loss_weight: Schedule,
    learning_rate: Schedule,
    n_transformer_blocks: int,
    transformer_nheads: int,
    transformer_ff_depth: int,
    transformer_ff_activation: Callable[[], torch.nn.Module],
    codebook_size: int,
    num_codebooks: int,
    enc_conv_arch: Conv1dArch,
    data_path: str,
    update_steps: int,
    batch_size: int,
    window_len: int,
    backprop_steps_per_update: int,
    grad_accumulate_steps: int,
    batch_sample_frequency: int,
    device: torch.device,
    dtype: torch.dtype,
):

    hp = TokeniserHP(
        rec_loss_weight=rec_loss_weight,
        commitment_loss_weight=commitment_loss_weight,
        codebook_loss_weight=codebook_loss_weight,
        learning_rate=learning_rate,
        n_transformer_blocks=n_transformer_blocks,
        transformer_nheads=transformer_nheads,
        transformer_ff_depth=transformer_ff_depth,
        transformer_ff_activation=transformer_ff_activation,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        enc_conv_arch=enc_conv_arch,
        data_path=data_path,
        update_steps=update_steps,
        batch_size=batch_size,
        window_len=window_len,
        backprop_steps_per_update=backprop_steps_per_update,
        grad_accumulate_steps=grad_accumulate_steps,
        batch_sample_frequency=batch_sample_frequency,
        device=device,
        dtype=dtype,
    )
    return hp


def base_exp(
    rec_loss_weight: Schedule = ConstantSchedule(start_value=1.0),
    commitment_loss_weight: Schedule = ConstantSchedule(start_value=0.25),
    codebook_loss_weight: Schedule = ConstantSchedule(start_value=1.0),
    learning_rate: Schedule = ConstantSchedule(start_value=0.001),
    n_transformer_blocks: int = 5,
    transformer_nheads: int = 8,
    transformer_ff_depth: int = 3,
    transformer_ff_activation: Callable[[], torch.nn.Module] = torch.nn.SiLU,
    codebook_size: int = 1024,
    num_codebooks: int = 12,
    enc_conv_arch: Conv1dArch = target_tps_out_channel_conv1d_arch(
        sample_rate=41000,
        target_tps=50,
        in_channels=2,
        out_channels=128,
        n_layers=3,
    ),
    data_path: str = "./data/train",
    update_steps: int = 1000,
    batch_size: int = 256,
    window_len: int = 82000,
    backprop_steps_per_update: int = 32,
    grad_accumulate_steps: int = 1,
    batch_sample_frequency: int = 20,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> TokeniserHP:
    return TokeniserHP(
        rec_loss_weight=rec_loss_weight,
        commitment_loss_weight=commitment_loss_weight,
        codebook_loss_weight=codebook_loss_weight,
        learning_rate=learning_rate,
        n_transformer_blocks=n_transformer_blocks,
        transformer_nheads=transformer_nheads,
        transformer_ff_depth=transformer_ff_depth,
        transformer_ff_activation=transformer_ff_activation,
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        enc_conv_arch=enc_conv_arch,
        data_path=data_path,
        update_steps=update_steps,
        batch_size=batch_size,
        window_len=window_len,
        backprop_steps_per_update=backprop_steps_per_update,
        grad_accumulate_steps=grad_accumulate_steps,
        batch_sample_frequency=batch_sample_frequency,
        device=device,
        dtype=dtype,
    )


def experiments():
    exp_dict = {}
    exp_dict["base"] = base_exp

    return exp_dict
