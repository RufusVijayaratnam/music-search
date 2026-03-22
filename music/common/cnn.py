from abc import ABC, abstractmethod
from ctypes import Union
from typing import Annotated
from pydantic import BaseModel
import torch


class Conv1DLayerArch(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True


class GroupNormArch(BaseModel):
    num_groups: int
    num_channels: int


class GELUArch(BaseModel):
    pass


class ConvTranspose1DLayerArch(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True


AnyConv1dComponent = Conv1DLayerArch | GroupNormArch | GELUArch
AnyConvTranspose1dComponent = ConvTranspose1DLayerArch | GroupNormArch | GELUArch


class Conv1dArch(BaseModel):
    layers: list[AnyConv1dComponent]

    @property
    def out_channels(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, Conv1DLayerArch):
                return layer.out_channels
        raise ValueError("No Conv1DLayerArch found in layers")


class Conv1dTransposeArch(BaseModel):
    layers: list[AnyConvTranspose1dComponent]

    @property
    def out_channels(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, ConvTranspose1DLayerArch):
                return layer.out_channels
        raise ValueError("No ConvTranspose1DLayerArch found in layers")


def create_conv1d_transpose_arch(encoder_arch: Conv1dArch) -> Conv1dTransposeArch:
    # Extract only the conv layers in reverse order
    conv_layers = [l for l in encoder_arch.layers if isinstance(l, Conv1DLayerArch)]
    layers: list[AnyConvTranspose1dComponent] = []
    reversed_convs = list(reversed(conv_layers))
    for i, enc in enumerate(reversed_convs):
        # output_padding to recover exact input length: (L_in - 1)*s - 2p + k + output_padding = L_out
        output_padding = (enc.stride - 1) % enc.stride  # 0 for most cases, handles odd strides
        layers.append(
            ConvTranspose1DLayerArch(
                in_channels=enc.out_channels,
                out_channels=enc.in_channels,
                kernel_size=enc.kernel_size,
                stride=enc.stride,
                padding=enc.padding,
                output_padding=output_padding,
                dilation=enc.dilation,
                groups=enc.groups,
                bias=enc.bias,
            )
        )
        is_last = i == len(reversed_convs) - 1
        if not is_last:
            layers.append(GroupNormArch(num_groups=8, num_channels=enc.in_channels))
            layers.append(GELUArch())
    return Conv1dTransposeArch(layers=layers)


def create_conv1d(layer_arch: Conv1dArch) -> torch.nn.Module:
    layers = []
    for arch in layer_arch.layers:
        if isinstance(arch, Conv1DLayerArch):
            layer = torch.nn.Conv1d(
                in_channels=arch.in_channels,
                out_channels=arch.out_channels,
                kernel_size=arch.kernel_size,
                stride=arch.stride,
                padding=arch.padding,
                dilation=arch.dilation,
                groups=arch.groups,
                bias=arch.bias,
            )
        elif isinstance(arch, GroupNormArch):
            layer = torch.nn.GroupNorm(num_groups=arch.num_groups, num_channels=arch.num_channels)  # type: ignore
        elif isinstance(arch, GELUArch):
            layer = torch.nn.GELU()  # type: ignore
        else:
            raise NotImplementedError()
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def create_conv1d_transpose(layer_arch: Conv1dTransposeArch) -> torch.nn.Module:
    layers = []
    for arch in layer_arch.layers:
        if isinstance(arch, ConvTranspose1DLayerArch):
            layer = torch.nn.ConvTranspose1d(
                in_channels=arch.in_channels,
                out_channels=arch.out_channels,
                kernel_size=arch.kernel_size,
                stride=arch.stride,
                padding=arch.padding,
                output_padding=arch.output_padding,
                dilation=arch.dilation,
                groups=arch.groups,
                bias=arch.bias,
            )
        elif isinstance(arch, GroupNormArch):
            layer = torch.nn.GroupNorm(num_groups=arch.num_groups, num_channels=arch.num_channels)  # type: ignore
        elif isinstance(arch, GELUArch):
            layer = torch.nn.GELU()  # type: ignore
        else:
            raise NotImplementedError()
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def target_tps_out_channel_conv1d_arch(
    sample_rate: int,
    target_tps: int,
    in_channels: int,
    out_channels: int,
    n_layers: int,
    overlap_factor: float = 2.0,
) -> Conv1dArch:
    total_stride = sample_rate // target_tps
    stride_per_layer = round(total_stride ** (1 / n_layers))
    strides = [stride_per_layer] * n_layers
    strides[-1] = total_stride // (stride_per_layer ** (n_layers - 1))

    base_channels = out_channels // (2 ** (n_layers - 1))
    if base_channels * (2 ** (n_layers - 1)) != out_channels:
        raise ValueError(
            f"out_channels {out_channels} must be base_channels * 2^(n_layers-1), got base_channels={base_channels}"
        )
    channels = [base_channels * 2**i for i in range(n_layers)]

    kernels = [round(s * overlap_factor) for s in strides]
    paddings = [(k - s) // 2 for k, s in zip(kernels, strides)]

    layers: list[AnyConv1dComponent] = []
    cur_in = in_channels
    for c, k, s, p in zip(channels, kernels, strides, paddings):
        layers.append(
            Conv1DLayerArch(in_channels=cur_in, out_channels=c, kernel_size=k, stride=s, padding=p)
        )
        layers.append(GroupNormArch(num_groups=8, num_channels=c))
        layers.append(GELUArch())
        cur_in = c
    return Conv1dArch(layers=layers)
