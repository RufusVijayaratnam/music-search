import torch

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

    encoder = torch.nn.Sequential(*layers)
    return encoder
