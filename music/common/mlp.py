from pydantic import BaseModel, ConfigDict
import torch.nn as nn
from typing import Callable


class MlpArchitecture(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    hidden_sizes: list[int]
    activation: Callable[[], nn.Module]


def create_mlp(input_size: int, arch: MlpArchitecture) -> nn.Module:
    layers: list[nn.Module] = []
    in_features = input_size
    for hs in arch.hidden_sizes:
        layers.append(nn.Linear(in_features, hs))
        layers.append(arch.activation())
        in_features = hs
    net = nn.Sequential(*layers)
    return net
