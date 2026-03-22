import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractTransformerBlock(nn.Module, ABC):
    def __init__(self, attention: nn.Module, norm1: nn.Module, norm2: nn.Module, ff: nn.Module):
        super().__init__()
        self.attention = attention
        self.norm1 = norm1
        self.norm2 = norm2
        self.ff = ff

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class PreNormTransformerBlock(AbstractTransformerBlock):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, blocks: list[AbstractTransformerBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, data):
        prev_out = data
        for block in self.blocks:
            prev_out = block(prev_out)
        return prev_out
