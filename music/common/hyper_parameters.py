from pydantic import BaseModel
import torch


class HyperParameters(BaseModel):
    device: torch.device
    dtype: torch.dtype
